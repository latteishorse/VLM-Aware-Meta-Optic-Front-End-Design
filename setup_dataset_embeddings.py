"""
B3 -- ImageNet-100 Pipeline
===========================

Purpose:
  Load ImageNet-100 val data and prepare VLM prompt embeddings for 100 classes.
  Used to measure clean-image baseline accuracy.

Components:
  [1] Dataset: HuggingFace clane9/imagenet-100
  [2] Class names: ImageNet synset -> human-readable name
  [3] Prompts: "a photo of a {class_name}"
  [4] Precompute text embeddings (CLIP, SigLIP)
  [5] 100-image smoke test: clean-image classification accuracy

Run:
  CUDA_VISIBLE_DEVICES=0 python setup_dataset_embeddings.py
  
  # to evaluate the full val set (5k)
  CUDA_VISIBLE_DEVICES=0 python setup_dataset_embeddings.py --full

Estimated time:
  - first run: ImageNet-100 download (~13GB, 30-60 min)
  - after caching: smoke test (100 imgs) ~2 min, full (5k) ~10 min
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "b3_output"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. Dataset: ImageNet-100
# =============================================================================

def load_imagenet100(split="validation", cache_dir=None):
    """
    Load ImageNet-100 from HuggingFace datasets.
    
    Tian et al. ECCV 2020 standard split, 100 classes.
    Val: 5,000 images (50 per class).
    """
    from datasets import load_dataset
    
    print(f"  Loading clane9/imagenet-100 ({split})...")
    t0 = time.time()
    ds = load_dataset("clane9/imagenet-100", split=split, cache_dir=cache_dir)
    print(f"  Loaded {len(ds)} samples in {time.time()-t0:.1f}s")
    print(f"  Features: {list(ds.features.keys())}")
    return ds


def get_class_names(ds):
    """
    Extract the class-index -> human-readable name mapping from the dataset.
    """
    if hasattr(ds.features['label'], 'names'):
        return ds.features['label'].names
    else:
        # Fallback: synset IDs
        return [f"class_{i}" for i in range(100)]


# =============================================================================
# 2. Prompts + text embeddings
# =============================================================================

def build_prompts(class_names, template="a photo of a {}"):
    """
    Class name -> text prompt.
    
    Args:
        class_names: list of str, e.g. ["dog", "cat", ...]
        template: format string with {} for class name
    """
    prompts = [template.format(name.replace("_", " ")) for name in class_names]
    return prompts


def compute_clip_text_embeddings(clip_model, clip_proc, prompts, device=DEVICE):
    """
    Compute 100 prompt embeddings with the CLIP text encoder (once).
    
    Returns:
        embeddings: (100, feature_dim) tensor, L2-normalized
    """
    print(f"  Computing CLIP text embeddings for {len(prompts)} prompts...")
    t0 = time.time()
    
    with torch.no_grad():
        inputs = clip_proc(text=prompts, return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs)
        # returns a tensor or a ModelOutput depending on the transformers version
        if not isinstance(text_features, torch.Tensor):
            # if it is a BaseModelOutput, pull from pooler_output or last_hidden_state
            if hasattr(text_features, 'pooler_output') and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                # take a CLS-like token (first token or mean)
                text_features = text_features.last_hidden_state.mean(dim=1)
            else:
                # Dict like
                text_features = text_features[0] if isinstance(text_features, tuple) else text_features
        text_features = F.normalize(text_features, dim=-1)
    
    print(f"  Text embeddings: {text_features.shape} in {time.time()-t0:.1f}s")
    return text_features


def compute_siglip_text_embeddings(siglip_model, siglip_proc, prompts, device=DEVICE):
    """SigLIP text embeddings."""
    print(f"  Computing SigLIP text embeddings for {len(prompts)} prompts...")
    t0 = time.time()
    
    with torch.no_grad():
        inputs = siglip_proc(text=prompts, return_tensors="pt", padding="max_length").to(device)
        text_features = siglip_model.get_text_features(**inputs)
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, 'pooler_output') and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif hasattr(text_features, 'last_hidden_state'):
                text_features = text_features.last_hidden_state.mean(dim=1)
            else:
                text_features = text_features[0] if isinstance(text_features, tuple) else text_features
        text_features = F.normalize(text_features, dim=-1)
    
    print(f"  SigLIP text embeddings: {text_features.shape} in {time.time()-t0:.1f}s")
    return text_features


# =============================================================================
# 3. Classification
# =============================================================================

def classify_with_vlm(images, vlm_model, vlm_proc, text_embeddings, 
                     vlm_name="clip", device=DEVICE, batch_size=32):
    """
    Classify an image batch with the VLM.
    
    Args:
        images: list of PIL Images OR (N, 3, H, W) tensor
        vlm_model, vlm_proc: CLIP or SigLIP model + processor
        text_embeddings: (100, D) pre-computed class embeddings
        vlm_name: 'clip' or 'siglip'
    
    Returns:
        preds: (N,) predicted class indices
        probs: (N, 100) probabilities
    """
    all_probs = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        with torch.no_grad():
            if isinstance(batch[0], Image.Image):
                inputs = vlm_proc(images=batch, return_tensors="pt").to(device)
            else:
                # Already tensor
                inputs = {"pixel_values": torch.stack(batch).to(device)}
            
            image_features = vlm_model.get_image_features(**inputs)
            # handle ModelOutput
            if not isinstance(image_features, torch.Tensor):
                if hasattr(image_features, 'pooler_output') and image_features.pooler_output is not None:
                    image_features = image_features.pooler_output
                elif hasattr(image_features, 'last_hidden_state'):
                    image_features = image_features.last_hidden_state.mean(dim=1)
                else:
                    image_features = image_features[0] if isinstance(image_features, tuple) else image_features
            image_features = F.normalize(image_features, dim=-1)
            
            logits = image_features @ text_embeddings.T   # (B, 100)
            
            if vlm_name == "clip":
                # temperature scaling matters for CLIP but not for argmax
                probs = logits.softmax(dim=-1)
            elif vlm_name == "siglip":
                probs = torch.sigmoid(logits)
            else:
                probs = logits.softmax(dim=-1)
            
            all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)
    preds = all_probs.argmax(dim=-1)
    return preds.numpy(), all_probs.numpy()


# =============================================================================
# 4. Main pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Use full 5k val set (default: 100 sample smoke test)")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of samples for smoke test")
    parser.add_argument("--skip_siglip", action="store_true",
                        help="Skip SigLIP (faster for smoke test)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Dataset cache directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" B3 -- ImageNet-100 Pipeline")
    print("=" * 60)
    print(f"  Device        : {DEVICE}")
    print(f"  Mode          : {'FULL val (5k)' if args.full else f'SMOKE ({args.n_samples})'}")
    
    # --- Load dataset ---
    print(f"\n[1. Dataset]")
    ds = load_imagenet100(split="validation", cache_dir=args.cache_dir)
    class_names = get_class_names(ds)
    print(f"  # classes: {len(class_names)}")
    print(f"  First 5  : {class_names[:5]}")
    
    # --- Build prompts ---
    print(f"\n[2. Prompts]")
    prompts = build_prompts(class_names)
    print(f"  Template : 'a photo of a {{class}}'")
    print(f"  Sample   : '{prompts[0]}'")
    print(f"  Total    : {len(prompts)} prompts")
    
    # --- Load VLMs ---
    print(f"\n[3. Load VLMs]")
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
    
    t0 = time.time()
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()
    print(f"  CLIP loaded in {time.time()-t0:.1f}s")
    
    if not args.skip_siglip:
        t0 = time.time()
        siglip_model = AutoModel.from_pretrained("google/siglip-large-patch16-256").to(DEVICE)
        siglip_proc = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")
        siglip_model.eval()
        print(f"  SigLIP loaded in {time.time()-t0:.1f}s")
    else:
        siglip_model = None
    
    # --- Compute text embeddings (once) ---
    print(f"\n[4. Text embeddings]")
    clip_text_emb = compute_clip_text_embeddings(clip_model, clip_proc, prompts)
    
    if siglip_model is not None:
        siglip_text_emb = compute_siglip_text_embeddings(siglip_model, siglip_proc, prompts)
    
    # Save fixed text embeddings for CODA reuse
    torch.save({
        'clip_text_emb': clip_text_emb.cpu(),
        'siglip_text_emb': siglip_text_emb.cpu() if siglip_model else None,
        'prompts': prompts,
        'class_names': class_names,
    }, os.path.join(OUT_DIR, "text_embeddings.pt"))
    print(f"  Saved embeddings to {OUT_DIR}/text_embeddings.pt")
    
    # --- Prepare samples ---
    if args.full:
        sample_indices = list(range(len(ds)))
    else:
        # stratified: n_samples/100 per class
        n_per_class = max(1, args.n_samples // 100)
        # HF datasets index access is O(1)
        labels = np.array(ds['label'])
        sample_indices = []
        for c in range(100):
            class_idx = np.where(labels == c)[0][:n_per_class]
            sample_indices.extend(class_idx.tolist())
        sample_indices = sample_indices[:args.n_samples]
    
    n_samples = len(sample_indices)
    print(f"\n[5. Evaluation]")
    print(f"  Samples: {n_samples}")
    
    # --- CLIP classification on clean images ---
    print(f"\n  CLIP (clean image baseline):")
    t0 = time.time()
    
    # Load images
    images = []
    labels = []
    for idx in sample_indices:
        sample = ds[int(idx)]
        images.append(sample['image'].convert('RGB'))
        labels.append(sample['label'])
    labels = np.array(labels)
    print(f"    Loaded {len(images)} images in {time.time()-t0:.1f}s")
    
    # Classify
    t0 = time.time()
    preds, probs = classify_with_vlm(images, clip_model, clip_proc, 
                                     clip_text_emb, vlm_name="clip")
    clip_acc = (preds == labels).mean()
    clip_top5 = np.mean([labels[i] in probs[i].argsort()[-5:] for i in range(len(labels))])
    print(f"    Inference time: {time.time()-t0:.1f}s")
    print(f"    Top-1 accuracy: {clip_acc*100:.2f}% ({(preds == labels).sum()}/{len(labels)})")
    print(f"    Top-5 accuracy: {clip_top5*100:.2f}%")
    
    results = {
        "clip_top1": float(clip_acc),
        "clip_top5": float(clip_top5),
        "n_samples": n_samples,
    }
    
    # --- SigLIP classification ---
    if siglip_model is not None:
        print(f"\n  SigLIP (clean image baseline):")
        t0 = time.time()
        preds, probs = classify_with_vlm(images, siglip_model, siglip_proc,
                                         siglip_text_emb, vlm_name="siglip")
        siglip_acc = (preds == labels).mean()
        siglip_top5 = np.mean([labels[i] in probs[i].argsort()[-5:] for i in range(len(labels))])
        print(f"    Inference time: {time.time()-t0:.1f}s")
        print(f"    Top-1 accuracy: {siglip_acc*100:.2f}%")
        print(f"    Top-5 accuracy: {siglip_top5*100:.2f}%")
        
        results["siglip_top1"] = float(siglip_acc)
        results["siglip_top5"] = float(siglip_top5)
    
    # --- Save results ---
    with open(os.path.join(OUT_DIR, "clean_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"  Samples evaluated    : {n_samples}")
    print(f"  CLIP top-1           : {clip_acc*100:.2f}%")
    print(f"  CLIP top-5           : {clip_top5*100:.2f}%")
    if siglip_model is not None:
        print(f"  SigLIP top-1         : {siglip_acc*100:.2f}%")
        print(f"  SigLIP top-5         : {siglip_top5*100:.2f}%")
    
    # GPU memory summary
    mem = torch.cuda.memory_allocated() / 1e9
    max_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n  GPU memory (now/peak): {mem:.2f} / {max_mem:.2f} GB")
    
    print(f"\n  Outputs:")
    print(f"    {OUT_DIR}/text_embeddings.pt  -- precomputed class embeddings (reused by CODA)")
    print(f"    {OUT_DIR}/clean_baseline.json -- clean-image reference accuracy")
    
    print(f"\n  Clean-image accuracy is a reference, not a theoretical upper bound.")


if __name__ == "__main__":
    main()
