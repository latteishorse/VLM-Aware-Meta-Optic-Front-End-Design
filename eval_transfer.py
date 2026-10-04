"""
eval_transfer.py -- transfer without optical re-optimization
=====================================================

Evaluate an optic trained on ImageNet-100 train, **without re-optimization**,
across ImageNet-100, CIFAR-100, and Food-101 using fixed text heads or a fixed
clean-image DINOv2 probe. Only target-dataset CLIP/SigLIP transfer is zero-shot
with respect to optical optimization; ImageNet-100 labels trained the optics.

Key idea:
  - the PSF is a property of the design (depends only on rho): computed once
    per design and reused across all datasets/VLMs
  - CLIP, SigLIP: zero-shot classification (text embeddings)
  - DINOv2: linear probe (trained per dataset, cached)

Run:
  conda activate pmp
  CUDA_VISIBLE_DEVICES=0 mpirun -np 16 --bind-to socket \\
      python eval_transfer.py \\
      --datasets cifar100 food101 \\
      --vlms clip siglip dinov2 \\
      --design_paths \\
        c1       c1_output/rho_design.npy \\
        Focus_s0 c2_output/seed0/c2_final.npz \\
        Focus_s1 c2_output/seed1/c2_final.npz \\
        Focus_s2 c2_output/seed2/c2_final.npz \\
        VLMcold_s0 c3_output/100iter_seed0_rand/c3_final.npz \\
        VLMwarm_s0 c3_output/100iter_seed0/c3_final.npz \\
        VLMwarm_s1 c3_output/100iter_seed1/c3_final.npz \\
        VLMwarm_s2 c3_output/100iter_seed2/c3_final.npz \\
      2>&1 | tee d2_all.log

  # single dataset only:
  CUDA_VISIBLE_DEVICES=0 mpirun -np 16 --bind-to socket \\
      python eval_transfer.py --datasets cifar100 --vlms clip siglip dinov2 \\
      --design_paths VLMwarm_s0 c3_output/100iter_seed0/c3_final.npz \\
      2>&1 | tee d2_cifar100_smoke.log

Cache files:
  b3_output/dinov2_probe_cifar100.npz    (created by this run)
  b3_output/dinov2_probe_food101.npz     (created by this run)
  b3_output/text_emb_cifar100.pt         (CLIP, SigLIP)
  b3_output/text_emb_food101.pt          (CLIP, SigLIP)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa
from mpi4py import MPI
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MPI setup
# =============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
is_master = (rank == 0)


def mprint(*args, **kwargs):
    if is_master:
        print(*args, **kwargs, flush=True)


# =============================================================================
# Constants (same as eval_zeroshot.py)
# =============================================================================

SX = 7.0
SY = 10.0
PML_THICK = 1.0
RESOLUTION = 67
SRC_Y = SY / 2 - PML_THICK - 0.3
MON_Y = -(SY / 2 - PML_THICK - 0.3)
DESIGN_X = 5.0
DESIGN_Y = 0.6
NX = int(DESIGN_X * RESOLUTION)
NY = int(DESIGN_Y * RESOLUTION)
N_DESIGN = NX * NY

AIR = mp.Medium(index=1.0)
TIO2 = mp.Medium(index=2.4)

WAVELENGTHS = [0.450, 0.550, 0.650]
ANGLES_DEG = [-15, 0, 15]
LAM_NAMES = {0.450: "450nm", 0.550: "550nm", 0.650: "650nm"}
DECAY_BY = 1e-3
FWIDTH_FACTOR = 0.1

WAVELENGTHS_NM = [650, 550, 450]
ZONE_ANGLES = [-15, 0, 15]

EXTERNAL_GRADS = {}


# =============================================================================
# Dataset configs
# =============================================================================

DATASET_CONFIGS = {
    "imagenet100": {
        "hf_id": "clane9/imagenet-100",
        "train_split": "train",
        "val_split": "validation",
        "label_key": "label",
        "image_key": "image",
        "n_classes": 100,
        "default_val_size": 5000,
        # transform style: standard resize+crop
        "transform": "resize_crop",
        "prompt_template": "a photo of a {}",
    },
    "cifar100": {
        "hf_id": "cifar100",
        "train_split": "train",
        "val_split": "test",
        "label_key": "fine_label",        # cifar100 HF uses fine_label
        "image_key": "img",                # cifar100 HF uses 'img' not 'image'
        "n_classes": 100,
        "default_val_size": 10000,         # full test set
        "transform": "upsample",           # 32x32 -> 224
        "prompt_template": "a photo of a {}",
    },
    "food101": {
        "hf_id": "food101",
        "train_split": "train",
        "val_split": "validation",
        "label_key": "label",
        "image_key": "image",
        "n_classes": 101,
        "default_val_size": 5000,          # subset for speed
        "transform": "resize_crop",
        "prompt_template": "a photo of {}, a type of food",
    },
}


VLM_CONFIGS = {
    "clip": {
        "model_id": "openai/clip-vit-large-patch14",
        "image_size": 224,
    },
    "siglip": {
        "model_id": "google/siglip-large-patch16-256",
        "image_size": 256,
    },
    "dinov2": {
        "model_id": "facebook/dinov2-large",
        "image_size": 224,
    },
}


# =============================================================================
# Meep OptimizationProblem builders
# =============================================================================

def build_opt_for_condition(lam_um, angle_deg, tag):
    fcen = 1.0 / lam_um
    rot_angle = np.deg2rad(angle_deg)
    k_point = mp.Vector3(
        x=fcen * np.sin(rot_angle),
        y=-fcen * np.cos(rot_angle),
        z=0,
    )
    src = mp.GaussianSource(fcen, fwidth=FWIDTH_FACTOR * fcen, is_integrated=True)
    source = mp.EigenModeSource(
        src=src,
        center=mp.Vector3(0, SRC_Y, 0),
        size=mp.Vector3(x=SX, y=0, z=0),
        direction=mp.AUTOMATIC if angle_deg == 0 else mp.NO_DIRECTION,
        eig_kpoint=k_point,
        eig_band=1,
        eig_parity=mp.ODD_Z,
        eig_match_freq=True,
    )
    design_variables = mp.MaterialGrid(mp.Vector3(NX, NY, 0), AIR, TIO2)
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(DESIGN_X, DESIGN_Y, 0),
        ),
    )
    geometry = [mp.Block(
        center=design_region.center,
        size=design_region.size,
        material=design_variables,
    )]
    pml = [mp.PML(thickness=PML_THICK, direction=mp.Y)]
    sim = mp.Simulation(
        cell_size=mp.Vector3(SX, SY, 0),
        boundary_layers=pml,
        sources=[source],
        geometry=geometry,
        default_material=AIR,
        resolution=RESOLUTION,
        k_point=k_point,
    )
    ob_region = mp.Volume(
        center=mp.Vector3(0, MON_Y, 0),
        size=mp.Vector3(SX, 0, 0),
    )
    objective_arg = mpa.FourierFields(sim, ob_region, mp.Ez)

    def obj_func(ez):
        if ez.ndim == 2:
            ez_1d = ez[0]
        else:
            ez_1d = ez
        n = len(ez_1d)
        w = EXTERNAL_GRADS.get(tag)
        if w is None or len(w) != n:
            EXTERNAL_GRADS[tag] = np.ones(n, dtype=np.float64)
            w = EXTERNAL_GRADS[tag]
        return npa.sum(w * npa.abs(ez_1d) ** 2)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[obj_func],
        objective_arguments=[objective_arg],
        design_regions=[design_region],
        frequencies=[fcen],
        decay_by=DECAY_BY,
    )
    return opt


def meep_forward_only(opts, tags_ordered, rho_np):
    psfs_list = []
    for tag in tags_ordered:
        w = EXTERNAL_GRADS.get(tag)
        if w is not None:
            w[:] = 1.0
        opts[tag]([rho_np], need_gradient=False)
        ez = np.asarray(opts[tag].get_objective_arguments()[0])
        if ez.ndim == 2:
            ez = ez[0]
        psf = np.abs(ez) ** 2
        psfs_list.append(psf.astype(np.float32))
    return np.stack(psfs_list, axis=0)


# =============================================================================
# Image formation
# =============================================================================

def normalize_and_resample_psfs(psfs_raw, target_size, device):
    psfs_gpu = psfs_raw.to(device)
    sums = psfs_gpu.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    psfs_norm = psfs_gpu / sums
    psfs_resampled = F.interpolate(
        psfs_norm.unsqueeze(1),
        size=target_size,
        mode='linear',
        align_corners=True,
    ).squeeze(1)
    sums2 = psfs_resampled.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    return psfs_resampled / sums2


def apply_meta_optic(images, psfs_norm, tags_ordered):
    B, C, H, W = images.shape
    psf_idx = {}
    for i, tag in enumerate(tags_ordered):
        parts = tag.split("_ang")
        lam_nm = int(parts[0].replace("nm", ""))
        ang = int(parts[1])
        psf_idx[(lam_nm, ang)] = i

    zone_bounds = [(0, W // 3), (W // 3, 2 * W // 3), (2 * W // 3, W)]
    blurry = torch.zeros_like(images)

    for zone_i, (x_s, x_e) in enumerate(zone_bounds):
        ang = ZONE_ANGLES[zone_i]
        zone_w = x_e - x_s
        for c in range(3):
            lam_nm = WAVELENGTHS_NM[c]
            psf = psfs_norm[psf_idx[(lam_nm, ang)]]
            pl = psf.shape[0]
            img_zone = images[:, c, :, x_s:x_e]
            img_flat = img_zone.reshape(B * H, 1, zone_w)
            kernel = psf.view(1, 1, pl)
            pad = pl // 2
            out = F.conv1d(img_flat, kernel, padding=pad)
            if out.shape[-1] != zone_w:
                out = out[..., :zone_w]
            blurry[:, c, :, x_s:x_e] = out.view(B, H, zone_w)
    return blurry


# =============================================================================
# Dataset utilities
# =============================================================================

def clean_food101_name(name):
    return name.replace("_", " ")


def clean_cifar100_name(name):
    return name.replace("_", " ")


def clean_imagenet100_name(name):
    parts = name.split(" ", 1)
    if len(parts) == 2 and parts[0].startswith("n") and parts[0][1:].isdigit():
        return parts[1].replace("_", " ")
    return name.replace("_", " ")


def get_class_names(dataset_name, ds):
    """Return a list of human-readable class names ordered by label index."""
    if dataset_name == "imagenet100":
        # Try cached first
        cache_path = "b3_output/text_embeddings.pt"
        if os.path.exists(cache_path):
            try:
                emb_data = torch.load(cache_path, weights_only=False)
                if "class_names" in emb_data and len(emb_data["class_names"]) == 100:
                    return emb_data["class_names"]
            except Exception:
                pass
        label_feat = ds.features.get("label")
        if label_feat is not None and hasattr(label_feat, "names"):
            return [clean_imagenet100_name(n) for n in label_feat.names]
        raise RuntimeError("Could not extract ImageNet-100 class names")

    elif dataset_name == "cifar100":
        label_feat = ds.features.get("fine_label", ds.features.get("label"))
        if label_feat is not None and hasattr(label_feat, "names"):
            return [clean_cifar100_name(n) for n in label_feat.names]
        raise RuntimeError("Could not extract CIFAR-100 class names")

    elif dataset_name == "food101":
        label_feat = ds.features.get("label")
        if label_feat is not None and hasattr(label_feat, "names"):
            return [clean_food101_name(n) for n in label_feat.names]
        raise RuntimeError("Could not extract Food101 class names")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def make_transform(dataset_name, image_size):
    from torchvision import transforms
    cfg = DATASET_CONFIGS[dataset_name]
    style = cfg["transform"]
    if style == "resize_crop":
        resize_to = max(256, image_size + 32)
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    elif style == "upsample":
        # CIFAR-100: 32x32 -> image_size, bicubic
        return transforms.Compose([
            transforms.Resize(image_size,
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unknown transform style: {style}")


class HFDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace dataset and applies transform.

    label_key, image_key vary by dataset.
    """
    def __init__(self, ds, transform, image_key, label_key):
        self.ds = ds
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[int(idx)]
        img = s[self.image_key]
        if not hasattr(img, "convert"):
            # cifar100 returns PIL-ish but sometimes numpy
            from PIL import Image as PILImage
            if isinstance(img, np.ndarray):
                img = PILImage.fromarray(img)
        img = img.convert("RGB")
        return self.transform(img), s[self.label_key]


def load_val_data(dataset_name, image_size, batch_size=32, val_size=None,
                  num_workers=4):
    """Load validation/test split for a dataset."""
    from datasets import load_dataset

    cfg = DATASET_CONFIGS[dataset_name]
    if val_size is None:
        val_size = cfg["default_val_size"]

    mprint(f"  Loading {dataset_name} ({cfg['hf_id']}, split={cfg['val_split']}, "
           f"target size {image_size})...")
    ds = load_dataset(cfg["hf_id"], split=cfg["val_split"])
    mprint(f"  Total samples: {len(ds)}")

    # Class names BEFORE subset selection (full features still attached)
    class_names = get_class_names(dataset_name, ds)
    mprint(f"  Classes: {len(class_names)} (first 3: {class_names[:3]})")

    if val_size < len(ds):
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(ds), size=val_size, replace=False)
        ds = ds.select(sub_idx.tolist())
        mprint(f"  Using random subset: {val_size} samples (seed=42)")

    tf = make_transform(dataset_name, image_size)
    torch_ds = HFDataset(ds, tf, cfg["image_key"], cfg["label_key"])
    loader = torch.utils.data.DataLoader(
        torch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader, class_names, len(torch_ds)


def load_train_data_for_probe(dataset_name, image_size, batch_size=128,
                              max_samples=None, num_workers=8):
    """Load train split for DINOv2 linear probe training."""
    from datasets import load_dataset

    cfg = DATASET_CONFIGS[dataset_name]
    mprint(f"  Loading {dataset_name} train split for probe...")
    ds = load_dataset(cfg["hf_id"], split=cfg["train_split"])
    mprint(f"  Total train samples: {len(ds)}")

    if max_samples is not None and max_samples < len(ds):
        rng = np.random.default_rng(0)
        sub_idx = rng.choice(len(ds), size=max_samples, replace=False)
        ds = ds.select(sub_idx.tolist())
        mprint(f"  Using train subset: {max_samples} samples")

    tf = make_transform(dataset_name, image_size)
    torch_ds = HFDataset(ds, tf, cfg["image_key"], cfg["label_key"])
    loader = torch.utils.data.DataLoader(
        torch_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader, len(torch_ds)


# =============================================================================
# VLM loading + text embeddings + DINOv2 probe (per-dataset cache)
# =============================================================================

def text_emb_cache_path(dataset_name):
    if dataset_name == "imagenet100":
        return "b3_output/text_embeddings.pt"
    return f"b3_output/text_emb_{dataset_name}.pt"


def dinov2_probe_cache_path(dataset_name):
    if dataset_name == "imagenet100":
        return "b3_output/dinov2_probe.npz"
    return f"b3_output/dinov2_probe_{dataset_name}.npz"


def compute_text_embeddings_clip(model, processor, class_names, device,
                                 prompt_template):
    prompts = [prompt_template.format(c) for c in class_names]
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_emb = model.get_text_features(**inputs)
        if not isinstance(text_emb, torch.Tensor):
            if hasattr(text_emb, 'pooler_output') and text_emb.pooler_output is not None:
                text_emb = text_emb.pooler_output
            elif hasattr(text_emb, 'last_hidden_state'):
                text_emb = text_emb.last_hidden_state.mean(dim=1)
        text_emb = F.normalize(text_emb, dim=-1)
    return text_emb


def compute_text_embeddings_siglip(model, processor, class_names, device,
                                   prompt_template):
    prompts = [prompt_template.format(c) for c in class_names]
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt",
                           padding="max_length").to(device)
        text_emb = model.get_text_features(**inputs)
        if not isinstance(text_emb, torch.Tensor):
            if hasattr(text_emb, 'pooler_output') and text_emb.pooler_output is not None:
                text_emb = text_emb.pooler_output
            elif hasattr(text_emb, 'last_hidden_state'):
                text_emb = text_emb.last_hidden_state.mean(dim=1)
        text_emb = F.normalize(text_emb, dim=-1)
    return text_emb


def get_or_compute_text_emb(vlm_name, dataset_name, class_names,
                            model, processor, device):
    """Get text embeddings, using cache if available."""
    cache_path = text_emb_cache_path(dataset_name)
    cache_key = f"{vlm_name}_text_emb"

    if os.path.exists(cache_path):
        try:
            emb_data = torch.load(cache_path, weights_only=False)
            if cache_key in emb_data:
                cached_class_names = emb_data.get("class_names")
                if cached_class_names is not None and \
                   list(cached_class_names) == list(class_names):
                    mprint(f"    Cache hit: {cache_path}[{cache_key}]")
                    return emb_data[cache_key].to(device)
                else:
                    mprint(f"    Cache miss (class_names mismatch): "
                           f"{cache_path}[{cache_key}]")
        except Exception as e:
            mprint(f"    Cache read failed: {e}")

    # Compute
    cfg = DATASET_CONFIGS[dataset_name]
    prompt_template = cfg["prompt_template"]
    mprint(f"    Computing {vlm_name} text emb for {len(class_names)} classes "
           f"(prompt: '{prompt_template}')...")
    if vlm_name == "clip":
        text_emb = compute_text_embeddings_clip(model, processor, class_names,
                                                device, prompt_template)
    elif vlm_name == "siglip":
        text_emb = compute_text_embeddings_siglip(model, processor, class_names,
                                                  device, prompt_template)
    else:
        raise ValueError(f"Unknown VLM for text emb: {vlm_name}")

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        emb_data = torch.load(cache_path, weights_only=False)
    else:
        emb_data = {}
    emb_data[cache_key] = text_emb.cpu()
    emb_data["class_names"] = list(class_names)
    torch.save(emb_data, cache_path)
    mprint(f"    Saved {cache_path}[{cache_key}]: {text_emb.shape}")
    return text_emb


def train_dinov2_probe_for_dataset(dataset_name, device, n_classes,
                                   max_train_samples=None):
    """Train (or load cached) linear probe for DINOv2 on this dataset."""
    cache_path = dinov2_probe_cache_path(dataset_name)
    if os.path.exists(cache_path):
        mprint(f"    DINOv2 probe cache hit: {cache_path}")
        probe = np.load(cache_path, allow_pickle=True)
        clean_acc = float(probe["clean_val_acc"])
        mprint(f"    clean_val_acc: {clean_acc:.2f}%")
        return {
            "W": torch.tensor(probe["W"], device=device),
            "b": torch.tensor(probe["b"], device=device),
            "clean_val_acc": clean_acc,
        }

    mprint(f"    DINOv2 probe cache miss. Training new probe for {dataset_name}...")

    # Load DINOv2 backbone
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    image_size = VLM_CONFIGS["dinov2"]["image_size"]
    img_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    img_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Extract train features
    train_loader, n_train = load_train_data_for_probe(
        dataset_name, image_size, batch_size=128,
        max_samples=max_train_samples,
    )
    mprint(f"    Extracting train features ({n_train} samples)...")
    feats, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            imgs_norm = (imgs - img_mean) / img_std
            out = backbone(pixel_values=imgs_norm)
            feats.append(out.pooler_output.cpu().numpy().astype(np.float32))
            labels.append(lbls.numpy())
            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)
                mprint(f"      train batch {batch_idx+1}/{len(train_loader)} | "
                       f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s")
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    mprint(f"    feats: {feats.shape}, labels: {labels.shape}")

    # Extract val features (for sanity)
    cfg = DATASET_CONFIGS[dataset_name]
    val_loader, _, n_val = load_val_data(
        dataset_name, image_size, batch_size=128,
        val_size=cfg["default_val_size"],
    )
    mprint(f"    Extracting val features ({n_val} samples)...")
    val_feats, val_labels = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            imgs_norm = (imgs - img_mean) / img_std
            out = backbone(pixel_values=imgs_norm)
            val_feats.append(out.pooler_output.cpu().numpy().astype(np.float32))
            val_labels.append(lbls.numpy())
    val_feats = np.concatenate(val_feats)
    val_labels = np.concatenate(val_labels)
    mprint(f"    val_feats: {val_feats.shape}")

    # Free backbone before fit
    del backbone
    torch.cuda.empty_cache()

    # Fit linear probe on GPU (Adam, faster than sklearn LBFGS)
    mprint(f"    Fitting linear probe (PyTorch GPU, {n_classes} classes)...")
    X_tr = torch.tensor(feats, device=device, dtype=torch.float32)
    y_tr = torch.tensor(labels, device=device, dtype=torch.long)
    X_va = torch.tensor(val_feats, device=device, dtype=torch.float32)
    y_va = torch.tensor(val_labels, device=device, dtype=torch.long)

    feat_dim = X_tr.shape[1]
    clf = nn.Linear(feat_dim, n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    EPOCHS = 30
    FIT_BATCH = 1024
    best_val = 0.0
    best_W = None
    best_b = None

    t0 = time.time()
    n_train_actual = len(X_tr)
    for epoch in range(EPOCHS):
        clf.train()
        perm = torch.randperm(n_train_actual, device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_train_actual, FIT_BATCH):
            idx = perm[i:i+FIT_BATCH]
            logits = clf(X_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        clf.eval()
        with torch.no_grad():
            tr_acc = (clf(X_tr).argmax(-1) == y_tr).float().mean().item() * 100
            va_acc = (clf(X_va).argmax(-1) == y_va).float().mean().item() * 100
        mprint(f"      epoch {epoch+1:>2d}/{EPOCHS} | "
               f"loss={total_loss/n_batches:.4f} | "
               f"train={tr_acc:.2f}% | val={va_acc:.2f}% | "
               f"{time.time()-t0:.1f}s")

        if va_acc > best_val:
            best_val = va_acc
            best_W = clf.weight.detach().cpu().numpy().astype(np.float32).copy()
            best_b = clf.bias.detach().cpu().numpy().astype(np.float32).copy()

    mprint(f"    Best val acc: {best_val:.2f}%")

    np.savez(
        cache_path,
        W=best_W, b=best_b,
        clean_val_acc=float(best_val),
        train_acc=float(tr_acc),
        feature_dim=best_W.shape[1],
        n_classes=best_W.shape[0],
    )
    mprint(f"    Saved {cache_path}")

    return {
        "W": torch.tensor(best_W, device=device),
        "b": torch.tensor(best_b, device=device),
        "clean_val_acc": float(best_val),
    }


def load_vlm(vlm_name, device):
    """Load VLM model + processor (no text emb here, computed per-dataset)."""
    cfg = VLM_CONFIGS[vlm_name]
    model_id = cfg["model_id"]
    mprint(f"  Loading {vlm_name} ({model_id})...")
    t0 = time.time()
    if vlm_name == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
    elif vlm_name == "siglip":
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id)
    elif vlm_name == "dinov2":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id).to(device).eval()
        processor = None
    else:
        raise ValueError(f"Unknown VLM: {vlm_name}")
    for p in model.parameters():
        p.requires_grad_(False)
    mprint(f"  Loaded in {time.time()-t0:.1f}s")
    return model, processor


# =============================================================================
# Classification (per-VLM)
# =============================================================================

@torch.no_grad()
def classify(blurry, vlm_name, model, processor, text_emb_or_probe, device,
             temperature=100.0):
    """Predict class indices for a batch."""
    if vlm_name == "dinov2":
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        normalized = (blurry - mean) / std
        out = model(pixel_values=normalized)
        image_features = out.pooler_output
        W = text_emb_or_probe["W"]
        b = text_emb_or_probe["b"]
        logits = image_features @ W.T + b
        return logits.argmax(dim=-1)

    if vlm_name == "clip":
        mean = torch.tensor(processor.image_processor.image_mean,
                            device=device).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_processor.image_std,
                           device=device).view(1, 3, 1, 1)
    elif vlm_name == "siglip":
        mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    else:
        raise ValueError(f"Unknown VLM: {vlm_name}")

    normalized = (blurry - mean) / std
    image_features = model.get_image_features(pixel_values=normalized)
    if not isinstance(image_features, torch.Tensor):
        if hasattr(image_features, 'pooler_output') and \
           image_features.pooler_output is not None:
            image_features = image_features.pooler_output
        elif hasattr(image_features, 'last_hidden_state'):
            image_features = image_features.last_hidden_state.mean(dim=1)
    image_features = F.normalize(image_features, dim=-1)
    logits = temperature * (image_features @ text_emb_or_probe.T)
    return logits.argmax(dim=-1)


# =============================================================================
# Rho loader
# =============================================================================

def load_rho(path):
    if path.endswith('.npy'):
        rho = np.load(path)
    elif path.endswith('.npz'):
        d = np.load(path)
        rho = None
        for key in ['rho', 'rho_final', 'design']:
            if key in d:
                rho = d[key]
                break
        if rho is None:
            keys = list(d.keys())
            mprint(f"  Warning: no standard key in {path}, using '{keys[0]}'")
            rho = d[keys[0]]
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim == 2:
        if rho.shape == (NX, NY):
            rho_flat = rho.flatten()
        elif rho.shape == (NY, NX):
            rho_flat = rho.T.flatten()
        else:
            raise ValueError(f"Unexpected rho 2D shape: {rho.shape}")
    elif rho.ndim == 1:
        if len(rho) != N_DESIGN:
            raise ValueError(f"Unexpected rho 1D length: {len(rho)}")
        rho_flat = rho
    else:
        raise ValueError(f"Unexpected rho ndim: {rho.ndim}")
    return np.clip(rho_flat, 0.0, 1.0)


# =============================================================================
# Eval one (design, vlm, dataset) triple
# =============================================================================

def eval_design_on_dataset(psfs_np, design_name, dataset_name, vlm_name,
                            model, processor, text_emb_or_probe,
                            val_loader, image_size, device):
    if not is_master:
        return None

    psfs_torch = torch.tensor(psfs_np, dtype=torch.float32, device=device)
    psfs_norm = normalize_and_resample_psfs(psfs_torch, image_size, device)

    # Build tags ordered
    tags_ordered = []
    for lam in WAVELENGTHS:
        for ang in ANGLES_DEG:
            tags_ordered.append(f"{LAM_NAMES[lam]}_ang{ang:+d}")

    n_correct = 0
    n_total = 0
    t0 = time.time()
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        blurry = apply_meta_optic(images, psfs_norm, tags_ordered).clamp(0, 1)
        preds = classify(blurry, vlm_name, model, processor,
                         text_emb_or_probe, device)
        n_correct += (preds == labels).sum().item()
        n_total += labels.shape[0]
        if (batch_idx + 1) % 30 == 0:
            running_acc = n_correct / n_total * 100
            elapsed = time.time() - t0
            eta = elapsed / (batch_idx + 1) * (len(val_loader) - batch_idx - 1)
            mprint(f"    batch {batch_idx+1}/{len(val_loader)} | "
                   f"acc={running_acc:.2f}% ({n_correct}/{n_total}) | "
                   f"ETA {eta:.0f}s")
    acc = n_correct / n_total
    dt = time.time() - t0
    mprint(f"    Final: acc = {acc*100:.2f}% ({n_correct}/{n_total}) in {dt:.1f}s")
    return {
        "design": design_name,
        "dataset": dataset_name,
        "vlm": vlm_name,
        "acc": acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "eval_time_s": dt,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar100", "food101"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--vlms", nargs="+",
                        default=["clip", "siglip", "dinov2"],
                        choices=list(VLM_CONFIGS.keys()))
    parser.add_argument("--design_paths", nargs="+", required=True,
                        help="Pairs: --design_paths name1 path1 name2 path2 ...")
    parser.add_argument("--val_size", type=int, default=None,
                        help="Override default val size per dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="eval_output")
    parser.add_argument("--probe_train_subsample", type=int, default=None,
                        help="Subsample train set for probe (default: full)")
    args = parser.parse_args()

    # Parse design pairs
    if len(args.design_paths) % 2 != 0:
        parser.error("--design_paths requires NAME PATH pairs (an even number of values)")
    designs = {}
    for i in range(0, len(args.design_paths), 2):
        if args.design_paths[i] in designs:
            parser.error(f"Duplicate design name: {args.design_paths[i]}")
        designs[args.design_paths[i]] = args.design_paths[i+1]

    # Validate on rank 0, then report errors on every rank before Meep starts.
    validation_error = None
    if is_master:
        errors = []
        for name, path in designs.items():
            if not path.endswith(('.npy', '.npz')):
                errors.append(f"{name}: expected a .npy or .npz checkpoint ({path})")
            elif not os.path.isfile(path):
                errors.append(f"{name}: checkpoint not found ({path})")
        if errors:
            validation_error = "\n".join(errors) + "\nGenerate checkpoints first; see README.md."
    validation_error = comm.bcast(validation_error, root=0)
    if validation_error:
        parser.error(validation_error)
    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)
        os.makedirs("b3_output", exist_ok=True)

    mprint("=" * 70)
    mprint(" Cross-Dataset Cross-VLM Transfer (eval_transfer.py)")
    mprint(f" Datasets : {args.datasets}")
    mprint(f" VLMs     : {args.vlms}")
    mprint(f" Designs  : {len(designs)}")
    for n, p in designs.items():
        mprint(f"   {n:15s} -> {p}")
    mprint("=" * 70)

    if not designs:
        return

    # ---- Phase 1: Build Meep opts (once on all ranks) ----
    mprint(f"\n[Phase 1] Build Meep opts on all {mpi_size} ranks")
    t0 = time.time()
    tags_ordered = []
    opts = {}
    for lam in WAVELENGTHS:
        for ang in ANGLES_DEG:
            tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
            tags_ordered.append(tag)
            opts[tag] = build_opt_for_condition(lam, ang, tag)
    mprint(f"  Built {len(opts)} OptimizationProblems in {time.time()-t0:.1f}s")

    # ---- Phase 2: Compute PSF for each design (cached in memory) ----
    mprint(f"\n[Phase 2] Compute PSFs for {len(designs)} designs")
    psf_cache = {}
    for design_name, rho_path in designs.items():
        mprint(f"\n  -- design: {design_name} -- {rho_path}")
        if is_master:
            rho_np = load_rho(rho_path)
            mprint(f"    rho: shape={rho_np.shape}, mean={rho_np.mean():.3f}")
        else:
            rho_np = np.empty(N_DESIGN, dtype=np.float64)
        comm.Bcast(rho_np, root=0)

        t0 = time.time()
        psfs_np = meep_forward_only(opts, tags_ordered, rho_np)
        dt = time.time() - t0
        mprint(f"    Meep forward 9 conditions: {dt:.1f}s, psfs shape={psfs_np.shape}")
        psf_cache[design_name] = psfs_np

        # Save psfs for reuse
        if is_master:
            np.savez(
                f"{args.out_dir}/d2_{design_name}_psfs.npz",
                psfs=psfs_np,
            )

    # Master-only from here
    if not is_master:
        comm.Barrier()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mprint(f"\n[Phase 3] Eval on {device}")

    all_results = []
    t_start = time.time()

    # ---- Phase 3: For each dataset, prepare text_emb/probe, then eval all (vlm, design) ----
    for dataset_name in args.datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        n_classes = cfg["n_classes"]
        mprint(f"\n{'='*70}")
        mprint(f" Dataset: {dataset_name} (n_classes={n_classes}, "
               f"prompt='{cfg['prompt_template']}')")
        mprint(f"{'='*70}")

        # For each VLM, load model + setup text_emb/probe + eval all designs
        for vlm_name in args.vlms:
            mprint(f"\n  --- VLM: {vlm_name} on {dataset_name} ---")

            image_size = VLM_CONFIGS[vlm_name]["image_size"]
            val_loader, class_names, val_size_actual = load_val_data(
                dataset_name, image_size,
                batch_size=args.batch_size,
                val_size=args.val_size,
            )
            mprint(f"  val_size: {val_size_actual}")

            # Setup VLM-specific eval input (text_emb or probe)
            if vlm_name in ("clip", "siglip"):
                model, processor = load_vlm(vlm_name, device)
                text_emb_or_probe = get_or_compute_text_emb(
                    vlm_name, dataset_name, class_names,
                    model, processor, device,
                )
            elif vlm_name == "dinov2":
                probe = train_dinov2_probe_for_dataset(
                    dataset_name, device, n_classes,
                    max_train_samples=args.probe_train_subsample,
                )
                # Load backbone for inference
                model, processor = load_vlm(vlm_name, device)
                text_emb_or_probe = probe
            else:
                continue

            # Eval each design on this (dataset, vlm)
            for design_name, psfs_np in psf_cache.items():
                mprint(f"\n  eval: {design_name} @ {dataset_name} @ {vlm_name}")
                result = eval_design_on_dataset(
                    psfs_np, design_name, dataset_name, vlm_name,
                    model, processor, text_emb_or_probe,
                    val_loader, image_size, device,
                )
                if result is not None:
                    all_results.append(result)
                    # Append-write to streaming JSON
                    with open(f"{args.out_dir}/d2_streaming.jsonl", "a") as f:
                        f.write(json.dumps(result) + "\n")

            # Free VLM
            del model
            if processor is not None:
                del processor
            torch.cuda.empty_cache()

    # ---- Final summary ----
    t_total = time.time() - t_start
    mprint(f"\n{'='*70}")
    mprint(f" SUMMARY · all results")
    mprint(f"{'='*70}")
    mprint(f"  {'Design':<12} {'Dataset':<12} {'VLM':<10} {'Acc':>8} {'N_correct':>10}")
    for r in all_results:
        mprint(f"  {r['design']:<12} {r['dataset']:<12} {r['vlm']:<10} "
               f"{r['acc']*100:>7.2f}% {r['n_correct']:>10d}")
    mprint(f"\n  Total time: {t_total/60:.1f} min")

    # Aggregate JSON
    summary = {"results": all_results}
    summary["_meta"] = {
        "datasets": args.datasets,
        "vlms": args.vlms,
        "n_designs": len(designs),
        "mpi_ranks": mpi_size,
        "total_time_min": t_total / 60,
    }
    out_path = f"{args.out_dir}/d2_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    mprint(f"\n  Results: {out_path}")

    # Pivot table (design x dataset x vlm)
    mprint(f"\n  Pivot (design × dataset × vlm):")
    for design_name in designs:
        for dataset_name in args.datasets:
            row = []
            for vlm_name in args.vlms:
                match = [r for r in all_results
                         if r['design'] == design_name and
                            r['dataset'] == dataset_name and
                            r['vlm'] == vlm_name]
                if match:
                    row.append(f"{vlm_name}={match[0]['acc']*100:.2f}%")
                else:
                    row.append(f"{vlm_name}=---")
            mprint(f"    {design_name:<12} | {dataset_name:<12} | {' | '.join(row)}")

    # Match the non-master wait after the shared Meep forward solves.
    comm.Barrier()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Rank-0 model/data failures must not leave workers in MPI collectives.
        if mpi_size > 1:
            import traceback
            traceback.print_exc(file=sys.stderr)
            comm.Abort(1)
        raise
