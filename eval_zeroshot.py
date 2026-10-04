"""
eval_zeroshot.py -- CLIP frozen-encoder evaluation for optical front-ends
====================================================================

Evaluates Fresnel, Focus-opt, VLM-cold, VLM-warm, or arbitrary design paths on
5,000 ImageNet-100 validation images with frozen CLIP ViT-L/14 text prompts.
The optics use ImageNet-100 training labels; the complete system is not zero-shot.

Pipeline:
  load rho -> 9 Meep forwards -> unit-sum PSF line-scan image formation
  -> frozen CLIP classification -> top-1 accuracy

Run:
  conda activate pmp
  mpirun -np 16 python eval_zeroshot.py --designs c1 c2_seed0 c2_seed1 c2_seed2
  mpirun -np 16 python eval_zeroshot.py --design_paths VLMwarm_s0 c3_output/100iter_seed0/c3_final.npz
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
import torch.nn.functional as F


# =============================================================================
# MPI setup
# =============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_size = comm.Get_size()
is_master = (rank == 0)


# =============================================================================
# Constants (same as design_codesign_vlm.py)
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

IMG_SIZE = 224
WAVELENGTHS_NM = [650, 550, 450]   # R, G, B
ZONE_ANGLES = [-15, 0, 15]

# Forward only -- external_grad is always 1
EXTERNAL_GRADS = {}


# =============================================================================
# Meep OptimizationProblem (same as design_codesign_vlm.py)
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
    """All ranks participate. PSF extraction (need_gradient=False)."""
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
# Image formation (same as design_codesign_vlm.py)
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
# CLIP forward (eval mode)
# =============================================================================

@torch.no_grad()
def clip_predict(blurry, clip_model, clip_proc, text_emb, device, temperature=100.0):
    """Returns: predicted class indices (B,)."""
    mean = torch.tensor(
        clip_proc.image_processor.image_mean, device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        clip_proc.image_processor.image_std, device=device
    ).view(1, 3, 1, 1)
    normalized = (blurry - mean) / std

    image_features = clip_model.get_image_features(pixel_values=normalized)

    if not isinstance(image_features, torch.Tensor):
        if hasattr(image_features, 'pooler_output') and image_features.pooler_output is not None:
            image_features = image_features.pooler_output
        elif hasattr(image_features, 'last_hidden_state'):
            image_features = image_features.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(f"Unexpected image_features type: {type(image_features)}")

    image_features = F.normalize(image_features, dim=-1)
    logits = temperature * (image_features @ text_emb.T)
    return logits.argmax(dim=-1)


# =============================================================================
# Data: ImageNet-100 val
# =============================================================================

def load_val_data(device, batch_size=32, val_size=5000):
    """ImageNet-100 val + CLIP text embeddings."""
    from datasets import load_dataset
    from torchvision import transforms

    print(f"  Loading ImageNet-100 val...")
    ds = load_dataset("clane9/imagenet-100", split="validation")
    print(f"  Total val samples: {len(ds)}")

    # diversity sanity check
    rng = np.random.default_rng(0)
    check_idx = rng.choice(len(ds), size=min(200, len(ds)), replace=False)
    check_labels = [ds[int(i)]['label'] for i in check_idx]
    n_unique = len(set(check_labels))
    print(f"  Label diversity (200 random): {n_unique} unique classes")

    if val_size < len(ds):
        # random subset (full val 5000 is normally used as-is)
        rng2 = np.random.default_rng(42)
        sub_idx = rng2.choice(len(ds), size=val_size, replace=False)
        ds = ds.select(sub_idx.tolist())
        print(f"  Using random subset: {val_size} samples")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, ds, transform):
            self.ds, self.transform = ds, transform
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            s = self.ds[int(idx)]
            return self.transform(s['image'].convert('RGB')), s['label']

    torch_ds = HFDataset(ds, transform)
    loader = torch.utils.data.DataLoader(
        torch_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    emb_data = torch.load("b3_output/text_embeddings.pt", map_location="cpu", weights_only=False)
    text_emb = emb_data['clip_text_emb'].to(device)
    print(f"  Text embeddings: {text_emb.shape}")

    return loader, text_emb, len(torch_ds)


# =============================================================================
# Rho loader (supports multiple formats)
# =============================================================================

def load_rho(path):
    """
    Load rho from .npy or .npz file.
    Returns: (N_DESIGN,) float64 numpy array.
    """
    if path.endswith('.npy'):
        rho = np.load(path)
    elif path.endswith('.npz'):
        d = np.load(path)
        # candidate keys
        for key in ['rho', 'rho_final', 'design']:
            if key in d:
                rho = d[key]
                break
        else:
            # use the first key
            keys = list(d.keys())
            print(f"  Warning: no standard key found in {path}, using '{keys[0]}'")
            rho = d[keys[0]]
    else:
        raise ValueError(f"Unsupported file extension: {path}")

    rho = np.asarray(rho, dtype=np.float64)

    # shape check and flatten
    if rho.ndim == 2:
        if rho.shape == (NX, NY):
            rho_flat = rho.flatten()
        elif rho.shape == (NY, NX):
            rho_flat = rho.T.flatten()
        else:
            raise ValueError(f"Unexpected rho 2D shape: {rho.shape}, expected ({NX},{NY})")
    elif rho.ndim == 1:
        if len(rho) != N_DESIGN:
            raise ValueError(f"Unexpected rho 1D length: {len(rho)}, expected {N_DESIGN}")
        rho_flat = rho
    else:
        raise ValueError(f"Unexpected rho ndim: {rho.ndim}")

    return rho_flat


# =============================================================================
# Single design evaluation
# =============================================================================

def evaluate_design(design_name, rho_path, opts, tags_ordered,
                    val_loader, val_size, clip_model, clip_proc,
                    text_emb, device):
    """Evaluate a single design. All ranks participate."""
    def mprint(*a, **k):
        if is_master:
            print(*a, **k, flush=True)

    mprint(f"\n{'='*60}")
    mprint(f" Evaluating: {design_name}")
    mprint(f" Rho path  : {rho_path}")
    mprint(f"{'='*60}")

    # --- 1. Load rho (rank 0) and broadcast ---
    if is_master:
        rho_np = load_rho(rho_path)
        rho_np = np.clip(rho_np, 0.0, 1.0)
        mprint(f"  Loaded rho: shape={rho_np.shape}, "
               f"mean={rho_np.mean():.3f}, "
               f"min={rho_np.min():.3f}, max={rho_np.max():.3f}")
    else:
        rho_np = np.empty(N_DESIGN, dtype=np.float64)
    comm.Bcast(rho_np, root=0)

    # --- 2. 9 Meep forwards (all ranks) ---
    mprint(f"\n  [Meep forward 9 conditions]")
    t0 = time.time()
    psfs_np = meep_forward_only(opts, tags_ordered, rho_np)
    n_focal = psfs_np.shape[1]
    dt_meep = time.time() - t0
    mprint(f"  Done: {dt_meep:.1f}s ({dt_meep/9:.1f}s/condition)")
    mprint(f"  PSF shape: {psfs_np.shape}, "
           f"sum range: [{psfs_np.sum(axis=-1).min():.3e}, "
           f"{psfs_np.sum(axis=-1).max():.3e}]")

    # --- 3. CLIP eval (rank 0 only) ---
    if is_master:
        mprint(f"\n  [CLIP frozen-encoder classification on {val_size} val images]")
        psfs_torch = torch.tensor(psfs_np, dtype=torch.float32, device=device)
        psfs_norm = normalize_and_resample_psfs(psfs_torch, IMG_SIZE, device)

        n_correct = 0
        n_total = 0
        t0 = time.time()
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            blurry = apply_meta_optic(images, psfs_norm, tags_ordered).clamp(0, 1)
            preds = clip_predict(blurry, clip_model, clip_proc, text_emb, device)

            n_correct += (preds == labels).sum().item()
            n_total += labels.shape[0]

            if (batch_idx + 1) % 20 == 0:
                running_acc = n_correct / n_total * 100
                elapsed = time.time() - t0
                eta = elapsed / (batch_idx + 1) * (len(val_loader) - batch_idx - 1)
                mprint(f"    batch {batch_idx+1}/{len(val_loader)} | "
                       f"acc={running_acc:.2f}% ({n_correct}/{n_total}) | "
                       f"ETA {eta:.0f}s")

        acc = n_correct / n_total
        dt_eval = time.time() - t0
        mprint(f"\n  Final: acc = {acc*100:.2f}% "
               f"({n_correct}/{n_total}) in {dt_eval:.1f}s")
    else:
        acc = 0.0
        n_correct = 0
        n_total = 0
        dt_eval = 0.0

    # Sync acc to all ranks
    acc = comm.bcast(acc, root=0)
    n_correct = comm.bcast(n_correct, root=0)
    n_total = comm.bcast(n_total, root=0)

    return {
        "design": design_name,
        "rho_path": rho_path,
        "acc": acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "psfs": psfs_np,
        "meep_time_s": dt_meep,
        "eval_time_s": dt_eval,
    }


# =============================================================================
# Main
# =============================================================================

# Default design list (editable)
DEFAULT_DESIGNS = {
    "c1":       "c1_output/rho_design.npy",
    "c2_seed0": "c2_output/seed0/c2_final.npz",
    "c2_seed1": "c2_output/seed1/c2_final.npz",
    "c2_seed2": "c2_output/seed2/c2_final.npz",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--designs", nargs="+", default=None,
                        help="Subset of designs to evaluate (default: all). "
                             "Use legacy checkpoint keys like 'c1 c2_seed1', or pass paper-facing custom names with --design_paths")
    parser.add_argument("--val_size", type=int, default=5000,
                        help="Number of val images to use (default: 5000=full)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="CLIP batch size (default: 32)")
    parser.add_argument("--out_dir", default="eval_output",
                        help="Output directory")
    parser.add_argument("--design_paths", nargs="+", default=None,
                        help="Custom (name, path) pairs e.g. "
                             "--design_paths VLMwarm_s0 c3_output/100iter_seed0/c3_final.npz")
    args = parser.parse_args()

    def mprint(*a, **k):
        if is_master:
            print(*a, **k, flush=True)

    # Build design dict
    if args.design_paths:
        if len(args.design_paths) % 2 != 0:
            parser.error("--design_paths requires NAME PATH pairs (an even number of values)")
        designs = {}
        for i in range(0, len(args.design_paths), 2):
            if args.design_paths[i] in designs:
                parser.error(f"Duplicate design name: {args.design_paths[i]}")
            designs[args.design_paths[i]] = args.design_paths[i+1]
    else:
        designs = DEFAULT_DESIGNS.copy()

    if args.designs:
        unknown = sorted(set(args.designs) - set(designs))
        if unknown:
            parser.error(f"Unknown design names: {', '.join(unknown)}. "
                         f"Available: {', '.join(designs)}")
        designs = {k: v for k, v in designs.items() if k in args.designs}

    # Validate on rank 0, then report errors on every rank before Meep starts.
    validation_error = None
    if is_master:
        valid_designs = {}
        errors = []
        for name, path in designs.items():
            if not path.endswith(('.npy', '.npz')):
                errors.append(f"{name}: expected a .npy or .npz checkpoint ({path})")
            elif os.path.isfile(path):
                valid_designs[name] = path
            elif args.design_paths or args.designs:
                errors.append(f"{name}: checkpoint not found ({path})")
            else:
                mprint(f"  ⚠ Skipping {name}: file not found ({path})")
        designs = valid_designs
        if not designs and not errors:
            errors.append("No design checkpoints found. Generate them first; see README.md.")
        if not os.path.isfile("b3_output/text_embeddings.pt"):
            errors.append("Missing b3_output/text_embeddings.pt. Run python setup_dataset_embeddings.py --full first.")
        if errors:
            validation_error = "\n".join(errors)
    validation_error = comm.bcast(validation_error, root=0)
    if validation_error:
        parser.error(validation_error)
    designs_keys = comm.bcast(list(designs.keys()), root=0)
    designs_vals = comm.bcast(list(designs.values()), root=0)
    designs = dict(zip(designs_keys, designs_vals))
    if is_master:
        os.makedirs(args.out_dir, exist_ok=True)

    mprint("=" * 60)
    mprint(" CLIP frozen-encoder evaluation for optical front-ends")
    mprint("=" * 60)
    mprint(f"  MPI ranks   : {mpi_size}")
    mprint(f"  Val size    : {args.val_size}")
    mprint(f"  Batch size  : {args.batch_size}")
    mprint(f"  Designs ({len(designs)}):")
    for name, path in designs.items():
        mprint(f"    {name:12s} -> {path}")
    mprint(f"  Output      : {args.out_dir}/")

    if not designs:
        mprint("  No valid designs found. Exiting.")
        return

    # --- 1. Build Meep opts (all ranks) ---
    mprint(f"\n[1. Build Meep opts on all {mpi_size} ranks]")
    t0 = time.time()
    tags_ordered = []
    opts = {}
    for lam in WAVELENGTHS:
        for ang in ANGLES_DEG:
            tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
            tags_ordered.append(tag)
            opts[tag] = build_opt_for_condition(lam, ang, tag)
    mprint(f"  Built {len(opts)} OptimizationProblems in {time.time()-t0:.1f}s")

    # --- 2. Load CLIP + val data (rank 0 only) ---
    if is_master:
        mprint(f"\n[2. Load CLIP + val data (rank 0 only)]")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mprint(f"  Device : {device}")
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device)
        clip_proc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad_(False)
        val_loader, text_emb, val_size = load_val_data(
            device, batch_size=args.batch_size, val_size=args.val_size,
        )
        if torch.cuda.is_available():
            mprint(f"  VRAM after CLIP load: "
                   f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
    else:
        device = None
        clip_model = None
        clip_proc = None
        text_emb = None
        val_loader = None
        val_size = args.val_size

    # --- 3. Evaluate each design ---
    results = {}
    t_start = time.time()

    for name, path in designs.items():
        result = evaluate_design(
            name, path, opts, tags_ordered,
            val_loader, val_size, clip_model, clip_proc, text_emb, device,
        )
        results[name] = result

        # Per-design save (in case we crash mid-loop)
        if is_master:
            np.savez(
                f"{args.out_dir}/{name}_eval.npz",
                psfs=result["psfs"],
            )
            with open(f"{args.out_dir}/{name}_eval.json", "w") as f:
                json.dump({k: v for k, v in result.items()
                           if k != "psfs"}, f, indent=2)

    # --- 4. Final summary ---
    if is_master:
        t_total = time.time() - t_start
        mprint(f"\n{'='*60}")
        mprint(f" SUMMARY")
        mprint(f"{'='*60}")
        mprint(f"  {'Design':<12} {'Acc':>8} {'N_correct':>10} "
               f"{'Meep(s)':>8} {'Eval(s)':>8}")
        for name, r in results.items():
            mprint(f"  {name:<12} {r['acc']*100:>7.2f}% "
                   f"{r['n_correct']:>10d} "
                   f"{r['meep_time_s']:>8.1f} "
                   f"{r['eval_time_s']:>8.1f}")
        mprint(f"\n  Total time: {t_total/60:.1f} min")

        # save everything at once
        summary = {
            name: {k: v for k, v in r.items() if k != "psfs"}
            for name, r in results.items()
        }
        summary["_meta"] = {
            "val_size": val_size,
            "batch_size": args.batch_size,
            "mpi_ranks": mpi_size,
            "total_time_min": t_total / 60,
        }
        with open(f"{args.out_dir}/baselines_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        mprint(f"\n  Results: {args.out_dir}/baselines_summary.json")


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
