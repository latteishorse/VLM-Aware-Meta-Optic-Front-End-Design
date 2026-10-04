"""
CODA VLM-aware Meep adjoint optimization (MPI rank-split)
=========================================================

Paper-facing variants:
  - VLM-cold: ``--init random``
  - VLM-warm: ``--init c2_warmstart`` from the same-seed Focus-opt iteration-100 checkpoint

Architecture:
  - all MPI ranks: build/run the Meep OptimizationProblem collectively
  - rank 0 only: holds GPU, CLIP, dataloader, and PyTorch optimizer

One training iteration:
  (a) rank 0 broadcasts rho
  (b) all ranks run 9 Meep forwards to extract raw monitor PSFs
  (c) rank 0 normalizes/resamples PSFs, runs frozen CLIP, and obtains dL/dPSF
  (d) the PSF cotangent is broadcast back to all ranks
  (e) all ranks run 9 forward+adjoint Meep passes to accumulate dL/drho
  (f) rank 0 clips the gradient, applies Adam, and projects rho to [0, 1]

The output directory prefix ``c3_output`` is retained for compatibility with the
reported checkpoints; it corresponds to VLM-cold/VLM-warm in the paper.
"""

import os
import sys
import time
import json
import argparse
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import meep as mp
import meep.adjoint as mpa
from autograd import numpy as npa          # autograd-compatible ops for the Meep objective
from focus_objective import FOCUS_OBJECTIVE_NAME

from mpi4py import MPI                     # explicit MPI collective

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
# Constants (same as the Focus-opt baseline script)
# =============================================================================

SX = 7.0
SY = 10.0
PML_THICK = 1.0
RESOLUTION = 67

SRC_Y = SY / 2 - PML_THICK - 0.3          # +3.7
MON_Y = -(SY / 2 - PML_THICK - 0.3)       # -3.7

DESIGN_X = 5.0
DESIGN_Y = 0.6
NX = int(DESIGN_X * RESOLUTION)           # 335
NY = int(DESIGN_Y * RESOLUTION)           # 40
N_DESIGN = NX * NY                        # 13,400

AIR = mp.Medium(index=1.0)
TIO2 = mp.Medium(index=2.4)

WAVELENGTHS = [0.450, 0.550, 0.650]
ANGLES_DEG = [-15, 0, 15]
LAM_NAMES = {0.450: "450nm", 0.550: "550nm", 0.650: "650nm"}

DECAY_BY = 1e-3
FWIDTH_FACTOR = 0.1

# Image formation
IMG_SIZE = 224
WAVELENGTHS_NM = [650, 550, 450]          # R, G, B
ZONE_ANGLES = [-15, 0, 15]                # left / mid / right zones


# =============================================================================
# External gradient state (module-level, closure target)
# =============================================================================
# Must stay identical across all ranks (Meep keeps them in sync via MPI).
# obj_func looks up EXTERNAL_GRADS[tag] at call time -> reflects the current value.
# Resized to the actual n_focal on first call.

EXTERNAL_GRADS = {}   # {tag: np.ndarray of shape (n_focal,)}


# =============================================================================
# OptimizationProblem builder (VJP-compatible objective)
# =============================================================================

def build_opt_for_condition(lam_um, angle_deg, tag):
    """
    OptimizationProblem for a single (lambda, angle) condition.
    Same geometry/source/k_point as Focus-opt; only the monitor and objective-interface change.
    """
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

    design_variables = mp.MaterialGrid(
        mp.Vector3(NX, NY, 0), AIR, TIO2,
    )
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

    # CODA uses the full focal line so PyTorch can supply a PSF cotangent at every sample
    ob_region = mp.Volume(
        center=mp.Vector3(0, MON_Y, 0),
        size=mp.Vector3(SX, 0, 0),
    )
    objective_arg = mpa.FourierFields(sim, ob_region, mp.Ez)

    # VJP-compatible objective. Meep/autograd differentiates w * |E|^2, equivalent to the Eq. (7) local VJP with 2*conj(E) under the solver convention.
    # must use autograd.numpy ops so Meep can auto-differentiate d(obj)/d(ez)
    # w is plain numpy (constant) -> only ez_1d is differentiated
    def obj_func(ez):
        if ez.ndim == 2:
            ez_1d = ez[0]
        else:
            ez_1d = ez
        n = len(ez_1d)
        w = EXTERNAL_GRADS.get(tag)
        if w is None or len(w) != n:
            # first call or size mismatch: initialize to ones
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


# =============================================================================
# Image formation: unit-sum PSFs, RGB wavelengths, three horizontal angle zones
# =============================================================================

def normalize_and_resample_psfs(psfs_raw, target_size, device):
    """(9, n_focal) raw -> (9, target_size) normalized, differentiable."""
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
    """Line-scan imaging: R/G/B -> 3λ, left/mid/right -> 3 angles."""
    B, C, H, W = images.shape
    assert C == 3

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


def clip_ce_loss(blurry, labels, clip_model, clip_proc, text_emb, device, temperature=100.0):
    """Frozen-CLIP classifier CE loss used for supervised optical optimization."""
    mean = torch.tensor(
        clip_proc.image_processor.image_mean, device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        clip_proc.image_processor.image_std, device=device
    ).view(1, 3, 1, 1)
    normalized = (blurry - mean) / std

    image_features = clip_model.get_image_features(pixel_values=normalized)

    # Unwrap BaseModelOutputWithPooling (some transformers 4.x versions)
    if not isinstance(image_features, torch.Tensor):
        if hasattr(image_features, 'pooler_output') and image_features.pooler_output is not None:
            image_features = image_features.pooler_output
        elif hasattr(image_features, 'last_hidden_state'):
            image_features = image_features.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError(
                f"Unexpected image_features type: {type(image_features)}"
            )

    image_features = F.normalize(image_features, dim=-1)

    logits = temperature * (image_features @ text_emb.T)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc

# =============================================================================
# Data pipeline (rank 0 only)
# =============================================================================

def load_train_data(device, batch_size=16, subset_size=None):
    """ImageNet-100 train + text embeddings. Called on rank 0 only.
    
    Note: the source dataset is class-sorted, so do not take a head subset.
    Use the full set with DataLoader shuffle=True to ensure class diversity.
    """
    from datasets import load_dataset
    from torchvision import transforms
    from collections import Counter

    print(f"  Loading ImageNet-100 train (full)...")
    ds = load_dataset("clane9/imagenet-100", split="train")
    print(f"  Total samples: {len(ds)}")
    
    # Sanity check: label diversity in random sample
    import numpy as _np
    rng = _np.random.default_rng(0)
    check_idx = rng.choice(len(ds), size=200, replace=False)
    check_labels = [ds[int(i)]['label'] for i in check_idx]
    n_unique = len(set(check_labels))
    print(f"  Label diversity (200 random samples): {n_unique} unique classes")
    if n_unique < 50:
        raise RuntimeError(
            f"Dataset appears not diverse enough: only {n_unique} unique labels "
            f"in 200 random samples. Expected ~100."
        )

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
        torch_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    emb_data = torch.load("b3_output/text_embeddings.pt", weights_only=False)
    text_emb = emb_data['clip_text_emb'].to(device)
    print(f"  Text embeddings: {text_emb.shape}")

    def infinite():
        while True:
            for b in loader:
                yield b

    return infinite(), text_emb

# =============================================================================
# Meep forward / adjoint (MPI collective: all ranks participate)
# =============================================================================

def meep_forward_only(opts, tags_ordered, rho_np):
    """
    All ranks participate. Forward-only PSF extraction.

    Returns:
        psfs_np: (9, n_focal) float32, identical across all ranks
    """
    psfs_list = []
    for tag in tags_ordered:
        # safety: reset external_grad (if None on first iter, obj_func inits to ones)
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


def meep_adjoint_with_grads(opts, tags_ordered, rho_np, grad_psfs_np):
    """
    All ranks participate. Inject the PyTorch gradient as the Meep adjoint source.

    Chain rule: ∂L/∂ρ = Σ_i (∂L/∂PSF_i) · (∂PSF_i/∂ρ)
      = Σ_i ∂/∂ρ [Σ_j (grad_PSF_i)_j · |Ez_i,j|²]
    With obj_func_i = sum_j w_j * |Ez_i,j|^2 (w = grad_PSF_i),
      Meep returns d(obj_func)/d(rho) -> accumulate.

    Args:
        grad_psfs_np: (9, n_focal) float64

    Returns:
        grad_rho_np: (N_DESIGN,) float64
    """
    grad_rho_accum = np.zeros(len(rho_np), dtype=np.float64)
    for i, tag in enumerate(tags_ordered):
        EXTERNAL_GRADS[tag][:] = grad_psfs_np[i]
        _, grad_rho_i = opts[tag]([rho_np], need_gradient=True)
        g = np.asarray(grad_rho_i).flatten()
        if len(g) > len(rho_np):
            g = g[:len(rho_np)]
        grad_rho_accum += g
    return grad_rho_accum


# =============================================================================
# Main
# =============================================================================

def load_initial_density(path):
    """Read a local density checkpoint and capture its provenance."""
    with np.load(path, allow_pickle=False) as data:
        rho = np.asarray(data["rho"], dtype=np.float64)
        if rho.shape not in ((NX, NY), (N_DESIGN,)):
            raise ValueError(f"Unexpected rho shape {rho.shape}; expected {(NX, NY)} or {(N_DESIGN,)}")
        if not np.isfinite(rho).all() or np.any((rho < 0) | (rho > 1)):
            raise ValueError("Checkpoint rho must contain finite values in [0, 1]")
        checkpoint_iter = None
        if "iter" in data:
            checkpoint_iter = int(np.asarray(data["iter"]).item())
        elif "history_iter" in data and data["history_iter"].size:
            checkpoint_iter = int(data["history_iter"][-1])
        objective_name = str(data["objective_name"].item()) if "objective_name" in data else None
        rho = rho.reshape(-1).copy()
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    return rho, {"path": path, "sha256": digest, "iteration": checkpoint_iter,
                 "objective_name": objective_name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["smoke", "50iter", "100iter", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--init", choices=["uniform", "random", "c2_warmstart"], default="random")
    parser.add_argument("--init-checkpoint", default=None,
                        help="VLM-warm density checkpoint (default: same-seed Focus-opt checkpoint_iter0100.npz)")
    args = parser.parse_args()
    if args.init_checkpoint is not None and args.init != "c2_warmstart":
        parser.error("--init-checkpoint requires --init c2_warmstart")
    if args.init == "c2_warmstart" and args.init_checkpoint is None:
        args.init_checkpoint = f"c2_output/seed{args.seed}/checkpoint_iter0100.npz"

    def mprint(*a, **k):
        if is_master:
            print(*a, **k, flush=True)

    n_iter = {"smoke": 10, "50iter": 50, "100iter": 100, "full": 200}[args.phase]

    # init-aware out_dir: avoid overwriting existing VLM-warm results
    #   --init random       -> c3_output/{phase}_seed{N}_rand/    (VLM-cold)
    #   --init uniform      -> c3_output/{phase}_seed{N}_uniform/
    #   --init c2_warmstart -> c3_output/{phase}_seed{N}/         (VLM-warm)
    init_tag = {"random": "_rand", "uniform": "_uniform", "c2_warmstart": ""}[args.init]
    out_dir = f"c3_output/{args.phase}_seed{args.seed}{init_tag}"

    # Check local inputs before any collective solver work. Every rank receives
    # the same failure, so an invalid rank-0 input cannot strand the other ranks.
    preflight_error = None
    rho_warm = None
    init_provenance = None
    if is_master:
        try:
            if os.path.exists(out_dir) and os.listdir(out_dir):
                raise RuntimeError(f"Output directory is non-empty: {out_dir}; choose a fresh run directory")
            if not os.path.isfile("b3_output/text_embeddings.pt"):
                raise FileNotFoundError("Run setup_dataset_embeddings.py to create b3_output/text_embeddings.pt first")
            if args.init == "c2_warmstart":
                rho_warm, init_provenance = load_initial_density(args.init_checkpoint)
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
    preflight_error = comm.bcast(preflight_error, root=0)
    if preflight_error:
        raise RuntimeError(f"Preflight failed: {preflight_error}")
    if is_master and init_provenance is not None:
        if init_provenance["objective_name"] != FOCUS_OBJECTIVE_NAME:
            mprint("Warning: this warm-start checkpoint does not declare the normalized Eq. (9) "
                   "Focus-opt objective. It may use the historical intensity objective; "
                   "its recorded provenance will be retained.")

    mprint("=" * 60)
    mprint(f" CODA -- frozen-CLIP meta-optic optimization (phase: {args.phase})")
    mprint("=" * 60)
    mprint(f"  MPI ranks  : {mpi_size}")
    mprint(f"  Iterations : {n_iter}")
    mprint(f"  Seed       : {args.seed}")
    mprint(f"  Init       : {args.init}")
    mprint(f"  LR         : {args.lr}")
    mprint(f"  Batch size : {args.batch_size}")
    mprint(f"  Design dof : {N_DESIGN} ({NX} × {NY})")
    mprint(f"  Monitor    : {SX} μm (full focal line)")
    mprint(f"  Output     : {out_dir}/")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- 1. Build 9 Meep opts (all ranks) ---
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
    mprint(f"  Tags: {tags_ordered}")

    # --- 2. Load CLIP + data (rank 0 only) ---
    if is_master:
        mprint(f"\n[2. Load CLIP + data (rank 0 only)]")
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
        dataloader, text_emb = load_train_data(device, batch_size=args.batch_size)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / 1e9
            mprint(f"  VRAM after CLIP load: {vram_gb:.2f} GB")
    else:
        device = None
        clip_model = None
        clip_proc = None
        text_emb = None
        dataloader = None

    # --- 3. Initialize ρ (rank 0 state; numpy copy broadcast each iter) ---
    if is_master:
        mprint(f"\n[3. Initialize rho (rank 0)]")
        if args.init == "uniform":
            rho_np_init = np.full((N_DESIGN,), 0.5, dtype=np.float64)
        elif args.init == "c2_warmstart":
            rho_np_init = rho_warm
            mprint(f"  Warm-start from: {args.init_checkpoint}")
            mprint(f"  Focus-opt rho: mean={rho_np_init.mean():.3f}, "
                   f"binary frac={((rho_np_init>0.9)|(rho_np_init<0.1)).mean()*100:.1f}%")
        else:  # "random"
            rng = np.random.default_rng(args.seed)
            rho_np_init = (0.3 + 0.4 * rng.random(N_DESIGN)).astype(np.float64)
        rho_torch = torch.tensor(rho_np_init, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam(
            [rho_torch], lr=args.lr, betas=(0.9, 0.999), eps=1e-8
        )
        mprint(f"  rho init : mean={rho_torch.mean():.3f}, "
               f"min={rho_torch.min():.3f}, max={rho_torch.max():.3f}")
    else:
        rho_torch = None
        optimizer = None

    # --- 4. Training loop ---
    mprint(f"\n[4. Training loop -- {n_iter} iter]")
    history = {"iter": [], "loss": [], "acc": [], "iter_time": []}
    t_start = time.time()

    for it in range(1, n_iter + 1):
        t_iter = time.time()

        # (a) Broadcast rho to all ranks
        if is_master:
            rho_clamped = rho_torch.clamp(0.0, 1.0) \
                .detach().cpu().numpy().astype(np.float64)
        else:
            rho_clamped = np.empty(N_DESIGN, dtype=np.float64)
        comm.Bcast(rho_clamped, root=0)

        # (b) All ranks: Meep forward -> PSF
        psfs_np = meep_forward_only(opts, tags_ordered, rho_clamped)
        n_focal = psfs_np.shape[1]

        # (c) Rank 0: VLM forward + backward
        if is_master:
            images, labels = next(dataloader)
            images = images.to(device)
            labels = labels.to(device)

            psfs_torch = torch.tensor(
                psfs_np, dtype=torch.float32, device=device, requires_grad=True,
            )
            psfs_norm = normalize_and_resample_psfs(psfs_torch, IMG_SIZE, device)
            blurry = apply_meta_optic(images, psfs_norm, tags_ordered).clamp(0, 1)
            loss, acc = clip_ce_loss(
                blurry, labels, clip_model, clip_proc, text_emb, device,
            )

            optimizer.zero_grad()
            loss.backward()
            grad_psfs_np = psfs_torch.grad.detach().cpu().numpy().astype(np.float64)
            loss_val = loss.item()
        else:
            grad_psfs_np = np.empty((9, n_focal), dtype=np.float64)
            loss_val = 0.0
            acc = 0.0

        # (d) Bcast grad_PSF, loss, acc
        comm.Bcast(grad_psfs_np, root=0)
        loss_val = comm.bcast(loss_val, root=0)
        acc = comm.bcast(acc, root=0)

        # (e) All ranks: Meep forward+adjoint with VJP
        grad_rho_np = meep_adjoint_with_grads(
            opts, tags_ordered, rho_clamped, grad_psfs_np,
        )

        # (f) Rank 0: optimizer step
        if is_master:
            rho_torch.grad = torch.tensor(
                grad_rho_np, dtype=torch.float32,
            ).view(rho_torch.shape)
            grad_norm = torch.nn.utils.clip_grad_norm_([rho_torch], max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                rho_torch.clamp_(0.0, 1.0)

            dt = time.time() - t_iter
            history["iter"].append(it)
            history["loss"].append(loss_val)
            history["acc"].append(acc)
            history["iter_time"].append(dt)
            elapsed = time.time() - t_start
            eta_min = elapsed / it * (n_iter - it) / 60
            rho_mean = rho_torch.mean().item()
            mprint(
                f"  iter {it:>3d}/{n_iter} | loss={loss_val:.4f} | "
                f"acc={acc*100:.1f}% | ρ_mean={rho_mean:.3f} | "
                f"‖g‖={grad_norm:.2e} | {dt:.1f}s/iter | ETA {eta_min:.1f}min"
            )

            # Checkpoint
            if it % 10 == 0 or it == n_iter:
                np.savez(
                    f"{out_dir}/checkpoint_iter{it:04d}.npz",
                    rho=rho_torch.detach().numpy().reshape(NX, NY),
                    iter=it,
                    # Training PSFs and gradients precede this iteration's
                    # update; keep the matching density explicitly alongside.
                    psfs=psfs_np,
                    rho_for_psfs=rho_clamped.reshape(NX, NY),
                    psfs_iteration=it - 1,
                    grad_psfs=grad_psfs_np.astype(np.float32),
                    grad_rho=grad_rho_np.astype(np.float32),
                    history_loss=np.array(history["loss"]),
                    history_acc=np.array(history["acc"]),
                )

    t_total = time.time() - t_start

    # Recompute collectively: the final artifact must pair the updated density
    # with PSFs from that same density, not the last pre-update training PSFs.
    rho_final_np = (rho_torch.detach().numpy().astype(np.float64)
                    if is_master else np.empty(N_DESIGN, dtype=np.float64))
    comm.Bcast(rho_final_np, root=0)
    psfs_final = meep_forward_only(opts, tags_ordered, rho_final_np)

    # --- 5. Final save (rank 0) ---
    if is_master:
        mprint(f"\n[5. Final save]")
        np.savez(
            f"{out_dir}/c3_final.npz",
            rho=rho_final_np.reshape(NX, NY),
            iter=n_iter,
            psfs_final=psfs_final,
            history_iter=np.array(history["iter"]),
            history_loss=np.array(history["loss"]),
            history_acc=np.array(history["acc"]),
            history_iter_time=np.array(history["iter_time"]),
        )
        with open(f"{out_dir}/summary.json", "w") as f:
            json.dump({
                "phase": args.phase,
                "n_iter": n_iter,
                "seed": args.seed,
                "init": args.init,
                "init_checkpoint": args.init_checkpoint,
                "init_checkpoint_provenance": init_provenance,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "mpi_ranks": mpi_size,
                "n_focal": int(n_focal),
                "final_loss": history["loss"][-1],
                "final_acc": history["acc"][-1],
                "mean_iter_time_s": float(np.mean(history["iter_time"])),
                "total_time_min": t_total / 60,
            }, f, indent=2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].plot(history["iter"], history["loss"], lw=1.0)
        axes[0].set_xlabel("Iter")
        axes[0].set_ylabel("CE loss")
        axes[0].set_title("VLM CE loss curve")
        axes[0].grid(alpha=0.3)
        axes[1].plot(history["iter"], [a*100 for a in history["acc"]],
                     lw=1.0, color='C2')
        axes[1].set_xlabel("Iter")
        axes[1].set_ylabel("Batch acc (%)")
        axes[1].set_title("Accuracy curve")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/c3_curves.png", dpi=120)
        plt.close()

        mprint(f"\n  ✅ Done in {t_total/60:.1f} min "
               f"({np.mean(history['iter_time']):.1f}s/iter)")
        mprint(f"  Final loss : {history['loss'][-1]:.4f}")
        mprint(f"  Final acc  : {history['acc'][-1]*100:.1f}%")
        mprint(f"  Results    : {out_dir}/")


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
