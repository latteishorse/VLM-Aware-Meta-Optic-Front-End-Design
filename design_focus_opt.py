"""
Focus-opt normalized focal-concentration baseline (manuscript Eq. 9)
=================================================================

Paper-facing role:
  Model-agnostic optical baseline used in Table 1 and Fig. 6. The script
  optimizes the same continuous density variables as CODA but uses no labels,
  text prompts, or frozen-encoder gradients.

Objective implemented here:
  minimize L_focus = -mean_k [sum_{|x| <= 0.25 um} |Ez_k(x)|^2 /
                              max(sum_full_sensor_line |Ez_k(x)|^2, 1e-20)].
  Each of the 9 wavelength-angle conditions has equal weight. A full-width
  7 um monitor supplies both the numerator and denominator; actual Meep sample
  coordinates define the central 0.5 um target window for every condition.
  The denominator floor only handles vanishing fields. The optimizer maximizes
  the positive mean concentration by minimizing its negative gradient.

  This corrects the supplied legacy narrow-monitor intensity objective. New
  checkpoints carry objective metadata; historical paper results are unchanged.

Compatibility note:
  Output files are written to ``c2_output/`` and named ``c2_final.npz`` to match
  the checkpoint paths used by the VLM-warm warm-start code.

Run:
  conda activate pmp
  mpirun -np 4 python design_focus_opt.py --seed 0 --iter 200 --init random --lr 0.05
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import meep.adjoint as mpa
from focus_objective import (
    FOCUS_EPS,
    FOCUS_OBJECTIVE_NAME,
    FOCUS_WINDOW_HALF_WIDTH_UM,
    focal_concentration_from_fields,
    focus_window_mask,
    normalized_focal_fraction,
)


# --- Setup ---
SX = 7.0
SY = 10.0
PML_THICK = 1.0
RESOLUTION = 67

SRC_Y = SY / 2 - PML_THICK - 0.3       # +3.7
MON_Y = -(SY / 2 - PML_THICK - 0.3)    # -3.7

DESIGN_X = 5.0
DESIGN_Y = 0.6
NX = int(DESIGN_X * RESOLUTION)
NY = int(DESIGN_Y * RESOLUTION)

AIR = mp.Medium(index=1.0)
TIO2 = mp.Medium(index=2.4)

# Conditions
WAVELENGTHS = [0.450, 0.550, 0.650]
ANGLES_DEG = [-15, 0, 15]
LAM_NAMES = {0.450: "450nm", 0.550: "550nm", 0.650: "650nm"}

DECAY_BY = 1e-3
FWIDTH_FACTOR = 0.1

# Adam
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8

FOCUS_OBJECTIVE_METADATA = {
    "objective_name": FOCUS_OBJECTIVE_NAME,
    "objective_aggregation": "equal_condition_mean",
    "objective_loss_sign": -1.0,
    "objective_monitor_width_um": SX,
    "objective_window_center_um": 0.0,
    "objective_window_half_width_um": FOCUS_WINDOW_HALF_WIDTH_UM,
    "objective_denominator_floor": FOCUS_EPS,
    "objective_coordinate_source": "meep_get_array_metadata",
    "history_quantity": "positive_mean_concentration_before_update",
}


def build_opt_for_condition(lam_um, angle_deg):
    """Build the OptimizationProblem for a single (lambda, angle) condition."""
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
        mp.Vector3(NX, NY, 0),
        AIR,
        TIO2,
    )
    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(DESIGN_X, DESIGN_Y, 0),
        ),
    )

    geometry = [
        mp.Block(
            center=design_region.center,
            size=design_region.size,
            material=design_variables,
        )
    ]

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

    # Eq. 9 needs the complete sensor line for the normalization denominator.
    ob_region = mp.Volume(
        center=mp.Vector3(0, MON_Y, 0),
        size=mp.Vector3(SX, 0, 0),
    )
    objective = mpa.FourierFields(sim, ob_region, mp.Ez, yee_grid=False)
    window_mask = None

    def obj_func(ez):
        nonlocal window_mask
        if window_mask is None:
            # Called only after the forward monitor is initialized. Meep's
            # metadata matches the voxel-centered FourierFields samples;
            # guessing a linspace can shift the focal-window boundary.
            x, _, _, _ = sim.get_array_metadata(vol=ob_region)
            window_mask = focus_window_mask(x)
        return focal_concentration_from_fields(ez, window_mask)

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[obj_func],
        objective_arguments=[objective],
        design_regions=[design_region],
        frequencies=[fcen],
        decay_by=DECAY_BY,
    )
    return opt


def evaluate_all_conditions(opts, rho_flat):
    """
    Run forward + adjoint over the 9 conditions.
    
    Returns:
        total_f: positive equal-condition mean concentration
        total_grad: gradient of that positive mean; Adam minimizes its negative
        per_condition: dict {tag: normalized concentration}
    """
    n_cond = len(opts)
    total_grad = np.zeros_like(rho_flat)
    total_f = 0.0
    per_cond = {}

    for tag, opt in opts.items():
        f, grad = opt([rho_flat])
        f_val = float(np.asarray(f).flatten()[0])
        grad_flat = np.asarray(grad).flatten()
        if len(grad_flat) > NX * NY:
            grad_flat = grad_flat[:NX * NY]

        per_cond[tag] = f_val
        total_f += f_val / n_cond           # equal-weight average
        total_grad += grad_flat / n_cond

    return total_f, total_grad, per_cond


def standalone_field_sim(rho_flat, lam_um, angle_deg, sim_time=80.0):
    """Extract a steady-state field map via ContinuousSource (for visualization)."""
    fcen = 1.0 / lam_um
    rot_angle = np.deg2rad(angle_deg)

    k_point = mp.Vector3(
        x=fcen * np.sin(rot_angle),
        y=-fcen * np.cos(rot_angle),
        z=0,
    )
    src = mp.ContinuousSource(fcen)
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
        weights=rho_flat,
    )
    geometry = [mp.Block(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(DESIGN_X, DESIGN_Y, 0),
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

    dft_full = sim.add_dft_fields(
        [mp.Ez], fcen, 0, 1,
        where=mp.Volume(center=mp.Vector3(), size=mp.Vector3(SX, SY, 0)),
    )
    dft_focal = sim.add_dft_fields(
        [mp.Ez], fcen, 0, 1,
        where=mp.Volume(center=mp.Vector3(0, MON_Y, 0), size=mp.Vector3(SX, 0, 0)),
    )

    sim.run(until=sim_time)
    mag_full = np.abs(sim.get_dft_array(dft_full, mp.Ez, 0))
    I_focal = np.abs(sim.get_dft_array(dft_focal, mp.Ez, 0)) ** 2
    x_focal, _, _, _ = sim.get_array_metadata(dft_cell=dft_focal)

    return mag_full, I_focal, x_focal


def main():
    # MPI-safe print
    global print
    if not mp.am_master():
        print = lambda *a, **k: None

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iter", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init", type=str, default="uniform",
                        choices=["uniform", "random"])
    parser.add_argument("--save_field_every", type=int, default=50,
                        help="Save field maps every N iterations (0 = never)")
    args = parser.parse_args()
    if args.iter < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--iter and --lr must be positive")

    out_dir = f"c2_output/seed{args.seed}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(" Focus-opt -- legacy unnormalized focal-intensity optimization")
    print("=" * 60)
    print(f"  Seed         : {args.seed}")
    print(f"  Iterations   : {args.iter}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Init         : {args.init}")
    print(f"  Objective    : {FOCUS_OBJECTIVE_NAME}")
    print(f"  Focal window : |x| <= {FOCUS_WINDOW_HALF_WIDTH_UM} um; full-line normalization")
    print(f"  Wavelengths  : {[int(w*1000) for w in WAVELENGTHS]} nm")
    print(f"  Angles       : {ANGLES_DEG}°")
    print(f"  Conditions   : {len(WAVELENGTHS) * len(ANGLES_DEG)}")
    print(f"  Output       : {out_dir}/")

    # --- Initial ρ ---
    if args.init == "uniform":
        rho = np.full((NX * NY,), 0.5)
    else:
        rng = np.random.default_rng(args.seed)
        rho = 0.3 + 0.4 * rng.random(NX * NY)

    rho_initial = rho.copy()

    # --- Build OptimizationProblem for each condition ---
    print(f"\n[Build] {len(WAVELENGTHS)} × {len(ANGLES_DEG)} = "
          f"{len(WAVELENGTHS)*len(ANGLES_DEG)} OptimizationProblems")
    t0 = time.time()
    opts = {}
    for lam in WAVELENGTHS:
        for ang in ANGLES_DEG:
            tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
            opts[tag] = build_opt_for_condition(lam, ang)
    print(f"  Build time: {time.time()-t0:.1f}s")

    # --- Adam state ---
    m = np.zeros_like(rho)
    v = np.zeros_like(rho)

    # --- History ---
    history = {
        "iter": [],
        "total_loss": [],
        "rho_mean": [],
        "rho_std": [],
        "iter_time": [],
        "per_condition": {tag: [] for tag in opts.keys()},
    }

    print(f"\n[Optimization] {args.iter} iterations")
    t_start = time.time()

    for it in range(1, args.iter + 1):
        t_iter = time.time()

        # Forward + adjoint on all 9 conditions
        total_f, total_grad, per_cond = evaluate_all_conditions(opts, rho)

        # Adam (maximize -> minimize -f)
        g = -total_grad
        m = ADAM_B1 * m + (1 - ADAM_B1) * g
        v = ADAM_B2 * v + (1 - ADAM_B2) * (g ** 2)
        m_hat = m / (1 - ADAM_B1 ** it)
        v_hat = v / (1 - ADAM_B2 ** it)
        rho = rho - args.lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
        rho = np.clip(rho, 0.0, 1.0)

        dt = time.time() - t_iter

        # Log
        history["iter"].append(it)
        history["total_loss"].append(total_f)
        history["rho_mean"].append(float(rho.mean()))
        history["rho_std"].append(float(rho.std()))
        history["iter_time"].append(dt)
        for tag, val in per_cond.items():
            history["per_condition"][tag].append(val)

        if it == 1 or it % 5 == 0 or it == args.iter:
            elapsed = time.time() - t_start
            eta = elapsed / it * (args.iter - it)
            print(f"  iter {it:>4d}/{args.iter}: "
                  f"f_total={total_f:.3e}, "
                  f"ρ_mean={rho.mean():.3f}, ρ_std={rho.std():.3f}, "
                  f"{dt:.1f}s/iter, ETA {eta/60:.1f}min")

        # Periodic save
        if mp.am_master() and (it % 25 == 0 or it == args.iter):
            np.savez(
                os.path.join(out_dir, f"checkpoint_iter{it:04d}.npz"),
                rho=rho.reshape(NX, NY),
                history=history,
                iter=it,
                **FOCUS_OBJECTIVE_METADATA,
            )

        if not np.isfinite(total_f):
            print(f"  ⚠ NaN/Inf at iter {it}, stopping")
            break

    t_total = time.time() - t_start

    # --- Final ρ ---
    rho_final = rho.copy()

    print("\n" + "=" * 60)
    print(" Optimization complete")
    print("=" * 60)
    print(f"  Total time   : {t_total/60:.1f} min ({t_total/len(history['iter']):.1f}s/iter)")
    print(f"  Final loss   : {history['total_loss'][-1]:.3e}")
    print(f"  Initial loss : {history['total_loss'][0]:.3e}")
    print(f"  Improvement  : {history['total_loss'][-1] / history['total_loss'][0]:.2f}×")
    print(f"  Final ρ_mean : {rho_final.mean():.3f}")
    print(f"  Final ρ_std  : {rho_final.std():.3f}")

    # --- Per-condition final values ---
    print(f"\n[Per-condition final loss]")
    for tag, vals in history["per_condition"].items():
        if vals:
            print(f"  {tag}: {vals[-1]:.3e}")

    # --- Final evaluation: detailed forward sims + field maps ---
    print(f"\n[Final evaluation: 9 forward sims for analysis]")
    final_results = {}
    field_maps = {}
    for lam in WAVELENGTHS:
        for ang in ANGLES_DEG:
            tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
            t0 = time.time()
            mag, I_focal, x_focal = standalone_field_sim(rho_final, lam, ang)
            field_maps[tag] = mag
            n = len(I_focal)
            # Use the same physical window and normalized ratio as training.
            mask_center = focus_window_mask(x_focal)
            energy_total = I_focal.sum()
            conc = float(normalized_focal_fraction(I_focal, mask_center))
            final_results[tag] = {
                "I_focal": I_focal,
                "x_focal": x_focal,
                "concentration": conc,
                "peak": I_focal[n//2-5:n//2+6].max(),
                "energy_total": energy_total,
            }
            print(f"  {tag}: conc={conc:.3f}, peak={final_results[tag]['peak']:.2e}, "
                  f"({time.time()-t0:.1f}s)")

    mean_conc = np.mean([r["concentration"] for r in final_results.values()])
    print(f"\n  Mean concentration: {mean_conc:.3f}")

    # --- Save (master only) ---
    if mp.am_master():
        # Final design
        np.savez(
            os.path.join(out_dir, "c2_final.npz"),
            rho=rho_final.reshape(NX, NY),
            rho_initial=rho_initial.reshape(NX, NY),
            iter=len(history["iter"]),
            **FOCUS_OBJECTIVE_METADATA,
            history_iter=np.array(history["iter"]),
            history_loss=np.array(history["total_loss"]),
            history_rho_mean=np.array(history["rho_mean"]),
            history_rho_std=np.array(history["rho_std"]),
            **{f"final_I_focal_{tag}": r["I_focal"] for tag, r in final_results.items()},
            **{f"final_x_focal_{tag}": r["x_focal"] for tag, r in final_results.items()},
            **{f"final_conc_{tag}": r["concentration"] for tag, r in final_results.items()},
        )

        # Summary
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(f"Focus-opt -- focal-concentration Meep adjoint optimization (seed={args.seed})\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Setup:\n")
            f.write(f"  Iterations  : {len(history['iter'])}/{args.iter}\n")
            f.write(f"  Optimizer   : Adam (lr={args.lr})\n")
            f.write(f"  Init        : {args.init}\n")
            f.write(f"  Objective   : {FOCUS_OBJECTIVE_NAME}\n")
            f.write(f"  Loss        : -mean_k sum(|x|<=0.25um PSF_k) / max(sum(full-line PSF_k), {FOCUS_EPS})\n")
            f.write(f"  Coordinates : actual Meep voxel-centered monitor samples\n")
            f.write(f"  History     : positive mean concentration before each update\n")
            f.write(f"  Total time  : {t_total/60:.1f} min ({t_total/len(history['iter']):.1f}s/iter)\n\n")
            f.write(f"Loss:\n")
            f.write(f"  Initial   : {history['total_loss'][0]:.3e}\n")
            f.write(f"  Final     : {history['total_loss'][-1]:.3e}\n")
            f.write(f"  Improve   : {history['total_loss'][-1] / history['total_loss'][0]:.2f}×\n\n")
            f.write(f"ρ stats:\n")
            f.write(f"  Initial   : mean={rho_initial.mean():.3f}, std={rho_initial.std():.3f}\n")
            f.write(f"  Final     : mean={rho_final.mean():.3f}, std={rho_final.std():.3f}\n\n")
            f.write(f"Final per-condition concentration:\n")
            for tag in sorted(final_results.keys()):
                f.write(f"  {tag}: {final_results[tag]['concentration']:.3f}, "
                        f"peak={final_results[tag]['peak']:.2e}\n")
            f.write(f"\n  Mean concentration: {mean_conc:.3f}\n")
            f.write(f"  Fresnel baseline mean conc: 0.091\n")
            f.write(f"  Focus-opt/Fresnel concentration ratio: {mean_conc/0.091:.2f}×\n")

        # Loss curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(history["iter"], history["total_loss"], lw=1.0)
        axes[0,0].set_xlabel("Iteration"); axes[0,0].set_ylabel("Total loss (mean across 9 cond)")
        axes[0,0].set_title("Loss curve"); axes[0,0].grid(alpha=0.3); axes[0,0].set_yscale("log")

        for tag, vals in history["per_condition"].items():
            axes[0,1].plot(history["iter"], vals, lw=0.8, label=tag)
        axes[0,1].set_xlabel("Iteration"); axes[0,1].set_ylabel("Per-condition loss")
        axes[0,1].set_title("Per-condition losses")
        axes[0,1].legend(fontsize=6, ncol=3, loc="lower right")
        axes[0,1].grid(alpha=0.3); axes[0,1].set_yscale("log")

        axes[1,0].plot(history["iter"], history["rho_mean"], lw=1.0, color='C1')
        axes[1,0].axhline(0.5, color='gray', ls=':', lw=1)
        axes[1,0].set_xlabel("Iteration"); axes[1,0].set_ylabel("ρ mean")
        axes[1,0].set_title("ρ mean trajectory"); axes[1,0].set_ylim(0, 1)
        axes[1,0].grid(alpha=0.3)

        axes[1,1].plot(history["iter"], history["rho_std"], lw=1.0, color='C2')
        axes[1,1].set_xlabel("Iteration"); axes[1,1].set_ylabel("ρ std")
        axes[1,1].set_title("ρ std (binarization)"); axes[1,1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "optimization_curves.png"), dpi=120)
        plt.close()

        # Final ρ
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        for ax, r, title in zip(
            axes, [rho_initial.reshape(NX, NY), rho_final.reshape(NX, NY)],
            [f"Initial ({args.init})", f"Final (after {len(history['iter'])} iter)"]
        ):
            im = ax.imshow(r.T, origin="lower", cmap="gray_r", aspect="auto",
                          extent=[-DESIGN_X/2, DESIGN_X/2, -DESIGN_Y/2, DESIGN_Y/2],
                          vmin=0, vmax=1)
            ax.set_title(f"{title} (mean={r.mean():.3f}, std={r.std():.3f})")
            ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
            plt.colorbar(im, ax=ax, fraction=0.04)
        plt.suptitle(f"Focus-opt design ρ (seed={args.seed})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rho_initial_vs_final.png"), dpi=120)
        plt.close()

        # 3×3 field map grid
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        for i, lam in enumerate(WAVELENGTHS):
            for j, ang in enumerate(ANGLES_DEG):
                tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
                ax = axes[i, j]
                mag = field_maps[tag]
                x_axis = np.linspace(-SX/2, SX/2, mag.shape[0])
                y_axis = np.linspace(-SY/2, SY/2, mag.shape[1])
                im = ax.imshow(mag.T, extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
                              origin="lower", cmap="inferno", aspect="auto")
                ax.axhline(DESIGN_Y/2, color="white", ls=":", lw=0.5, alpha=0.5)
                ax.axhline(-DESIGN_Y/2, color="white", ls=":", lw=0.5, alpha=0.5)
                ax.axhline(MON_Y, color="lime", ls="--", lw=0.8, alpha=0.7)
                conc = final_results[tag]["concentration"]
                ax.set_title(f"{LAM_NAMES[lam]}, {ang:+d}° (conc={conc:.3f})")
                if i == 2: ax.set_xlabel("x (μm)")
                if j == 0: ax.set_ylabel("y (μm)")
                plt.colorbar(im, ax=ax, fraction=0.04)
        plt.suptitle(f"Focus-opt final |Ez| field maps (seed={args.seed}, mean conc={mean_conc:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "field_maps_final.png"), dpi=100)
        plt.close()

        # Fresnel vs Focus-opt comparison plot
        try:
            c1 = np.load("c1_output/c1_results.npz")
            fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
            for ax, lam in zip(axes, WAVELENGTHS):
                lam_n = LAM_NAMES[lam]
                # 0° only
                tag = f"{lam_n}_ang+0"
                # Fresnel
                if f"x_focal_{tag}" in c1 and f"I_focal_{tag}" in c1:
                    ax.plot(c1[f"x_focal_{tag}"], c1[f"I_focal_{tag}"],
                           lw=1.0, color='gray', label="Fresnel")
                # Focus-opt
                ax.plot(final_results[tag]["x_focal"], final_results[tag]["I_focal"],
                       lw=1.5, color='C0', label="Focus-opt")
                ax.set_xlabel("x (μm)"); ax.set_ylabel("|E|²")
                ax.set_title(f"{lam_n}, 0°")
                ax.set_xlim(-2.5, 2.5)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
            plt.suptitle("Fresnel zone plate vs Focus-opt at on-axis")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "c1_vs_c2_focal.png"), dpi=120)
            plt.close()
        except FileNotFoundError:
            print("  (Fresnel results not found, skipping comparison plot)")

    print(f"\n  Results: {out_dir}/")
    print(f"    summary.txt              -- numeric summary")
    print(f"    c2_final.npz             -- Final ρ + history + per-cond results")
    print(f"    optimization_curves.png  -- Loss + ρ trajectory")
    print(f"    rho_initial_vs_final.png -- ρ before/after")
    print(f"    field_maps_final.png     -- 3λ × 3angle final field maps")
    print(f"    c1_vs_c2_focal.png       -- Fresnel vs Focus-opt comparison (if available)")


if __name__ == "__main__":
    main()
