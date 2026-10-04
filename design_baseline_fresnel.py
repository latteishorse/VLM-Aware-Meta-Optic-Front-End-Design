"""
Fresnel zone-plate analytical reference
=======================================

Paper-facing role:
  Analytical diffractive baseline under the same 5 um aperture, 600 nm design
  thickness, wavelengths, angles, FDTD grid, and line-scan evaluation pipeline.

Design method:
  Quantize a hyperboloidal phase profile to a binary Fresnel zone plate designed
  at 550 nm. Other wavelengths and angles are evaluated without re-optimization.

Run:
  conda activate pmp
  mpirun -np 4 python design_baseline_fresnel.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

OUT_DIR = "c1_output"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Setup ---
SX = 7.0
SY = 10.0
PML_THICK = 1.0
RESOLUTION = 67

SRC_Y = SY / 2 - PML_THICK - 0.3       # +3.7
MON_Y = -(SY / 2 - PML_THICK - 0.3)    # -3.7

DESIGN_X = 5.0
DESIGN_Y = 0.6
NX = int(DESIGN_X * RESOLUTION)        # 335
NY = int(DESIGN_Y * RESOLUTION)        # 40

AIR = mp.Medium(index=1.0)
TIO2 = mp.Medium(index=2.4)

# --- Lens design ---
DESIGN_LAMBDA = 0.550   # um, single-wavelength design
FOCAL_LENGTH = 3.7      # μm, lens center (y=0) -> focal plane (y=-3.7)
N_TIO2 = 2.4

# --- Evaluation ---
WAVELENGTHS_EVAL = [0.450, 0.550, 0.650]
ANGLES_EVAL = [-15, 0, 15]
LAM_NAMES = {0.450: "450nm", 0.550: "550nm", 0.650: "650nm"}
SIM_TIME = 80.0


def design_fresnel_zone_plate(design_lambda, focal_length, n_material):
    """
    Build the binary Fresnel zone plate rho pattern.

    Phase profile (lens equation):
        φ(x) = -2π/λ · (√(f² + x²) - f)

    Binarization:
        ρ(x, y) = 1 if cos(φ(x)) > 0 else 0

    uniform along y (the only variable is x)

    Returns:
        rho: (NX, NY) array, ρ ∈ {0, 1}
    """
    # x coordinates of design grid (μm)
    x_pixels = np.linspace(-DESIGN_X/2, DESIGN_X/2, NX)

    # Hyperboloidal phase
    phase = -2 * np.pi / design_lambda * (
        np.sqrt(focal_length**2 + x_pixels**2) - focal_length
    )

    # Binary zone plate
    rho_x = (np.cos(phase) > 0).astype(float)

    # Extend to (NX, NY) - y direction uniform
    rho = np.tile(rho_x[:, np.newaxis], (1, NY))
    return rho, phase, x_pixels


def setup_simulation(rho_flat, lam_um, angle_deg):
    """Same structure as the standalone field simulation."""
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
    return sim, fcen


def run_forward(rho_flat, lam_um, angle_deg):
    """Run the forward sim; return focal intensity + full field."""
    sim, fcen = setup_simulation(rho_flat, lam_um, angle_deg)

    dft_focal = sim.add_dft_fields(
        [mp.Ez], fcen, 0, 1,
        where=mp.Volume(center=mp.Vector3(0, MON_Y, 0), size=mp.Vector3(SX, 0, 0)),
    )
    dft_full = sim.add_dft_fields(
        [mp.Ez], fcen, 0, 1,
        where=mp.Volume(center=mp.Vector3(), size=mp.Vector3(SX, SY, 0)),
    )

    sim.run(until=SIM_TIME)

    ez_focal = sim.get_dft_array(dft_focal, mp.Ez, 0)
    ez_full = sim.get_dft_array(dft_full, mp.Ez, 0)

    return {
        "I_focal": np.abs(ez_focal) ** 2,
        "mag_full": np.abs(ez_full),
    }


def compute_metrics(I_focal, x_focal):
    """
    Compute metrics from the focal-plane intensity:
      - Peak intensity at center (x=0)
      - Total energy in central 0.5μm
      - FWHM
      - Strehl ratio (peak / sum)
    """
    n = len(I_focal)
    # peak near x=0
    center_idx = n // 2
    window = 5
    peak_center = I_focal[center_idx-window:center_idx+window+1].max()

    # Central 0.5μm energy
    mask_center = np.abs(x_focal) < 0.25
    energy_center = I_focal[mask_center].sum()

    # Total energy
    energy_total = I_focal.sum()

    # FWHM (simple estimate)
    half_max = peak_center / 2
    above = I_focal > half_max
    if above.sum() > 0:
        idx_above = np.where(above)[0]
        fwhm = (x_focal[idx_above[-1]] - x_focal[idx_above[0]])
    else:
        fwhm = 0.0

    # Concentration ratio (energy in 0.5μm / total)
    concentration = energy_center / max(energy_total, 1e-20)

    return {
        "peak_center": float(peak_center),
        "energy_center": float(energy_center),
        "energy_total": float(energy_total),
        "concentration": float(concentration),
        "fwhm_um": float(fwhm),
    }


def main():
    # MPI-safe print
    global print
    if not mp.am_master():
        print = lambda *a, **k: None

    print("=" * 60)
    print(" Fresnel zone plate -- analytical diffractive baseline")
    print("=" * 60)
    print(f"  Aperture       : {DESIGN_X} μm")
    print(f"  Focal length   : {FOCAL_LENGTH} μm (f-number = {FOCAL_LENGTH/DESIGN_X:.2f})")
    print(f"  Design λ       : {DESIGN_LAMBDA*1000:.0f} nm")
    print(f"  Material       : TiO₂ n={N_TIO2}")
    print(f"  Thickness      : {DESIGN_Y*1000:.0f} nm")

    # --- Design Fresnel zone plate ---
    print(f"\n[Design] Binary Fresnel zone plate")
    rho, phase, x_pixels = design_fresnel_zone_plate(
        DESIGN_LAMBDA, FOCAL_LENGTH, N_TIO2
    )
    rho_flat = rho.flatten()

    print(f"  Pattern shape: {rho.shape}")
    print(f"  Fill ratio   : {rho.mean():.3f}")
    print(f"  N zones (both sides): ~{int(np.unwrap(phase).ptp() / (2*np.pi))}")

    # --- Evaluate at 9 conditions ---
    print(f"\n[Evaluation] {len(WAVELENGTHS_EVAL)} λ × {len(ANGLES_EVAL)} angles = "
          f"{len(WAVELENGTHS_EVAL)*len(ANGLES_EVAL)} sims")

    results = {}
    field_maps = {}
    total_t0 = time.time()

    for lam in WAVELENGTHS_EVAL:
        for ang in ANGLES_EVAL:
            tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
            print(f"  -- {tag}")
            t0 = time.time()
            out = run_forward(rho_flat, lam, ang)
            dt = time.time() - t0

            n = len(out["I_focal"])
            x_focal = np.linspace(-SX/2, SX/2, n)
            metrics = compute_metrics(out["I_focal"], x_focal)

            results[tag] = {
                "lam_nm": int(lam * 1000),
                "ang_deg": ang,
                "I_focal": out["I_focal"].tolist(),
                "x_focal": x_focal.tolist(),
                "time": dt,
                **metrics,
            }
            field_maps[tag] = out["mag_full"]

            print(f"     time={dt:.1f}s, peak={metrics['peak_center']:.2e}, "
                  f"conc={metrics['concentration']:.3f}, FWHM={metrics['fwhm_um']*1000:.0f}nm")

    total_t = time.time() - total_t0

    # --- Save ---
    if mp.am_master():
        # ρ design
        np.save(os.path.join(OUT_DIR, "rho_design.npy"), rho)

        # Summary
        with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
            f.write("Fresnel zone plate -- analytical diffractive baseline\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Design:\n")
            f.write(f"  Aperture     : {DESIGN_X} μm\n")
            f.write(f"  Focal length : {FOCAL_LENGTH} μm (f/{FOCAL_LENGTH/DESIGN_X:.2f})\n")
            f.write(f"  Design λ     : {DESIGN_LAMBDA*1000:.0f} nm\n")
            f.write(f"  Fill ratio   : {rho.mean():.3f}\n")
            f.write(f"  Pattern      : Binary Fresnel zone plate\n")
            f.write(f"  Total eval time: {total_t:.1f}s\n\n")

            f.write(f"{'condition':<14} {'peak':>10} {'conc':>7} {'FWHM/nm':>9} "
                    f"{'E_center':>10} {'E_total':>10}\n")
            for tag in sorted(results.keys()):
                r = results[tag]
                f.write(f"{tag:<14} {r['peak_center']:>10.2e} "
                        f"{r['concentration']:>7.3f} "
                        f"{r['fwhm_um']*1000:>9.0f} "
                        f"{r['energy_center']:>10.2e} "
                        f"{r['energy_total']:>10.2e}\n")

            # Aggregated stats
            peaks = [r['peak_center'] for r in results.values()]
            concs = [r['concentration'] for r in results.values()]
            f.write(f"\nAggregated:\n")
            f.write(f"  Mean peak   : {np.mean(peaks):.2e}\n")
            f.write(f"  Mean conc   : {np.mean(concs):.3f}\n")
            f.write(f"  By λ (mean conc):\n")
            for lam in WAVELENGTHS_EVAL:
                lam_concs = [r['concentration'] for tag, r in results.items()
                             if r['lam_nm'] == int(lam*1000)]
                f.write(f"    {LAM_NAMES[lam]}: {np.mean(lam_concs):.3f}\n")
            f.write(f"  By angle (mean conc):\n")
            for ang in ANGLES_EVAL:
                ang_concs = [r['concentration'] for tag, r in results.items()
                             if r['ang_deg'] == ang]
                f.write(f"    {ang:+d}°: {np.mean(ang_concs):.3f}\n")

        # ρ pattern + phase
        fig, axes = plt.subplots(1, 2, figsize=(12, 3))
        im = axes[0].imshow(
            rho.T, cmap="gray_r", aspect="auto", origin="lower",
            extent=[-DESIGN_X/2, DESIGN_X/2, -DESIGN_Y/2, DESIGN_Y/2],
            vmin=0, vmax=1,
        )
        axes[0].set_title(f"Fresnel ρ design (Fresnel zone plate, λ_d={DESIGN_LAMBDA*1000:.0f}nm)")
        axes[0].set_xlabel("x (μm)")
        axes[0].set_ylabel("y (μm)")
        plt.colorbar(im, ax=axes[0], label="ρ")

        axes[1].plot(x_pixels, np.unwrap(phase) / np.pi, lw=1.2)
        axes[1].set_xlabel("x (μm)")
        axes[1].set_ylabel("Phase / π")
        axes[1].set_title("Designed phase profile (hyperboloidal)")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "rho_design.png"), dpi=120)
        plt.close()

        # Field maps grid (3 λ × 3 angles)
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        for i, lam in enumerate(WAVELENGTHS_EVAL):
            for j, ang in enumerate(ANGLES_EVAL):
                tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
                ax = axes[i, j]
                mag = field_maps[tag]
                x_axis = np.linspace(-SX/2, SX/2, mag.shape[0])
                y_axis = np.linspace(-SY/2, SY/2, mag.shape[1])
                im = ax.imshow(
                    mag.T,
                    extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
                    origin="lower", cmap="inferno", aspect="auto",
                )
                ax.axhline(DESIGN_Y/2, color="white", ls=":", lw=0.5, alpha=0.5)
                ax.axhline(-DESIGN_Y/2, color="white", ls=":", lw=0.5, alpha=0.5)
                ax.axhline(MON_Y, color="lime", ls="--", lw=0.8, alpha=0.7)
                ax.set_title(f"{LAM_NAMES[lam]}, {ang:+d}°")
                if i == 2:
                    ax.set_xlabel("x (μm)")
                if j == 0:
                    ax.set_ylabel("y (μm)")
                plt.colorbar(im, ax=ax, fraction=0.04)
        plt.suptitle("Fresnel zone plate: |Ez| field maps (3 lambda x 3 angles)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "field_maps_3x3.png"), dpi=100)
        plt.close()

        # Focal intensity profiles (3 λ at 0°)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        for ax, lam in zip(axes, WAVELENGTHS_EVAL):
            for ang in ANGLES_EVAL:
                tag = f"{LAM_NAMES[lam]}_ang{ang:+d}"
                r = results[tag]
                ax.plot(r["x_focal"], r["I_focal"], lw=1.0, label=f"{ang:+d}°")
            ax.set_xlabel("x (μm)")
            ax.set_ylabel("|E|²")
            ax.set_title(f"{LAM_NAMES[lam]}, focal plane")
            ax.set_xlim(-2.5, 2.5)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        plt.suptitle("Fresnel focal intensity profiles")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "focal_profiles.png"), dpi=120)
        plt.close()

        # Save raw results for later comparison
        np.savez(
            os.path.join(OUT_DIR, "c1_results.npz"),
            rho=rho,
            **{f"I_focal_{tag}": np.array(r["I_focal"]) for tag, r in results.items()},
            **{f"x_focal_{tag}": np.array(r["x_focal"]) for tag, r in results.items()},
        )

    # --- Print summary ---
    peaks = [r['peak_center'] for r in results.values()]
    concs = [r['concentration'] for r in results.values()]

    print("\n" + "=" * 60)
    print(" Fresnel baseline done")
    print("=" * 60)
    print(f"  Total time     : {total_t:.1f}s")
    print(f"  Mean peak |E|² : {np.mean(peaks):.2e}")
    print(f"  Mean concentration: {np.mean(concs):.3f}")
    print(f"  By λ:")
    for lam in WAVELENGTHS_EVAL:
        lam_concs = [r['concentration'] for tag, r in results.items()
                     if r['lam_nm'] == int(lam*1000)]
        print(f"    {LAM_NAMES[lam]}: conc={np.mean(lam_concs):.3f}")
    print(f"\n  Results: {OUT_DIR}/")
    print(f"    summary.txt              -- numeric summary")
    print(f"    rho_design.png           -- Fresnel zone plate ρ + phase")
    print(f"    field_maps_3x3.png       -- 3λ × 3angle field maps")
    print(f"    focal_profiles.png       -- Focal intensity profiles")
    print(f"    c1_results.npz           -- raw data for Focus-opt/CODA comparison")


if __name__ == "__main__":
    main()