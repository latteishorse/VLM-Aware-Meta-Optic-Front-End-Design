"""Array-level Focus-opt objective from manuscript Eq. (9), without Meep imports."""

import numpy as np
from autograd import numpy as npa


FOCUS_WINDOW_HALF_WIDTH_UM = 0.25
FOCUS_EPS = 1e-20
FOCUS_OBJECTIVE_NAME = "normalized_focal_concentration_eq9_v1"


def focus_window_mask(x_coordinates):
    """Select the fixed central 0.5 um window on actual Meep sample coordinates."""
    x = np.asarray(x_coordinates, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional sensor coordinate array")
    mask = np.abs(x) <= FOCUS_WINDOW_HALF_WIDTH_UM
    if not mask.any():
        raise ValueError("The sensor grid has no samples in the focal window")
    return mask


def normalized_focal_fraction(psf, window_mask, epsilon=FOCUS_EPS):
    """Return one positive Eq. (9) term with a safe full-line denominator.

    This is a discrete sample-sum ratio, not a quadrature-weighted integral.
    ``psf`` is one full-line intensity array; ``window_mask`` is fixed geometry.
    Meep differentiates through both the numerator and denominator. Training
    averages the nine terms equally and minimizes the negative of that mean.
    """
    intensity = npa.reshape(psf, (-1,))
    mask = np.asarray(window_mask, dtype=float).reshape(-1)
    if intensity.shape != mask.shape:
        raise ValueError(f"PSF/mask shape mismatch: {intensity.shape} versus {mask.shape}")
    numerator = npa.sum(intensity * mask)
    denominator = npa.maximum(npa.sum(intensity), epsilon)
    return numerator / denominator


def focal_concentration_from_fields(ez, window_mask, epsilon=FOCUS_EPS):
    """Autograd-compatible complex-field objective, including zero field values."""
    intensity = npa.real(ez * npa.conj(ez))
    return normalized_focal_fraction(intensity, window_mask, epsilon)
