"""Hessian-eigenvalue-based ridge filters (Frangi, Meijering, Sato) for
lunar wrinkle ridge detection -- Arm B, the same family of filters used for
vessel detection in medical imaging, applied here to a slope map instead.

Unlike Arm A's phase symmetry (custom-implemented, see
src/classical/phase_symmetry.py), these are already in scikit-image, so
this module is a thin wrapper rather than a from-scratch implementation --
there's no correctness-risk reason to reimplement well-tested, standard
filters that ship in a library this project already depends on.
"""
from __future__ import annotations

import numpy as np
from skimage.filters import frangi, meijering, sato

FILTERS = {
    "frangi": frangi,
    "meijering": meijering,
    "sato": sato,
}


def hessian_ridge_response(
    image: np.ndarray,
    method: str = "frangi",
    sigmas: range = range(1, 6),
    black_ridges: bool = False,
) -> np.ndarray:
    """Compute a Hessian-based ridge response map.

    Parameters
    ----------
    image:
        2D grayscale input (the same slope grayscale Arm A uses, via
        `dem_to_slope.slope_to_grayscale`, for a fair comparison).
    method:
        One of "frangi", "meijering", "sato".
    sigmas:
        Range of scales the filter is evaluated at, analogous to Arm A's
        `n_scales`/`min_wavelength`/`mult` sweep.
    black_ridges:
        Whether ridges are darker than their surroundings. A slope map's
        ridges (high-slope flanks) are *brighter* than flat surroundings,
        so this defaults to False -- opposite of the typical dark-vessel
        medical-imaging default in scikit-image.

    Returns
    -------
    Float array, same shape as `image`, response normalized to [0, 1].
    """
    if method not in FILTERS:
        raise ValueError(f"unknown method {method!r}, expected one of {list(FILTERS)}")

    response = FILTERS[method](image.astype(np.float64), sigmas=sigmas, black_ridges=black_ridges)

    peak = response.max()
    if peak > 0:
        response = response / peak
    return response
