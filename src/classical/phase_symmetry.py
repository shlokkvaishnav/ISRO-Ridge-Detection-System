"""Monogenic phase symmetry (Kovesi, 1997) for ridge detection.

Implemented directly rather than depending on the `phasepack` package (last
released ~2015; uses numpy APIs removed since ~1.24 and does not install
cleanly against current Python/numpy — see research/DECISION_LOG.md for the
verification that led to this choice).

This is the log-Gabor + Riesz-transform monogenic-signal formulation used by
the reference lunar wrinkle ridge literature: it responds to locally
symmetric structure (a ridge or a valley) independent of illumination
direction or contrast polarity, which is why it's preferred over plain
gradient-based edge detection for degraded/eroded ridges.
"""
from __future__ import annotations

import numpy as np


def monogenic_phase_symmetry(
    image: np.ndarray,
    n_scales: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    noise_threshold: float = 2.0,
) -> np.ndarray:
    """Compute a monogenic phase-symmetry response map.

    Parameters
    ----------
    image:
        2D grayscale input (e.g. the slope grayscale from
        ``dem_to_slope.slope_to_grayscale``).
    n_scales:
        Number of log-Gabor filter scales to sum over.
    min_wavelength:
        Wavelength (pixels) of the smallest-scale filter.
    mult:
        Wavelength scaling factor between successive scales.
    sigma_onf:
        Log-Gabor filter's radial bandwidth (smaller = narrower band).
    noise_threshold:
        Fixed per-scale energy floor subtracted before summing. This is a
        simplified stand-in for Kovesi's adaptive noise estimate (computed
        from the smallest scale's median response) — see
        research/DECISION_LOG.md for why a fixed threshold was used first,
        and adaptive estimation is a documented follow-up.

    Returns
    -------
    Float array, same shape as ``image``, normalized to [0, 1], where higher
    values indicate stronger local symmetry.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")

    rows, cols = image.shape
    img = image.astype(np.float64)
    IMG = np.fft.fft2(img)

    u = np.fft.fftfreq(cols)
    v = np.fft.fftfreq(rows)
    ux, vy = np.meshgrid(u, v)
    radius = np.sqrt(ux**2 + vy**2)
    radius[0, 0] = 1.0  # avoid divide-by-zero; DC bin is excluded from every filter below

    # Riesz transform kernels: the 2D, orientation-free analogue of the
    # Hilbert transform, giving the two odd (antisymmetric) components of
    # the monogenic signal.
    riesz_x = 1j * ux / radius
    riesz_y = 1j * vy / radius

    total_energy = np.zeros((rows, cols))
    total_amplitude = np.zeros((rows, cols)) + 1e-6

    wavelength = min_wavelength
    for _ in range(n_scales):
        center_freq = 1.0 / wavelength
        log_gabor = np.exp(
            -(np.log(radius / center_freq)) ** 2 / (2 * np.log(sigma_onf) ** 2)
        )
        log_gabor[0, 0] = 0.0

        even = np.real(np.fft.ifft2(IMG * log_gabor))
        odd1 = np.real(np.fft.ifft2(IMG * log_gabor * riesz_x))
        odd2 = np.real(np.fft.ifft2(IMG * log_gabor * riesz_y))

        odd_amplitude = np.sqrt(odd1**2 + odd2**2)
        amplitude = np.sqrt(even**2 + odd1**2 + odd2**2)

        energy = np.abs(even) - odd_amplitude - noise_threshold
        total_energy += np.maximum(energy, 0.0)
        total_amplitude += amplitude

        wavelength *= mult

    symmetry = total_energy / total_amplitude
    peak = symmetry.max()
    if peak > 0:
        symmetry = symmetry / peak
    return symmetry
