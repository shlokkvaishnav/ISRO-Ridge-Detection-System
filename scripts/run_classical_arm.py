"""CLI entrypoint for Arm A: classical phase-symmetry + morphology ridge
detection.

Usage:
    python scripts/run_classical_arm.py --dem path/to/dem.tif --out results/
    python scripts/run_classical_arm.py --out results/   # synthetic fallback DEM
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classical.pipeline import detect_ridges  # noqa: E402


def synthetic_fallback_dem(size: int = 256) -> np.ndarray:
    """A curved ridge on a flat plain with a little noise, used when no real
    DEM is given yet -- lets this script run end-to-end before any real
    lunar DEM tile has been downloaded."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:size, 0:size]
    centerline = size / 2 + 30 * np.sin(2 * np.pi * x[0] / size)
    ridge = 12.0 * np.exp(-((y - centerline[np.newaxis, :]) ** 2) / (2 * 4.0**2))
    noise = rng.normal(0.0, 0.05, size=(size, size))
    return ridge + noise


def load_dem(path: str) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64)


def save_visualization(result: dict, out_dir: str) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, key in zip(axes, ["slope", "grayscale", "phase_symmetry", "ridge_mask"]):
        ax.imshow(result[key], cmap="gray")
        ax.set_title(key)
        ax.axis("off")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "classical_arm_stages.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=str, default=None, help="path to a DEM GeoTIFF")
    parser.add_argument("--out", type=str, default="results", help="output directory")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dem:
        elevation = load_dem(args.dem)
        print(f"Loaded DEM from {args.dem}, shape={elevation.shape}")
    else:
        elevation = synthetic_fallback_dem()
        print("No --dem given; using synthetic fallback DEM (curved ridge + noise)")

    result = detect_ridges(elevation, pixel_size=args.pixel_size, threshold_percentile=args.threshold_percentile)

    mask_path = os.path.join(args.out, "ridge_mask.npy")
    np.save(mask_path, result["ridge_mask"])
    print(f"Ridge mask saved to {mask_path} ({result['ridge_mask'].sum()} ridge pixels)")

    viz_path = save_visualization(result, args.out)
    print(f"Stage visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
