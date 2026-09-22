"""CLI entrypoint for Arm B: Hessian ridge filter (Frangi/Meijering/Sato) +
morphology.

Usage:
    python scripts/run_hessian_arm.py --dem path/to/dem.tif --out results/
    python scripts/run_hessian_arm.py --out results/ --method meijering
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_classical_arm import load_dem, synthetic_fallback_dem  # noqa: E402
from src.hessian.pipeline import detect_ridges  # noqa: E402


def save_visualization(result: dict, out_dir: str) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, key in zip(axes, ["slope", "grayscale", "hessian_response", "ridge_mask"]):
        ax.imshow(result[key], cmap="gray")
        ax.set_title(key)
        ax.axis("off")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "hessian_arm_stages.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=str, default=None, help="path to a DEM GeoTIFF")
    parser.add_argument("--out", type=str, default="results", help="output directory")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--method", type=str, default="frangi", choices=["frangi", "meijering", "sato"])
    parser.add_argument("--threshold-percentile", type=float, default=75.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dem:
        elevation = load_dem(args.dem)
        print(f"Loaded DEM from {args.dem}, shape={elevation.shape}")
    else:
        elevation = synthetic_fallback_dem()
        print("No --dem given; using synthetic fallback DEM (curved ridge + noise)")

    result = detect_ridges(
        elevation,
        pixel_size=args.pixel_size,
        method=args.method,
        threshold_percentile=args.threshold_percentile,
    )

    mask_path = os.path.join(args.out, "ridge_mask.npy")
    np.save(mask_path, result["ridge_mask"])
    print(f"Ridge mask saved to {mask_path} ({result['ridge_mask'].sum()} ridge pixels)")

    viz_path = save_visualization(result, args.out)
    print(f"Stage visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
