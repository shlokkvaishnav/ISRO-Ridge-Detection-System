"""Kaggle kernel entrypoint for Arm C training.

Runs on Kaggle's GPU, with the tile dataset attached as a Kaggle Dataset
(see research/arm_c_deep_learning/SPEC.md -- training happens here rather
than locally because this machine has no discrete GPU, only integrated
Intel graphics).

Clones this repo at runtime for the actual model/training code rather than
duplicating it into the kernel, so there's exactly one copy of
src/deep/*.py to keep correct.
"""
import os
import subprocess
import sys

REPO_URL = "https://github.com/shlokkvaishnav/ISRO-Ridge-Detection-System.git"
REPO_BRANCH = "research/arm-c-deep-learning"  # NOT master -- this code hasn't
# been reviewed/merged yet (research/GIT_WORKFLOW.md's full spec -> PR ->
# review -> manual-merge path). A plain clone would silently pull master and
# miss every file this script needs; must pin the branch explicitly.
REPO_DIR = "/kaggle/working/repo"
DATASET_SLUG = "isro-ridge-tiles"

subprocess.run(["pip", "install", "-q", "pyshp"], check=True)

if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR],
        check=True,
    )

sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from src.deep.train import train  # noqa: E402
from src.deep.evaluate import evaluate  # noqa: E402


def find_dataset_dir(slug: str) -> str:
    """Locate the mounted dataset directory by walking /kaggle/input for a
    directory containing manifest.json, rather than assuming a fixed mount
    path. Three prior runs each hit a different mount-path assumption
    failure -- the classic /kaggle/input/<slug>/ layout, a dataset-still-
    processing timing issue, and (the actual cause, confirmed here) a newer
    Kaggle layout that nests datasets under /kaggle/input/datasets/<owner>/
    <slug>/ instead. Searching directly is more robust than hardcoding
    whichever layout happened to be observed most recently.
    """
    for root, dirs, files in os.walk("/kaggle/input"):
        if "manifest.json" in files and slug in root:
            return root
    raise FileNotFoundError(
        f"No directory containing manifest.json found under /kaggle/input "
        f"with '{slug}' in its path. /kaggle/input tree: "
        + "\n".join(
            os.path.join(r, d) for r, dirs, _ in os.walk("/kaggle/input") for d in dirs
        )
    )


DATASET_DIR = find_dataset_dir(DATASET_SLUG)
print(f"Found dataset at: {DATASET_DIR}")

MANIFEST = os.path.join(DATASET_DIR, "manifest.json")
TILES_DIR = DATASET_DIR
SHP = os.path.join(DATASET_DIR, "shapefile", "WRINKLE_RIDGES_180.SHP")
OUT_DIR = "/kaggle/working/arm_c"

# The exact 6 tiles Arm A/B were compared on (research/DECISION_LOG.md,
# 2026-09-22 entries) -- held out from training entirely, not just from a
# random split, so Arm C's number on them is a fair three-way comparison
# rather than partly evaluating on tiles it was trained on.
BENCHMARK_HOLDOUT_IDS = [3674, 1851, 748, 5017, 4098, 3461]

result = train(
    manifest_path=MANIFEST,
    tiles_dir=TILES_DIR,
    shp_path=SHP,
    out_dir=OUT_DIR,
    epochs=60,
    batch_size=8,
    lr=1e-3,
    holdout_ids=BENCHMARK_HOLDOUT_IDS,
)

eval_result = evaluate(
    model_path=os.path.join(OUT_DIR, "model.pt"),
    manifest_path=MANIFEST,
    tiles_dir=TILES_DIR,
    shp_path=SHP,
    tile_ids=result["val_ids"],
)

import json  # noqa: E402

with open(os.path.join(OUT_DIR, "eval.json"), "w") as f:
    json.dump(eval_result, f, indent=2)

print(f"Recall: {eval_result['recall_frac']} ({100*eval_result['recall']:.1f}%)")
print(f"Coverage: {eval_result['coverage_pct']:.1f}%")
