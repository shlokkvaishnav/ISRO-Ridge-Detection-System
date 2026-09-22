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
DATASET_DIR = "/kaggle/input/isro-ridge-tiles"

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

MANIFEST = os.path.join(DATASET_DIR, "manifest.json")
TILES_DIR = DATASET_DIR
SHP = os.path.join(DATASET_DIR, "shapefile", "WRINKLE_RIDGES_180.SHP")
OUT_DIR = "/kaggle/working/arm_c"

# Diagnostics printed unconditionally, not just on failure: two prior runs
# each hit a different mount-path failure (one where DATASET_DIR existed but
# manifest.json didn't -- likely a dataset-processing timing issue; one
# where DATASET_DIR itself didn't exist at all, despite server-side kernel
# metadata confirming the dataset was attached -- cause still unclear).
# Print what /kaggle/input actually contains rather than assuming the exact
# subfolder name matches the dataset slug, so a naming mismatch (rather than
# a missing/unready dataset) is immediately visible in the log next time.
print(f"/kaggle/input contents: {sorted(os.listdir('/kaggle/input')) if os.path.exists('/kaggle/input') else 'DOES NOT EXIST'}")
if os.path.exists(DATASET_DIR):
    print(f"Contents of {DATASET_DIR}: {sorted(os.listdir(DATASET_DIR))}")

if not os.path.exists(MANIFEST):
    raise FileNotFoundError(
        f"{MANIFEST} not found. /kaggle/input contents: "
        f"{sorted(os.listdir('/kaggle/input')) if os.path.exists('/kaggle/input') else '/kaggle/input missing entirely'}. "
        "If DATASET_DIR's name doesn't match what's listed above, fix "
        "DATASET_DIR in this script. If nothing is listed at all, the "
        "dataset may not be attached/ready -- check "
        "https://www.kaggle.com/datasets/shlokkvaishnav/isro-ridge-tiles "
        "shows a completed version before re-running."
    )

result = train(
    manifest_path=MANIFEST,
    tiles_dir=TILES_DIR,
    shp_path=SHP,
    out_dir=OUT_DIR,
    epochs=60,
    batch_size=8,
    lr=1e-3,
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
