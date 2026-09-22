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

# Diagnostic printed unconditionally, not just on failure: a prior run hit a
# FileNotFoundError for MANIFEST here, and the dataset turned out to
# genuinely contain the file (confirmed independently afterward) -- most
# likely a dataset-processing-not-finished-yet timing issue on Kaggle's
# side, not a real bug in this script. Printing the actual mount contents
# up front makes that distinction obvious from the log next time, instead
# of guessing again.
print(f"Contents of {DATASET_DIR}: {sorted(os.listdir(DATASET_DIR)) if os.path.exists(DATASET_DIR) else 'DOES NOT EXIST'}")
if not os.path.exists(MANIFEST):
    raise FileNotFoundError(
        f"{MANIFEST} not found. Dataset mount contents: "
        f"{sorted(os.listdir(DATASET_DIR)) if os.path.exists(DATASET_DIR) else 'DATASET_DIR missing entirely'}. "
        "If this dataset was just created/updated, it may not have finished "
        "processing yet -- check https://www.kaggle.com/datasets/shlokkvaishnav/isro-ridge-tiles "
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
