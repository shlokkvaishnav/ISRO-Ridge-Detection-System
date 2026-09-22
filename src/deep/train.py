"""Training loop for Arm C. Device-agnostic (uses CUDA if available, CPU
otherwise) so the same script runs as a local smoke test and as the real
training run on a Kaggle GPU notebook (see
research/arm_c_deep_learning/kaggle_notebook.py).
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.deep.dataset import RidgeTileDataset
from src.deep.model import DualBranchRidgeNet


def train(
    manifest_path: str,
    tiles_dir: str,
    shp_path: str,
    out_dir: str,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 0,
    holdout_ids: Optional[Sequence[int]] = None,
) -> dict:
    """holdout_ids: if given, these tile ids are excluded from training and
    used as the *entire* validation set (val_fraction is ignored), instead
    of a random split. Needed for a fair three-way arm comparison: Arm A/B
    were evaluated on a specific 6-tile set (research/DECISION_LOG.md), and
    a random split could put some of those tiles in Arm C's training set,
    which would make comparing recall on them meaningless."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RidgeTileDataset(manifest_path, tiles_dir, shp_path)

    if holdout_ids:
        holdout_set = set(holdout_ids)
        train_indices = [i for i, e in enumerate(dataset.manifest) if e["id"] not in holdout_set]
        val_indices = [i for i, e in enumerate(dataset.manifest) if e["id"] in holdout_set]
        missing = holdout_set - {dataset.manifest[i]["id"] for i in val_indices}
        if missing:
            raise ValueError(f"holdout_ids not found in manifest: {missing}")
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        print(f"Dataset: {len(dataset)} tiles ({len(train_set)} train / {len(val_set)} held out by id)")
    else:
        n_val = max(int(len(dataset) * val_fraction), 1)
        n_train = len(dataset) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
        print(f"Dataset: {len(dataset)} tiles ({n_train} train / {n_val} val, random split)")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = DualBranchRidgeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Track the best-val-loss checkpoint alongside the final one, in the
    # same run: the merged Arm C result (research/DECISION_LOG.md,
    # 2026-09-22) only saved the final (epoch 60) checkpoint despite val
    # loss peaking at epoch 4, leaving "would the loss-optimal checkpoint
    # give a similar recall?" unanswered. Saving both from one run answers
    # it without a second training run's seed variance confounding the
    # comparison (research/arm_c_early_stopping/SPEC.md).
    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = None

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for dem, aspect, label, _ids in train_loader:
            dem, aspect, label = dem.to(device), aspect.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(dem, aspect)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * dem.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dem, aspect, label, _ids in val_loader:
                dem, aspect, label = dem.to(device), aspect.to(device), label.to(device)
                pred = model(dem, aspect)
                val_loss += loss_fn(pred, label).item() * dem.size(0)
        val_loss /= len(val_set)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    if best_state_dict is not None:
        torch.save(best_state_dict, os.path.join(out_dir, "model_best.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    val_ids = [dataset.manifest[i]["id"] for i in val_set.indices]
    with open(os.path.join(out_dir, "val_ids.json"), "w") as f:
        json.dump(val_ids, f, indent=2)
    with open(os.path.join(out_dir, "best_epoch.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_loss": best_val_loss, "final_epoch": epochs}, f, indent=2)

    print(f"Saved model (final + best, epoch {best_epoch}), history, and val split to {out_dir}")
    return {"history": history, "val_ids": val_ids, "best_epoch": best_epoch}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/tiles/manifest.json")
    parser.add_argument("--tiles-dir", default="data/tiles")
    parser.add_argument("--shp", default="data/raw/wrinkle_ridges_shapefile/WRINKLE_RIDGES_180.SHP")
    parser.add_argument("--out-dir", default="results/arm_c")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        manifest_path=args.manifest,
        tiles_dir=args.tiles_dir,
        shp_path=args.shp,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
