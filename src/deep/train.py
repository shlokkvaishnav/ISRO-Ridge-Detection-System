"""Training loop for Arm C. Device-agnostic (uses CUDA if available, CPU
otherwise) so the same script runs as a local smoke test and as the real
training run on a Kaggle GPU notebook (see
research/arm_c_deep_learning/kaggle_notebook.py).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, random_split

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
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = RidgeTileDataset(manifest_path, tiles_dir, shp_path)
    n_val = max(int(len(dataset) * val_fraction), 1)
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    print(f"Dataset: {len(dataset)} tiles ({n_train} train / {n_val} val)")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = DualBranchRidgeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

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

    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    val_ids = [dataset.manifest[i]["id"] for i in val_set.indices]
    with open(os.path.join(out_dir, "val_ids.json"), "w") as f:
        json.dump(val_ids, f, indent=2)

    print(f"Saved model, history, and val split to {out_dir}")
    return {"history": history, "val_ids": val_ids}


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
