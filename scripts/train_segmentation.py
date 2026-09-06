"""Train the IDD-Lite drivable-space segmenter. See road_perception/train.py."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from road_perception.train import TrainConfig, train      # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=TrainConfig.root)
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--width", type=float, default=TrainConfig.width)
    ap.add_argument("--boundary-weight", type=float, default=TrainConfig.boundary_weight)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--checkpoint", default=TrainConfig.checkpoint)
    a = ap.parse_args()
    train(TrainConfig(root=a.root, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                      width=a.width, boundary_weight=a.boundary_weight, seed=a.seed,
                      checkpoint=a.checkpoint))
