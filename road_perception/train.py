"""Train Fast-SCNN on IDD-Lite. Reproducible: fixed seed, recorded configuration.

    python scripts/train_segmentation.py --root Datasets/idd-lite/idd20k_lite --epochs 60

Selection is on validation DRIVABLE IoU, not on loss and not on pixel accuracy, because drivable
IoU is what the downstream corridor estimate depends on.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import IDDLite, NUM_CLASSES, class_pixel_counts, class_weights
from .evaluate import SegmentationMetrics, format_summary
from .losses import DrivableSegmentationLoss
from .model import FastSCNN, parameter_count


@dataclass
class TrainConfig:
    root: str = "Datasets/idd-lite/idd20k_lite"
    epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-3
    weight_decay: float = 1e-4
    width: float = 1.0
    boundary_weight: float = 3.0
    weight_cap: float = 12.0
    seed: int = 1234
    workers: int = 0                     # 0 keeps it deterministic and avoids Windows spawn cost
    checkpoint: str = "road_perception/checkpoints/fastscnn_iddlite.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)     # cudnn determinism is moot on CPU


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    metrics = SegmentationMetrics()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            metrics.update(pred, y.numpy())
    return metrics.summary()


def train(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cpu")
    root = Path(cfg.root)

    train_ds = IDDLite(root, "train", augment=True, seed=cfg.seed)
    val_ds = IDDLite(root, "val", augment=False)
    print(f"train {len(train_ds)}   val {len(val_ds)}")

    counts = class_pixel_counts(train_ds)
    weights = class_weights(counts, cap=cfg.weight_cap)
    print("class pixel share: " + ", ".join(
        f"{n} {100.0 * c / counts.sum():.1f}%" for n, c in
        zip(["driv", "nondriv", "living", "veh", "roadside", "far", "sky"], counts)))
    print("class weights:     " + ", ".join(f"{w:.2f}" for w in weights))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.workers)

    model = FastSCNN(NUM_CLASSES, width=cfg.width).to(device)
    print(f"Fast-SCNN width {cfg.width}: {parameter_count(model) / 1e6:.2f} M parameters")

    criterion = DrivableSegmentationLoss(torch.from_numpy(weights),
                                         boundary_weight=cfg.boundary_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    ckpt = Path(cfg.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            opt.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            total += float(loss) * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = total / max(n, 1)

        s = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": train_loss,
                        "drivable_iou": s["drivable_iou"], "mean_iou": s["mean_iou"],
                        "boundary_f1": s["boundary_f1"]})
        flag = ""
        if s["drivable_iou"] is not None and s["drivable_iou"] > best:
            best = s["drivable_iou"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "val": s, "epoch": epoch}, ckpt)
            flag = "  <- best"
        print(f"epoch {epoch:3d}/{cfg.epochs}  loss {train_loss:.4f}  "
              f"drivable IoU {s['drivable_iou']:.4f}  mIoU {s['mean_iou']:.4f}  "
              f"boundary F1 {s['boundary_f1']:.4f}  [{time.perf_counter() - t0:.0f}s]{flag}",
              flush=True)

    minutes = (time.perf_counter() - t_start) / 60.0
    print(f"\ntrained in {minutes:.1f} min; best validation drivable IoU {best:.4f}")
    print(f"checkpoint: {ckpt}")

    best_state = torch.load(ckpt, map_location=device, weights_only=False)
    print("\n" + format_summary(best_state["val"]))
    (ckpt.parent / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_state["val"]
