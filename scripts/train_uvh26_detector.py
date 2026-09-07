"""Phase 7C: fine-tune the EXISTING Phase 6 detector on the 750-image UVH-26 subset.

    python scripts/train_uvh26_detector.py --epochs 3

WHAT CHANGES AND WHAT DOES NOT
    The architecture is the Phase 6 one, unchanged: Faster R-CNN MobileNetV3-Large-320-FPN. The
    only structural edit is the final box predictor, which must go from COCO's 91 outputs to our
    7, because a head that predicts 91 classes cannot predict AUTO_RICKSHAW. Everything else,
    including the COCO weights, is inherited.

    The backbone is FROZEN for this first run (`trainable_backbone_layers=0`). With 500 training
    images, unfreezing a feature extractor trained on 118,000 is the fastest route to overfitting,
    and the question this run asks is narrow: does a small amount of Indian-scene supervision move
    the head at all, without wrecking what the backbone already knows.

MODEL SELECTION
    On validation mAP, never on the test split, which is not loaded by this script at all. The
    validation set exists to choose an epoch and nothing else.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader                                  # noqa: E402
from torchvision.models import detection as tvd                          # noqa: E402
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor   # noqa: E402

from perception_detector.detect_eval import evaluate                     # noqa: E402
from perception_detector.uvh26 import (CLASS_NAMES, NUM_CLASSES, UVH26Subset,  # noqa: E402
                                       collate, load_manifest)

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "perception_detector" / "checkpoints"


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 4
    lr: float = 0.002                  # torchvision's 0.02 reference is for batch 16 from scratch
    momentum: float = 0.9
    weight_decay: float = 5e-4
    trainable_backbone_layers: int = 0
    seed: int = 20260907
    score_threshold: float = 0.35
    iou_threshold: float = 0.5
    augment_hflip: bool = True
    model_name: str = "fasterrcnn_mobilenet_v3_large_320_fpn"


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)   # some detection ops have no deterministic kernel


def build_model(cfg: TrainConfig) -> torch.nn.Module:
    w = tvd.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    m = getattr(tvd, cfg.model_name)(
        weights=w, trainable_backbone_layers=cfg.trainable_backbone_layers)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    return m


def predictor_for(model: torch.nn.Module, score_thr: float):
    """Wrap a model as the `predict(path)` callable `detect_eval.evaluate` expects."""
    from PIL import Image

    def predict(path):
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(img.transpose(2, 0, 1))
        model.eval()
        with torch.no_grad():
            o = model([x])[0]
        return (o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy())
    return predict


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.002)
    a = ap.parse_args()
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr)
    set_seed(cfg.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    man = load_manifest()
    train = UVH26Subset("train", man, augment=cfg.augment_hflip, seed=cfg.seed)
    val = UVH26Subset("val", man, augment=False, seed=cfg.seed)
    print(f"train {len(train)} images / {sum(len(v[0]) for v in train.boxes.values())} boxes")
    print(f"val   {len(val)} images / {sum(len(v[0]) for v in val.boxes.values())} boxes")
    print("classes:", {i: n for i, n in enumerate(CLASS_NAMES)})
    print("THE TEST SPLIT IS NOT LOADED BY THIS SCRIPT.", flush=True)

    model = build_model(cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"\n{cfg.model_name}: {sum(p.numel() for p in model.parameters())/1e6:.2f} M params, "
          f"{sum(p.numel() for p in trainable)/1e6:.2f} M trainable "
          f"(backbone layers trainable = {cfg.trainable_backbone_layers})")

    opt = torch.optim.SGD(trainable, lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg.seed))

    history, best = [], -1.0
    ckpt_path = CKPT_DIR / "fasterrcnn_uvh26.pt"
    t_start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n, t0 = 0.0, 0, time.perf_counter()
        for k, (imgs, tgts) in enumerate(loader):
            losses = model(list(imgs), list(tgts))
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(imgs); n += len(imgs)
            if (k + 1) % 25 == 0:
                print(f"    epoch {epoch} step {k+1}/{len(loader)} "
                      f"loss {total/max(n,1):.4f} [{time.perf_counter()-t0:.0f}s]", flush=True)
        train_loss = total / max(n, 1)
        train_s = time.perf_counter() - t0

        v = evaluate(predictor_for(model, cfg.score_threshold), val, CLASS_NAMES,
                     cfg.iou_threshold, cfg.score_threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_seconds": train_s,
                        "val_mAP": v["mAP"], "val_precision": v["precision"],
                        "val_recall": v["recall"],
                        "val_per_class_recall": {k: x["recall"] for k, x in v["per_class"].items()}})
        flag = ""
        if v["mAP"] == v["mAP"] and v["mAP"] > best:
            best = v["mAP"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg),
                        "class_names": CLASS_NAMES, "epoch": epoch, "val": v,
                        "manifest": str(ROOT / "docs" / "uvh26_manifest.json"),
                        "git_commit": git_hash()}, ckpt_path)
            flag = "  <- best, saved"
        print(f"epoch {epoch}/{cfg.epochs}  train loss {train_loss:.4f}  "
              f"val mAP {v['mAP']:.4f}  P {v['precision']:.3f}  R {v['recall']:.3f}  "
              f"[{train_s:.0f}s train]{flag}", flush=True)

    mins = (time.perf_counter() - t_start) / 60.0
    print(f"\ntrained in {mins:.1f} min; best validation mAP {best:.4f}")
    (CKPT_DIR / "uvh26_history.json").write_text(
        json.dumps({"config": asdict(cfg), "git_commit": git_hash(),
                    "class_names": CLASS_NAMES, "history": history,
                    "total_minutes": mins}, indent=2), encoding="utf-8")
    print(f"checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
