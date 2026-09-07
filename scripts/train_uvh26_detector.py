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
import shutil
import subprocess
from collections import Counter
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
    sampler: str = "uniform"        # "uniform" reproduces Phase 7C exactly; "class_aware" is 7E
    weight_cap: float = 16.0        # see class_weights() for why, and for its ceiling
    run_name: str = "uvh26"


def class_weights(train, cap: float) -> tuple[dict, dict]:
    """Inverse box frequency per class, capped, then one weight per IMAGE.

    THE SAMPLING UNIT IS THE IMAGE, not the box. Oversampling boxes independently is impossible
    here anyway (a box cannot be drawn without its image) and would be wrong in principle: drawing
    an image drags all of its other objects along, so the honest unit is the image.

    An image takes the MAXIMUM weight over the classes it contains, so a frame holding both a
    bicycle and six motorcycles is drawn for the bicycle. That is the intent: the rare class is
    what makes the image valuable.

    THE CAP, AND THE CEILING IT REVEALS. Raw inverse frequency gives bicycle 22.8x motorcycle.
    Measured before training, the resulting box-level exposure is far smaller than that ratio
    suggests:

        cap      bicycle exposure     bicycle-image share of an epoch
          8              1.33x                    26.9%
         16              1.95x                    39.4%
         25 (uncapped)   2.38x                    48.2%

    Even UNCAPPED, bicycle boxes only become 2.4x more frequent. The reason is structural: the 101
    bicycle images hold 106 bicycles between them, about one each, alongside many motorcycles. So
    image-level oversampling can raise how often a bicycle is SEEN but cannot change the fact that
    each sighting brings a crowd of common classes with it. That is a property of the data, not of
    the sampler, and it bounds what this experiment can achieve.

    16 is the operating point: bicycle exposure nearly doubles and bicycle frames become 39% of an
    epoch, without letting 101 images become half of it. The cost is that truck exposure dips
    slightly (0.92x) because bicycle and bus frames crowd it out.
    """
    boxes = Counter()
    for _, labels in train.boxes.values():
        for i in labels:
            boxes[CLASS_NAMES[i]] += 1
    most = max(boxes.values())
    cw = {c: min(most / n, cap) for c, n in boxes.items()}
    iw = []
    for e in train.entries:
        present = train.boxes[e["image_id"]][1]
        names = {CLASS_NAMES[i] for i in present}
        iw.append(max(cw[n] for n in names) if names else 1.0)
    return cw, {"boxes": dict(boxes), "image_weights": iw}


def report_sampling(train, cw: dict, info: dict) -> dict:
    """What the sampler will ACTUALLY do, printed before a single step is taken."""
    w = np.asarray(info["image_weights"], dtype=float)
    p = w / w.sum()
    n = len(train)
    exp_boxes = Counter()
    for k, e in enumerate(train.entries):
        for i in train.boxes[e["image_id"]][1]:
            exp_boxes[CLASS_NAMES[i]] += p[k] * n
    out = {"class_weight": cw, "expected_boxes_per_epoch": dict(exp_boxes),
           "uniform_boxes_per_epoch": info["boxes"]}
    print()
    print(f"{'class':<16}{'boxes':>8}{'weight':>9}{'imgs w/ cls':>13}"
          f"{'uniform/ep':>12}{'sampled/ep':>12}{'ratio':>8}")
    for c in ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]:
        nb = info["boxes"].get(c, 0)
        imgs = sum(1 for e in train.entries
                   if c in {CLASS_NAMES[i] for i in train.boxes[e["image_id"]][1]})
        ex = exp_boxes.get(c, 0.0)
        print(f"{c:<16}{nb:>8}{cw.get(c,0):>9.2f}{imgs:>13}{nb:>12}{ex:>12.0f}"
              f"{ex/max(nb,1):>8.2f}x")
    share = sum(p[k] for k, e in enumerate(train.entries)
                if "BICYCLE" in {CLASS_NAMES[i] for i in train.boxes[e["image_id"]][1]})
    uni = 100 * sum(1 for e in train.entries
                    if 6 in train.boxes[e["image_id"]][1]) / len(train)
    print()
    print(f"bicycle-containing images are {100*share:.1f}% of each sampled epoch "
          f"(uniform would be {uni:.1f}%)")
    out["bicycle_image_share"] = float(share)
    return out


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
    ap.add_argument("--sampler", choices=["uniform", "class_aware"], default="uniform")
    ap.add_argument("--weight-cap", type=float, default=16.0)
    ap.add_argument("--run-name", default=None)
    a = ap.parse_args()
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, sampler=a.sampler,
                      weight_cap=a.weight_cap,
                      run_name=a.run_name or ("uvh26_oversampled" if a.sampler == "class_aware"
                                              else "uvh26"))
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

    sampling = None
    gen = torch.Generator().manual_seed(cfg.seed)
    if cfg.sampler == "class_aware":
        cw, info = class_weights(train, cfg.weight_cap)
        sampling = report_sampling(train, cw, info)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(info["image_weights"], dtype=torch.double),
            num_samples=len(train), replacement=True, generator=gen)
        loader = DataLoader(train, batch_size=cfg.batch_size, sampler=sampler,
                            collate_fn=collate)
    else:
        loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                            generator=gen)
    print()
    print(f"sampler: {cfg.sampler}", flush=True)

    history, best = [], -1.0
    ckpt_path = CKPT_DIR / f"fasterrcnn_{cfg.run_name}.pt"
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
        # EVERY epoch is kept. Selecting on mAP alone chose an epoch where bicycle recall was
        # still exactly zero, which is the wrong answer for a phase whose whole purpose is
        # rare-class recovery, and the losing epoch's weights had already been discarded.
        torch.save({"model": model.state_dict(), "config": asdict(cfg),
                    "class_names": CLASS_NAMES, "epoch": epoch, "val": v,
                    "manifest": str(ROOT / "docs" / "uvh26_manifest.json"),
                    "sampling": sampling, "git_commit": git_hash()},
                   CKPT_DIR / f"fasterrcnn_{cfg.run_name}_ep{epoch}.pt")
        flag = ""
        if v["mAP"] == v["mAP"] and v["mAP"] > best:
            best = v["mAP"]
            flag = "  <- best mAP"
        print(f"epoch {epoch}/{cfg.epochs}  train loss {train_loss:.4f}  "
              f"val mAP {v['mAP']:.4f}  P {v['precision']:.3f}  R {v['recall']:.3f}  "
              f"[{train_s:.0f}s train]{flag}", flush=True)

    mins = (time.perf_counter() - t_start) / 60.0

    # ---- checkpoint selection: VALIDATION ONLY, rule fixed before the test split is touched --- #
    #   1. keep every epoch whose mAP is within MAP_SLACK of the best     (overall performance)
    #   2. among those, take the highest bicycle recall                   (rare-class recovery)
    #   3. break ties on mAP
    # Ranking on mAP alone is what selected an epoch with zero bicycle recall, which optimises the
    # metric this phase is least interested in.
    MAP_SLACK = 0.02
    best_map = max(h["val_mAP"] for h in history)
    eligible = [h for h in history if h["val_mAP"] >= best_map - MAP_SLACK]
    chosen = max(eligible, key=lambda h: ((h["val_per_class_recall"].get("BICYCLE") or 0.0),
                                          h["val_mAP"]))
    print()
    print(f"selection: best mAP {best_map:.4f}; within {MAP_SLACK} of it: "
          f"{[h['epoch'] for h in eligible]}")
    for h in history:
        b = h["val_per_class_recall"].get("BICYCLE") or 0.0
        mark = "  <- SELECTED" if h["epoch"] == chosen["epoch"] else ""
        print(f"   epoch {h['epoch']}  mAP {h['val_mAP']:.4f}  bicycle recall {b:.3f}{mark}")
    shutil.copyfile(CKPT_DIR / f"fasterrcnn_{cfg.run_name}_ep{chosen['epoch']}.pt", ckpt_path)
    print()
    print(f"trained in {mins:.1f} min; selected epoch {chosen['epoch']}, "
          f"val mAP {chosen['val_mAP']:.4f}")
    (CKPT_DIR / f"{cfg.run_name}_history.json").write_text(
        json.dumps({"config": asdict(cfg), "git_commit": git_hash(),
                    "class_names": CLASS_NAMES, "history": history,
                    "sampling": sampling, "total_minutes": mins,
                    "selected_epoch": chosen["epoch"],
                    "selection_rule": "max bicycle recall among epochs within 0.02 mAP of best"},
                   indent=2), encoding="utf-8")
    print(f"checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
