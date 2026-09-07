"""Phase 7C steps 6 to 8: the COCO baseline against the UVH-26 fine-tuned model, same test set.

    python scripts/eval_uvh26_ab.py

BOTH ARMS ARE SCORED BY THE SAME FUNCTION
    `detect_eval.evaluate` is called twice with the same dataset, the same IoU threshold, the same
    score threshold and the same matching. The only thing that differs is the callable that turns
    an image into predictions. Nothing else can vary between A and B, which is the point.

THE ONE ASYMMETRY, STATED RATHER THAN HIDDEN
    Arm A is a COCO detector, so its predictions arrive as COCO class ids and are translated into
    our six classes by the Phase 6 mapping. COCO has no auto-rickshaw, so arm A can NEVER predict
    that class and its recall there is zero by construction, not by failure. Arm B predicts our
    classes directly. This is not a scoring trick; it is the actual gap the phase exists to close,
    and the report says so beside the number.

    COCO also has no "three-wheeler" that could be mistaken for an auto-rickshaw, so arm A's
    auto-rickshaw boxes do not appear as some other class either: those objects are simply missed,
    which is what the false-negative count shows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image                                                    # noqa: E402

from perception_detector.detect_eval import evaluate, format_table       # noqa: E402
from perception_detector.detector import COCO_TO_OBJECT_TYPE             # noqa: E402
from perception_detector.uvh26 import (CLASS_NAMES, NUM_CLASSES, UVH26Subset,  # noqa: E402
                                       load_manifest)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "perception_detector" / "checkpoints" / "fasterrcnn_uvh26.pt"
OUT = ROOT / "docs" / "phase7_results.json"
CLASSES = ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]

# COCO id -> our class id, via the Phase 6 ObjectType map. Only classes that exist in BOTH.
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
COCO_TO_OURS = {c: NAME_TO_ID[t.value] for c, t in COCO_TO_OBJECT_TYPE.items()
                if t.value in NAME_TO_ID}


def load_image(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def coco_baseline_predictor():
    """Arm A: the unmodified Phase 6 detector, its COCO ids translated to our classes."""
    from torchvision.models import detection as tvd
    w = tvd.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    m = tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).eval()

    def predict(path):
        with torch.no_grad():
            o = m([load_image(path)])[0]
        b = o["boxes"].cpu().numpy(); l = o["labels"].cpu().numpy(); s = o["scores"].cpu().numpy()
        keep = [i for i, lab in enumerate(l) if int(lab) in COCO_TO_OURS]
        if not keep:
            return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
        keep = np.asarray(keep)
        return b[keep], np.asarray([COCO_TO_OURS[int(x)] for x in l[keep]]), s[keep]
    return predict, m


def finetuned_predictor(path: Path):
    """Arm B: the fine-tuned model, predicting our classes directly."""
    from torchvision.models import detection as tvd
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    state = torch.load(path, map_location="cpu", weights_only=False)
    m = tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    m.load_state_dict(state["model"])
    m.eval()

    def predict(p):
        with torch.no_grad():
            o = m([load_image(p)])[0]
        return (o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(), o["scores"].cpu().numpy())
    return predict, m, state


def show(name: str, r: dict) -> None:
    print(f"\n--- {name} ---")
    print(f"images {r['images']}   TP {r['tp']}  FP {r['fp']}  FN {r['fn']}")
    print(f"precision {r['precision']:.3f}   recall {r['recall']:.3f}   mAP {r['mAP']:.3f}   "
          f"latency {r['latency_ms_mean']:.0f} ms")
    print(f"{'class':<16}{'gt':>6}{'pred':>7}{'TP':>6}{'FP':>6}{'FN':>6}{'prec':>8}{'recall':>8}{'AP':>7}")
    for c in CLASSES:
        v = r["per_class"].get(c)
        if v is None:
            print(f"{c:<16}{'-':>6}{'-':>7}{'-':>6}{'-':>6}{'-':>6}{'-':>8}{'-':>8}{'-':>7}")
            continue
        pr = "n/a" if v["precision"] != v["precision"] else f"{v['precision']:.2f}"
        rc = "n/a" if v["recall"] != v["recall"] else f"{v['recall']:.2f}"
        apv = "n/a" if v["ap"] != v["ap"] else f"{v['ap']:.2f}"
        print(f"{c:<16}{v['gt']:>6}{v['predictions']:>7}{v['tp']:>6}{v['fp']:>6}{v['fn']:>6}"
              f"{pr:>8}{rc:>8}{apv:>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--checkpoint-c", type=Path,
                    default=ROOT / "perception_detector" / "checkpoints"
                            / "fasterrcnn_uvh26_oversampled.pt",
                    help="Phase 7E class-aware run; skipped if absent")
    a = ap.parse_args()

    test = UVH26Subset("test", load_manifest(), augment=False)
    print("=" * 78)
    print(f"Phase 7C A/B on the LOCKED test split: {len(test)} images, "
          f"{sum(len(v[0]) for v in test.boxes.values())} boxes")
    print(f"identical settings for both arms: IoU {a.iou}, score {a.score}, class-aware matching")
    print("=" * 78, flush=True)

    pa, _ = coco_baseline_predictor()
    print("\nevaluating A: COCO baseline ...", flush=True)
    ra = evaluate(pa, test, CLASS_NAMES, a.iou, a.score)
    show("A  COCO-pretrained (Phase 6 baseline)", ra)

    pb, _, state = finetuned_predictor(a.checkpoint)
    print(f"\nevaluating B: UVH-26 fine-tuned (epoch {state.get('epoch')}, "
          f"git {str(state.get('git_commit'))[:8]}) ...", flush=True)
    rb = evaluate(pb, test, CLASS_NAMES, a.iou, a.score)
    show("B  UVH-26 fine-tuned", rb)

    rc = None
    if a.checkpoint_c.exists():
        pc, _, state_c = finetuned_predictor(a.checkpoint_c)
        print(f"\nevaluating C: UVH-26 class-aware oversampling "
              f"(epoch {state_c.get('epoch')}) ...", flush=True)
        rc = evaluate(pc, test, CLASS_NAMES, a.iou, a.score)
        show("C  UVH-26 fine-tuned, class-aware oversampling", rc)

    print("\n" + "=" * 78)
    print("STEP 8 comparison")
    print("=" * 78)
    print(format_table("COCO", ra, "UVH-26 FT", rb, CLASSES))

    if rc is not None:
        print("\n" + "-" * 63)
        print("three-way: COCO / uniform sampling / class-aware oversampling")
        print("-" * 63)
        hdr = f"{'':<26}{'COCO':>10}{'FT-Uniform':>13}{'FT-Oversamp':>14}"
        print(hdr)
        print("-" * len(hdr))

        def tri(label, key, cls=None, decimals=3):
            def get(r):
                if cls is None:
                    return r[key]
                v = r["per_class"].get(cls)
                x = v[key] if v else float("nan")
                return 0.0 if x != x else x
            f = f"{{:>{{w}}.{decimals}f}}"
            print(f"{label:<26}"
                  + f.format(get(ra), w=10) + f.format(get(rb), w=13) + f.format(get(rc), w=14))

        tri("precision", "precision")
        tri("recall", "recall")
        tri("mAP", "mAP")
        for c in CLASSES:
            tri(f"{c.lower()} recall", "recall", c)
        tri("latency ms", "latency_ms_mean", None, 1)

    print("\nimprovements / regressions / unchanged (recall, threshold 0.01):")
    for c in CLASSES:
        va, vb = ra["per_class"].get(c), rb["per_class"].get(c)
        x = (va or {}).get("recall", float("nan"))
        y = (vb or {}).get("recall", float("nan"))
        x = 0.0 if x != x else x
        y = 0.0 if y != y else y
        d = y - x
        tag = "IMPROVED" if d > 0.01 else ("REGRESSED" if d < -0.01 else "unchanged")
        print(f"   {c:<16}{x:.3f} -> {y:.3f}   {tag}")

    OUT.write_text(json.dumps({
        "test_images": len(test), "iou": a.iou, "score_threshold": a.score,
        "checkpoint_epoch": state.get("epoch"), "git_commit": state.get("git_commit"),
        "A_coco_baseline": ra, "B_uvh26_finetuned": rb,
        "C_uvh26_oversampled": rc,
        "caveats": [
            "UVH-26 is elevated fixed-camera CCTV imagery, not vehicle-mounted dashcam imagery.",
            "The held-out test split is also CCTV, so CCTV-to-dashcam transfer is unvalidated.",
            "COCO has no auto-rickshaw class, so arm A's recall there is zero by construction.",
            "UVH-26 has no person and no animal class; neither Phase 6 failure is addressed.",
        ]}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
