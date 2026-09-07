"""Phase 7D: why bicycle, bus and truck got worse. Diagnosis only, no training.

    python scripts/diagnose_uvh26_regression.py

Raw predictions are computed ONCE per model at a very low score threshold and cached in memory,
then every threshold and every failure question is answered from that cache. Inference is never
repeated, so nothing here can drift between questions.

WHAT A MISSED OBJECT ACTUALLY MEANS
    Recall alone cannot distinguish "the detector never saw it" from "the detector saw it and
    called it something else" from "the detector saw it but scored it 0.31 against a 0.35 cut".
    Those have completely different fixes, so every missed ground-truth box is classified by
    looking at what the model actually produced near it:

      WRONG_CLASS        a confident box of another class sits on it
      LOW_CONFIDENCE     a right-class box sits on it, below the score threshold
      POOR_LOCALISATION  a right-class box overlaps it, but under the IoU threshold
      ABSENT             nothing of any class overlaps it at all
    and each is also tagged small / medium / large by box area, because "too small to see" is a
    different problem from "wrong class".

THE TEST SET IS NOT BEING TUNED ON
    The threshold sweep that could inform a decision is run on VALIDATION. The same sweep is shown
    on the test split too, labelled diagnostic, purely so the report can say whether the regression
    is calibration or capability. No threshold is chosen from the test numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image                                                    # noqa: E402

from perception_detector.detect_eval import evaluate, iou_matrix, match  # noqa: E402
from perception_detector.detector import COCO_TO_OBJECT_TYPE             # noqa: E402
from perception_detector.uvh26 import (CLASS_NAMES, NUM_CLASSES, UVH26Subset,  # noqa: E402
                                       load_manifest)

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "perception_detector" / "checkpoints" / "fasterrcnn_uvh26.pt"
OUT = ROOT / "docs" / "phase7d_diagnosis.json"
CLASSES = ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
COCO_TO_OURS = {c: NAME_TO_ID[t.value] for c, t in COCO_TO_OBJECT_TYPE.items()
                if t.value in NAME_TO_ID}
FOCUS = ["BICYCLE", "BUS", "TRUCK"]
LOW = 0.01          # cache everything above this; every threshold is applied afterwards


def load_image(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1))


def build_models():
    from torchvision.models import detection as tvd
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    w = tvd.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    coco = tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).eval()

    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    ft = tvd.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    in_f = ft.roi_heads.box_predictor.cls_score.in_features
    ft.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    ft.load_state_dict(state["model"])
    ft.eval()
    return coco, ft, state


def cache_predictions(model, dataset, translate: dict | None) -> list[dict]:
    """Every prediction above LOW, once. `translate` maps COCO ids to ours for the baseline."""
    out = []
    for i in range(len(dataset)):
        with torch.no_grad():
            o = model([load_image(dataset.path_of(i))])[0]
        b, l, s = (o["boxes"].numpy(), o["labels"].numpy(), o["scores"].numpy())
        if translate is not None:
            keep = np.array([k for k, lab in enumerate(l) if int(lab) in translate], dtype=int)
            if len(keep):
                b, l, s = b[keep], np.array([translate[int(x)] for x in l[keep]]), s[keep]
            else:
                b, l, s = np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
        keep = s >= LOW
        out.append({"boxes": b[keep], "labels": l[keep].astype(int), "scores": s[keep]})
    return out


def score_at(cache, dataset, thr: float, iou: float = 0.5) -> dict:
    """Replay the cached predictions at a threshold. Same matching as detect_eval."""
    n_gt, n_pred, tp_cls = Counter(), Counter(), Counter()
    TP = FP = FN = 0
    for i in range(len(dataset)):
        gb, gl = dataset.ground_truth(i)
        c = cache[i]
        k = c["scores"] >= thr
        pb, pl, ps = c["boxes"][k], c["labels"][k], c["scores"][k]
        pairs, used = match(pb, pl, ps, gb, gl, iou)
        TP += len(pairs); FP += len(pb) - len(pairs); FN += len(gb) - len(used)
        for lab in pl:
            n_pred[int(lab)] += 1
        for lab in gl:
            n_gt[int(lab)] += 1
        for _, j in pairs:
            tp_cls[int(gl[j])] += 1
    per = {}
    for cid in sorted(set(n_gt) | set(n_pred)):
        g, p, t = n_gt[cid], n_pred[cid], tp_cls[cid]
        per[CLASS_NAMES[cid]] = {"gt": g, "pred": p, "tp": t, "fp": p - t, "fn": g - t,
                                 "precision": t / p if p else float("nan"),
                                 "recall": t / g if g else float("nan")}
    return {"threshold": thr, "tp": TP, "fp": FP, "fn": FN,
            "precision": TP / (TP + FP) if TP + FP else float("nan"),
            "recall": TP / (TP + FN) if TP + FN else float("nan"), "per_class": per}


def classify_misses(cache, dataset, thr: float, iou: float = 0.5) -> dict:
    """Every missed ground-truth box of the focus classes, labelled by WHY it was missed."""
    reasons = defaultdict(Counter)
    sizes = defaultdict(Counter)
    wrong_as = defaultdict(Counter)
    for i in range(len(dataset)):
        gb, gl = dataset.ground_truth(i)
        c = cache[i]
        k = c["scores"] >= thr
        pb, pl, ps = c["boxes"][k], c["labels"][k], c["scores"][k]
        pairs, used = match(pb, pl, ps, gb, gl, iou)
        for j in range(len(gb)):
            name = CLASS_NAMES[int(gl[j])]
            if j in used or name not in FOCUS:
                continue
            area = (gb[j, 2] - gb[j, 0]) * (gb[j, 3] - gb[j, 1])
            size = "small" if area < 32 * 32 else ("medium" if area < 96 * 96 else "large")
            sizes[name][size] += 1

            g1 = gb[j:j + 1]
            # a confident box of ANOTHER class sitting on it
            if len(pb):
                M = iou_matrix(pb, g1)[:, 0]
                over = np.where(M >= iou)[0]
                other = [o for o in over if int(pl[o]) != int(gl[j])]
                if other:
                    best = max(other, key=lambda o: ps[o])
                    reasons[name]["WRONG_CLASS"] += 1
                    wrong_as[name][CLASS_NAMES[int(pl[best])]] += 1
                    continue
            # a right-class box below the threshold
            allb, alll, alls = c["boxes"], c["labels"], c["scores"]
            same = np.where(alll == int(gl[j]))[0]
            if len(same):
                M2 = iou_matrix(allb[same], g1)[:, 0]
                hit = np.where(M2 >= iou)[0]
                if len(hit):
                    reasons[name]["LOW_CONFIDENCE"] += 1
                    continue
                if (M2 >= 0.1).any():
                    reasons[name]["POOR_LOCALISATION"] += 1
                    continue
            if len(allb) and (iou_matrix(allb, g1)[:, 0] >= 0.1).any():
                reasons[name]["OVERLAPS_SOMETHING_ELSE"] += 1
                continue
            reasons[name]["ABSENT"] += 1
    return {"reasons": {k: dict(v) for k, v in reasons.items()},
            "sizes": {k: dict(v) for k, v in sizes.items()},
            "wrong_class_as": {k: dict(v) for k, v in wrong_as.items()}}


def confidence_profile(cache, dataset) -> dict:
    out = {}
    for cid, name in enumerate(CLASS_NAMES):
        if cid == 0:
            continue
        s = np.concatenate([c["scores"][c["labels"] == cid] for c in cache]) \
            if any((c["labels"] == cid).any() for c in cache) else np.zeros(0)
        out[name] = {
            "predictions_above_0.01": int(len(s)),
            "above_0.35": int((s >= 0.35).sum()),
            "max": float(s.max()) if len(s) else 0.0,
            "p90": float(np.percentile(s, 90)) if len(s) else 0.0,
            "median": float(np.median(s)) if len(s) else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    man = load_manifest()
    test = UVH26Subset("test", man)
    val = UVH26Subset("val", man)
    coco, ft, state = build_models()

    print("=" * 78)
    print("PHASE 7D DIAGNOSIS - no training, no threshold chosen from the test split")
    print("=" * 78, flush=True)
    print("caching predictions once per model at score >= 0.01 ...", flush=True)
    c_coco_test = cache_predictions(coco, test, COCO_TO_OURS)
    c_ft_test = cache_predictions(ft, test, None)
    c_ft_val = cache_predictions(ft, val, None)

    # ---- STEP 1 -------------------------------------------------------------------------- #
    A = score_at(c_coco_test, test, a.score, a.iou)
    B = score_at(c_ft_test, test, a.score, a.iou)
    print("\n" + "=" * 78)
    print(f"STEP 1  verified on the locked test split, threshold {a.score}, IoU {a.iou}")
    print("=" * 78)
    print(f"{'class':<15}{'GT':>5}{'A pred':>8}{'A TP':>6}{'A FP':>6}{'A FN':>6}"
          f"{'B pred':>8}{'B TP':>6}{'B FP':>6}{'B FN':>6}{'A rec':>8}{'B rec':>8}")
    for c in CLASSES:
        x, y = A["per_class"].get(c), B["per_class"].get(c)
        f = lambda v, k: (v[k] if v else 0)
        ra = f(x, "recall") if x else 0.0
        rb = f(y, "recall") if y else 0.0
        ra = 0.0 if ra != ra else ra
        rb = 0.0 if rb != rb else rb
        print(f"{c:<15}{f(x,'gt') or f(y,'gt'):>5}{f(x,'pred'):>8}{f(x,'tp'):>6}{f(x,'fp'):>6}"
              f"{f(x,'fn'):>6}{f(y,'pred'):>8}{f(y,'tp'):>6}{f(y,'fp'):>6}{f(y,'fn'):>6}"
              f"{ra:>8.3f}{rb:>8.3f}")

    # ---- STEP 2 -------------------------------------------------------------------------- #
    mis = classify_misses(c_ft_test, test, a.score, a.iou)
    print("\n" + "=" * 78)
    print("STEP 2  why each missed object was missed, fine-tuned model")
    print("=" * 78)
    for c in FOCUS:
        r = mis["reasons"].get(c, {})
        tot = sum(r.values())
        print(f"\n{c}  ({tot} missed)")
        for k, v in sorted(r.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<26}{v:>5}  {100*v/max(tot,1):5.1f}%")
        print(f"    size of missed boxes: {mis['sizes'].get(c, {})}")
        if mis["wrong_class_as"].get(c):
            print(f"    when wrong, called: {mis['wrong_class_as'][c]}")

    # ---- STEP 3 -------------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 3  confidence profile, fine-tuned model on the test split")
    print("=" * 78)
    prof = confidence_profile(c_ft_test, test)
    print(f"{'class':<15}{'preds>=.01':>12}{'>=0.35':>9}{'median':>9}{'p90':>8}{'max':>8}")
    for c in CLASSES:
        p = prof[c]
        print(f"{c:<15}{p['predictions_above_0.01']:>12}{p['above_0.35']:>9}"
              f"{p['median']:>9.3f}{p['p90']:>8.3f}{p['max']:>8.3f}")

    # ---- STEP 4 -------------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STEP 4  threshold sweep on VALIDATION (decisions may use this)")
    print("=" * 78)
    sweeps_val = [score_at(c_ft_val, val, t, a.iou) for t in (0.10, 0.20, 0.30, 0.40, 0.50)]
    hdr = f"{'thr':>6}{'prec':>8}{'recall':>8}" + "".join(f"{c[:9]:>10}" for c in FOCUS)
    print(hdr)
    for s in sweeps_val:
        row = f"{s['threshold']:>6.2f}{s['precision']:>8.3f}{s['recall']:>8.3f}"
        for c in FOCUS:
            v = s["per_class"].get(c)
            r = v["recall"] if v and v["recall"] == v["recall"] else 0.0
            row += f"{r:>10.3f}"
        print(row)

    print("\n(diagnostic only, NOT used to choose anything) same sweep on the test split:")
    sweeps_test = [score_at(c_ft_test, test, t, a.iou) for t in (0.10, 0.20, 0.30, 0.40, 0.50)]
    print(hdr)
    for s in sweeps_test:
        row = f"{s['threshold']:>6.2f}{s['precision']:>8.3f}{s['recall']:>8.3f}"
        for c in FOCUS:
            v = s["per_class"].get(c)
            r = v["recall"] if v and v["recall"] == v["recall"] else 0.0
            row += f"{r:>10.3f}"
        print(row)

    OUT.write_text(json.dumps({
        "checkpoint_epoch": state.get("epoch"), "score_threshold": a.score, "iou": a.iou,
        "step1_test": {"A_coco": A, "B_finetuned": B},
        "step2_miss_analysis": mis,
        "step3_confidence_profile": prof,
        "step4_sweep_val": sweeps_val, "step4_sweep_test_diagnostic": sweeps_test,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
