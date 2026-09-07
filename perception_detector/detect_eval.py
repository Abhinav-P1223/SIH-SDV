"""Detection scoring, written once so the A/B comparison cannot drift between its two arms.

Both the COCO baseline and the fine-tuned model are scored by `evaluate` below, with the same IoU
threshold, the same score threshold, the same greedy class-aware matching and the same average
precision definition. Nothing is passed in that could differ between the arms except the function
that turns an image into predictions.

The matching is the same one Phase 6 used: highest score first, one prediction per ground-truth
box, class-aware so a car box on a motorcycle is a false positive rather than a hit.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict

import numpy as np


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)))
    x0 = np.maximum(pred[:, None, 0], gt[None, :, 0])
    y0 = np.maximum(pred[:, None, 1], gt[None, :, 1])
    x1 = np.minimum(pred[:, None, 2], gt[None, :, 2])
    y1 = np.minimum(pred[:, None, 3], gt[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = ap[:, None] + ag[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def match(pboxes, plabels, pscores, gboxes, glabels, thr: float):
    """Greedy, highest score first, one prediction per ground-truth box, class-aware."""
    order = np.argsort(-pscores) if len(pscores) else np.zeros(0, dtype=int)
    M = iou_matrix(pboxes, gboxes)
    used, pairs = set(), []
    for i in order:
        best, best_j = thr, None
        for j in range(len(gboxes)):
            if j in used or glabels[j] != plabels[i]:
                continue
            if M[i, j] >= best:
                best, best_j = M[i, j], j
        if best_j is not None:
            used.add(best_j)
            pairs.append((int(i), int(best_j)))
    return pairs, used


def average_precision(records: list[tuple[float, bool]], n_gt: int) -> float:
    """Area under the precision-recall curve, all-point interpolation, 101-point sampling."""
    if n_gt == 0 or not records:
        return float("nan")
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([1.0 if hit else 0.0 for _, hit in records])
    fp = np.cumsum([0.0 if hit else 1.0 for _, hit in records])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.clip(np.searchsorted(rec, grid, side="left"), 0, len(prec) - 1)
    return float(np.mean(np.where(grid <= rec[-1], prec[idx], 0.0)))


def evaluate(predict, dataset, class_names, iou_thr: float = 0.5,
             score_thr: float = 0.35) -> dict:
    """`predict(image_path) -> (boxes Nx4, labels N, scores N)` in OUR class ids.

    Returns overall and per-class precision, recall and average precision, plus the counts the
    Phase 7C brief asks for: ground-truth instances, predictions, false positives and misses.
    """
    per_cls_records: dict[int, list] = defaultdict(list)
    n_gt: dict[int, int] = defaultdict(int)
    n_pred: dict[int, int] = defaultdict(int)
    tp_cls: dict[int, int] = defaultdict(int)
    TP = FP = FN = 0
    lat = []

    for i in range(len(dataset)):
        gboxes, glabels = dataset.ground_truth(i)
        t0 = time.perf_counter()
        pboxes, plabels, pscores = predict(dataset.path_of(i))
        lat.append((time.perf_counter() - t0) * 1e3)
        keep = pscores >= score_thr if len(pscores) else np.zeros(0, dtype=bool)
        pboxes, plabels, pscores = pboxes[keep], plabels[keep], pscores[keep]

        pairs, used = match(pboxes, plabels, pscores, gboxes, glabels, iou_thr)
        matched = {a for a, _ in pairs}
        TP += len(pairs)
        FP += len(pboxes) - len(pairs)
        FN += len(gboxes) - len(used)
        for k, lab in enumerate(plabels):
            n_pred[int(lab)] += 1
            per_cls_records[int(lab)].append((float(pscores[k]), k in matched))
        for lab in glabels:
            n_gt[int(lab)] += 1
        for a, b in pairs:
            tp_cls[int(glabels[b])] += 1

    per_class = {}
    for cid in sorted(set(n_gt) | set(n_pred)):
        name = class_names[cid]
        g, p, t = n_gt[cid], n_pred[cid], tp_cls[cid]
        per_class[name] = {
            "gt": g, "predictions": p, "tp": t, "fp": p - t, "fn": g - t,
            "precision": t / p if p else float("nan"),
            "recall": t / g if g else float("nan"),
            "ap": average_precision(per_cls_records.get(cid, []), g),
        }
    aps = [v["ap"] for v in per_class.values() if v["ap"] == v["ap"]]
    return {
        "images": len(dataset), "tp": TP, "fp": FP, "fn": FN,
        "precision": TP / (TP + FP) if TP + FP else float("nan"),
        "recall": TP / (TP + FN) if TP + FN else float("nan"),
        "mAP": float(np.mean(aps)) if aps else float("nan"),
        "per_class": per_class,
        "latency_ms_mean": float(statistics.mean(lat)) if lat else float("nan"),
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else float("nan"),
        "iou_threshold": iou_thr, "score_threshold": score_thr,
    }


def format_table(name_a: str, a: dict, name_b: str, b: dict, classes) -> str:
    """The Step 8 comparison, with deltas, so a regression cannot hide behind an improvement."""
    L = []
    w = f"{'':<26}{name_a:>12}{name_b:>14}{'delta':>10}"
    L.append(w); L.append("-" * len(w))

    def row(label, va, vb, fmt="{:>12.3f}", dfmt="{:>+10.3f}"):
        if va != va or vb != vb:
            L.append(f"{label:<26}{'n/a':>12}{'n/a':>14}{'':>10}"); return
        L.append(f"{label:<26}" + fmt.format(va) + fmt.replace('12', '14').format(vb)
                 + dfmt.format(vb - va))

    row("overall precision", a["precision"], b["precision"])
    row("overall recall", a["recall"], b["recall"])
    row("mAP", a["mAP"], b["mAP"])
    L.append("")
    for c in classes:
        pa, pb = a["per_class"].get(c), b["per_class"].get(c)
        if not pa and not pb:
            continue
        ra = pa["recall"] if pa else float("nan")
        rb = pb["recall"] if pb else float("nan")
        row(f"{c.lower()} recall", ra, rb)
    L.append("")
    row("latency ms", a["latency_ms_mean"], b["latency_ms_mean"], "{:>12.1f}", "{:>+10.1f}")
    return "\n".join(L)
