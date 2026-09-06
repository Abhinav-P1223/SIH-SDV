"""Evaluate the trained drivable-space model: metrics, latency, figures, Mode A vs Mode B.

    python scripts/eval_segmentation.py --root Datasets/idd-lite/idd20k_lite

Produces `docs/img/seg_*.png` (image, ground truth, prediction, drivable mask, corridor) and
prints every metric Phase 3B asks for. Examples are chosen by score, not by eye: the best, a
median case and the WORST are all shown, so a failure case is always in the figure set.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from PIL import Image                                                  # noqa: E402

from road_perception.dataset import (CLASS_NAMES, DRIVABLE, IDDLite, INPUT_H, INPUT_W,  # noqa: E402
                                     NUM_CLASSES, find_pairs)
from road_perception.drivable import clean_mask, drivable_mask, extract_corridor  # noqa: E402
from road_perception.evaluate import SegmentationMetrics, boundary_f1, format_summary  # noqa: E402
from road_perception.integration import DrivableSpacePerception, corridor_agreement  # noqa: E402

PALETTE = np.array([
    [107, 142, 35],    # drivable      olive
    [220, 20, 60],     # non-drivable  crimson
    [255, 200, 0],     # living things amber
    [0, 130, 200],     # vehicles      blue
    [145, 30, 180],    # road-side     purple
    [128, 128, 128],   # far objects   grey
    [135, 206, 235],   # sky           light blue
], dtype=np.uint8)


def colourise(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask < NUM_CLASSES
    out[valid] = PALETTE[mask[valid]]
    return out


def evaluate_split(per: DrivableSpacePerception, root: Path, split: str) -> dict:
    ds = IDDLite(root, split, augment=False)
    metrics = SegmentationMetrics()
    per_image, latency = [], []
    for i in range(len(ds)):
        p = ds._item(i)
        img = np.asarray(Image.open(p.image).convert("RGB"))
        pred, ms = per.segment(img)
        latency.append(ms)
        gt = np.asarray(Image.open(p.mask).resize((INPUT_W, INPUT_H), Image.NEAREST))
        metrics.update(pred, gt)
        per_image.append((boundary_f1(pred, gt)["f1"], i))
    s = metrics.summary()
    s["latency_ms_mean"] = float(statistics.mean(latency))
    s["latency_ms_p95"] = float(np.percentile(latency, 95))
    s["fps"] = 1000.0 / s["latency_ms_mean"]
    s["_per_image"] = sorted(per_image)
    return s


def figure(per: DrivableSpacePerception, ds: IDDLite, idx: int, score: float,
           label: str, out: Path) -> None:
    p = ds._item(idx)
    img = np.asarray(Image.open(p.image).convert("RGB").resize((INPUT_W, INPUT_H)))
    gt = np.asarray(Image.open(p.mask).resize((INPUT_W, INPUT_H), Image.NEAREST))
    r = per.perceive(np.asarray(Image.open(p.image).convert("RGB")))

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.4))
    ax[0].imshow(img); ax[0].set_title(f"{label}: image\n{p.image.name}", fontsize=9)
    ax[1].imshow(colourise(gt)); ax[1].set_title("ground truth (level-1)", fontsize=9)
    ax[2].imshow(colourise(r.prediction)); ax[2].set_title(
        f"prediction   boundary F1 {score:.3f}", fontsize=9)
    ov = img.copy()
    ov[r.drivable] = (0.45 * ov[r.drivable] + 0.55 * np.array([80, 220, 120])).astype(np.uint8)
    ax[3].imshow(ov)
    ax[3].set_title(f"drivable mask -> corridor\n{r.corridor.valid_rows} rows, "
                    f"{r.inference_ms:.0f} ms + {r.postprocess_ms:.0f} ms", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("Datasets/idd-lite/idd20k_lite"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("road_perception/checkpoints/fastscnn_iddlite.pt"))
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    per = DrivableSpacePerception(a.checkpoint)
    print("=" * 78 + "\nIDD-Lite drivable-space segmentation: held-out validation\n" + "=" * 78)
    s = evaluate_split(per, a.root, "val")
    print("\n" + format_summary(s))
    print(f"\ninference           {s['latency_ms_mean']:.1f} ms mean, "
          f"{s['latency_ms_p95']:.1f} ms p95  ({s['fps']:.1f} FPS, CPU)")

    print("\nconfusion matrix (rows = truth, cols = prediction):")
    cm = np.asarray(s["confusion_matrix"], dtype=np.float64)
    frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print("           " + "".join(f"{n[:8]:>9s}" for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n[:10]:>10s} " + "".join(f"{frac[i, j]:9.3f}" for j in range(NUM_CLASSES)))

    # ---- figures: best, median and worst, chosen by score rather than by eye ---------------- #
    ds = IDDLite(a.root, "val", augment=False)
    ranked = s.pop("_per_image")
    picks = [("worst", ranked[0]), ("median", ranked[len(ranked) // 2]), ("best", ranked[-1])]
    for label, (score, idx) in picks:
        out = a.out / f"seg_{label}.png"
        figure(per, ds, idx, score, label, out)
        print(f"  wrote {out}  (boundary F1 {score:.3f})")

    # ---- Mode A vs Mode B ------------------------------------------------------------------- #
    print("\n" + "=" * 78 + "\nMode A (ground-truth corridor) vs Mode B (predicted corridor)\n"
          + "=" * 78)
    rows = []
    for i in range(len(ds)):
        p = ds._item(i)
        img = np.asarray(Image.open(p.image).convert("RGB"))
        gt = np.asarray(Image.open(p.mask).resize((INPUT_W, INPUT_H), Image.NEAREST))
        t0 = time.perf_counter()
        r = per.perceive(img)
        extra_ms = (time.perf_counter() - t0) * 1e3
        truth_corridor = extract_corridor(clean_mask(gt == DRIVABLE), per.projection)
        ag = corridor_agreement(r.corridor, truth_corridor)
        rows.append({
            "mode_a_rows": truth_corridor.valid_rows, "mode_b_rows": r.corridor.valid_rows,
            "mode_a_road": truth_corridor.valid_rows >= 3,
            "mode_b_road": r.road is not None, "extra_ms": extra_ms, **ag})

    ok = [r for r in rows if r["left_rmse_m"] is not None]
    print(f"\nvalidation images                   {len(rows)}")
    print(f"Mode A yielded a usable corridor    {sum(r['mode_a_road'] for r in rows)}")
    print(f"Mode B yielded a usable corridor    {sum(r['mode_b_road'] for r in rows)}")
    print(f"comparable pairs                    {len(ok)}")
    if ok:
        print(f"left boundary RMSE                  "
              f"{statistics.median(r['left_rmse_m'] for r in ok):.2f} m median")
        print(f"right boundary RMSE                 "
              f"{statistics.median(r['right_rmse_m'] for r in ok):.2f} m median")
        print(f"corridor width bias                 "
              f"{statistics.median(r['width_bias_m'] for r in ok):+.2f} m median")
        print(f"overlapping range                   "
              f"{statistics.median(r['overlap_m'] for r in ok):.1f} m median")
    print(f"added perception latency            "
          f"{statistics.mean(r['extra_ms'] for r in rows):.1f} ms mean per frame")

    (a.out.parent / "phase3b_results.json").write_text(
        json.dumps({"validation": s, "mode_comparison": {
            "images": len(rows), "comparable": len(ok),
            "mode_a_usable": sum(r["mode_a_road"] for r in rows),
            "mode_b_usable": sum(r["mode_b_road"] for r in rows),
            "left_rmse_median_m": statistics.median(r["left_rmse_m"] for r in ok) if ok else None,
            "right_rmse_median_m": statistics.median(r["right_rmse_m"] for r in ok) if ok else None,
            "width_bias_median_m": statistics.median(r["width_bias_m"] for r in ok) if ok else None,
            "extra_latency_ms": statistics.mean(r["extra_ms"] for r in rows)}},
            indent=2), encoding="utf-8")
    print("\nwrote docs/phase3b_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
