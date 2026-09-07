"""Phase 6: measure the camera detector, then push its output through the EXISTING tracker.

    python scripts/eval_detector.py --nuscenes Datasets/v1.0-mini --idd Datasets/idd-lite/idd20k_lite

THREE THINGS ARE MEASURED, AND THEY ANSWER DIFFERENT QUESTIONS
    1. DETECTION QUALITY, on nuScenes-mini camera keyframes against 3D boxes projected into 2D.
       The detector was trained on COCO and has never seen this data, so it is a legitimate
       held-out set. Precision, recall and average precision per class, stratified by range,
       because a driving detector that only works at 10 m is a different thing from one that
       works at 50 m and the aggregate hides that completely.

    2. INTEGRATION, on the same frames: detector boxes become world-frame `Detection`s and go into
       the shipped `SensorFusionTracker` with nothing modified. What comes out is tracks.

    3. INDIAN-ROAD BEHAVIOUR, on IDD-Lite photographs. There is no detection ground truth there,
       so this is qualitative by construction and is reported as counts and figures, never as
       accuracy. Saying otherwise would be inventing a benchmark.

The ground-truth boxes are used for scoring only. They are never presented as detector output.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image                                                    # noqa: E402

from autonomy.core.config import ObjectProfiles, PerceptionConfig        # noqa: E402
from autonomy.core.types import ObjectType, VehicleState                 # noqa: E402
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig  # noqa: E402
from dataset_adapters.nuscenes_2d import NuScenesCamera2D                # noqa: E402
from perception_detector import (CameraGeometry, CameraObjectDetector, DetectorConfig,  # noqa: E402
                                 boxes_to_detections)
from perception_detector.detector import UNREACHABLE_TYPES               # noqa: E402

RANGE_BANDS = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0), (50.0, 1e9))


def iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr: float = 0.5):
    """Greedy highest-score-first matching, one prediction per ground-truth box.

    Class-aware: a car box on a pedestrian is not a hit. Returns (matched pairs, unmatched preds,
    unmatched gts) so precision and recall come from the same pass.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    used, pairs = set(), []
    for i in order:
        p = preds[i]
        best, best_j = thr, None
        for j, g in enumerate(gts):
            if j in used or g.object_type is not p.object_type:
                continue
            v = iou((p.x0, p.y0, p.x1, p.y1), (g.x0, g.y0, g.x1, g.y1))
            if v >= best:
                best, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j))
    return pairs, [i for i in range(len(preds)) if i not in {a for a, _ in pairs}], \
        [j for j in range(len(gts)) if j not in used]


def average_precision(records: list[tuple[float, bool]], n_gt: int) -> float:
    """AP by the area under the precision-recall curve, all-point interpolation."""
    if n_gt == 0 or not records:
        return float("nan")
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([1 if hit else 0 for _, hit in records], dtype=float)
    fp = np.cumsum([0 if hit else 1 for _, hit in records], dtype=float)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    idx = np.clip(idx, 0, len(prec) - 1)
    return float(np.mean(np.where(np.linspace(0, 1, 101) <= rec[-1], prec[idx], 0.0)))


def evaluate_nuscenes(det, ds, channels, limit, iou_thr) -> dict:
    frames = ds.camera_keyframes(channels)
    if limit:
        frames = frames[:limit]
    per_cls_records: dict[ObjectType, list] = defaultdict(list)
    n_gt: dict[ObjectType, int] = defaultdict(int)
    band_tp: dict[tuple, int] = defaultdict(int)
    band_gt: dict[tuple, int] = defaultdict(int)
    tp = fp = fn = 0
    lat, n_img = [], 0

    for sd in frames:
        fr = ds.frame(sd)
        if not fr.path.exists():
            continue
        img = np.asarray(Image.open(fr.path).convert("RGB"))
        preds = det.detect(img)
        lat.append(det.last_inference_ms)
        n_img += 1
        pairs, un_p, un_g = match(preds, fr.boxes, iou_thr)
        tp += len(pairs); fp += len(un_p); fn += len(un_g)
        matched_p = {a for a, _ in pairs}
        for i, p in enumerate(preds):
            per_cls_records[p.object_type].append((p.score, i in matched_p))
        for g in fr.boxes:
            n_gt[g.object_type] += 1
            for band in RANGE_BANDS:
                if band[0] <= g.distance_m < band[1]:
                    band_gt[band] += 1
        for _, j in pairs:
            g = fr.boxes[j]
            for band in RANGE_BANDS:
                if band[0] <= g.distance_m < band[1]:
                    band_tp[band] += 1

    per_class = {}
    for cls, n in sorted(n_gt.items(), key=lambda kv: -kv[1]):
        recs = per_cls_records.get(cls, [])
        hits = sum(1 for _, h in recs if h)
        per_class[cls.value] = {
            "gt": n, "predictions": len(recs), "tp": hits,
            "precision": hits / len(recs) if recs else float("nan"),
            "recall": hits / n if n else float("nan"),
            "ap": average_precision(list(recs), n),
        }
    return {
        "images": n_img, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "mAP": float(np.nanmean([v["ap"] for v in per_class.values()])) if per_class else float("nan"),
        "per_class": per_class,
        "range_recall": {f"{int(a)}-{'inf' if b > 1e8 else int(b)} m":
                         (band_tp[(a, b)] / band_gt[(a, b)] if band_gt[(a, b)] else float("nan"))
                         for a, b in RANGE_BANDS},
        "range_gt": {f"{int(a)}-{'inf' if b > 1e8 else int(b)} m": band_gt[(a, b)]
                     for a, b in RANGE_BANDS},
        "latency_ms_mean": float(statistics.mean(lat)) if lat else float("nan"),
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else float("nan"),
    }


def integrate_with_tracker(det, ds, channels, limit) -> dict:
    """Detector -> Detection -> the SHIPPED tracker. Nothing in autonomy/ is modified."""
    frames = [ds.frame(sd) for sd in ds.camera_keyframes(channels)[:limit]]
    frames = [f for f in frames if f.path.exists()]
    perception = PerceptionConfig.load()
    tracker = SensorFusionTracker(TrackerConfig(**perception.tracker), ObjectProfiles.load())
    ego = VehicleState(timestamp=0.0, x=0.0, y=0.0, yaw=0.0, longitudinal_velocity=0.0)
    counts, det_counts, adapt_ms = [], [], []
    t0_wall = time.perf_counter()
    for k, fr in enumerate(frames):
        img = np.asarray(Image.open(fr.path).convert("RGB"))
        raw = det.detect(img)
        geom = CameraGeometry.from_intrinsic(fr.intrinsic)
        t0 = time.perf_counter()
        dets = boxes_to_detections(raw, geom, 0.0, 0.0, 0.0, timestamp=k * 0.5)
        adapt_ms.append((time.perf_counter() - t0) * 1e3)
        tracker.ingest(dets, k * 0.5, ego)
        counts.append(len(tracker.get_object_states(k * 0.5)))
        det_counts.append(len(dets))
    return {
        "frames": len(frames),
        "detections_total": int(sum(det_counts)),
        "detections_per_frame": float(np.mean(det_counts)) if det_counts else 0.0,
        "confirmed_tracks_final": counts[-1] if counts else 0,
        "confirmed_tracks_mean": float(np.mean(counts)) if counts else 0.0,
        "box_to_detection_ms_mean": float(np.mean(adapt_ms)) if adapt_ms else 0.0,
        "wall_s": time.perf_counter() - t0_wall,
    }


def indian_roads(det, idd_root: Path, limit: int) -> dict:
    """Qualitative only: IDD-Lite has no detection ground truth, so nothing here is an accuracy."""
    imgs = sorted((idd_root / "leftImg8bit" / "val").rglob("*_image.jpg"))[:limit]
    per_cls: dict[str, int] = defaultdict(int)
    counts, lat = [], []
    for p in imgs:
        arr = np.asarray(Image.open(p).convert("RGB"))
        dets = det.detect(arr)
        lat.append(det.last_inference_ms)
        counts.append(len(dets))
        for d in dets:
            per_cls[d.object_type.value] += 1
    return {
        "images": len(imgs),
        "detections_per_image": float(np.mean(counts)) if counts else 0.0,
        "images_with_no_detection": int(sum(1 for c in counts if c == 0)),
        "by_class": dict(sorted(per_cls.items(), key=lambda kv: -kv[1])),
        "latency_ms_mean": float(statistics.mean(lat)) if lat else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes", type=Path, default=Path("Datasets/v1.0-mini"))
    ap.add_argument("--idd", type=Path, default=Path("Datasets/idd-lite/idd20k_lite"))
    ap.add_argument("--channels", nargs="*", default=["CAM_FRONT"])
    ap.add_argument("--limit", type=int, default=0, help="0 = every keyframe of the chosen cameras")
    ap.add_argument("--idd-limit", type=int, default=60)
    ap.add_argument("--track-limit", type=int, default=40)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.35)
    ap.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_320_fpn")
    ap.add_argument("--out", type=Path, default=Path("docs/phase6_results.json"))
    a = ap.parse_args()

    det = CameraObjectDetector(DetectorConfig(score_threshold=a.score, model_name=a.model))
    print("=" * 78)
    print(f"Phase 6: {a.model}, COCO weights (mAP {det.coco_map}), never trained by us")
    print("=" * 78, flush=True)
    print("cannot detect at all: " + ", ".join(t.value for t in UNREACHABLE_TYPES))

    ds = NuScenesCamera2D(a.nuscenes)
    channels = tuple(a.channels) if a.channels else None
    print(f"\nnuScenes-mini, cameras {channels}, IoU {a.iou}, score {a.score} ...", flush=True)
    ns = evaluate_nuscenes(det, ds, channels, a.limit or None, a.iou)
    print(f"\nimages {ns['images']}   TP {ns['tp']}  FP {ns['fp']}  FN {ns['fn']}")
    print(f"precision {ns['precision']:.3f}   recall {ns['recall']:.3f}   mAP@{a.iou} {ns['mAP']:.3f}")
    print(f"inference {ns['latency_ms_mean']:.0f} ms mean, {ns['latency_ms_p95']:.0f} ms p95")
    print(f"\n{'class':<14}{'gt':>6}{'pred':>6}{'TP':>6}{'prec':>7}{'recall':>8}{'AP':>7}")
    for k, v in ns["per_class"].items():
        print(f"{k:<14}{v['gt']:>6}{v['predictions']:>6}{v['tp']:>6}"
              f"{v['precision']:>7.2f}{v['recall']:>8.2f}{v['ap']:>7.2f}")
    print("\nrecall by ground-truth range:")
    for k, v in ns["range_recall"].items():
        print(f"   {k:<10} {v:.3f}   ({ns['range_gt'][k]} boxes)")

    print("\n" + "-" * 78)
    print("Detector -> Detection -> the SHIPPED tracker, nothing in autonomy/ modified")
    print("-" * 78, flush=True)
    tr = integrate_with_tracker(det, ds, channels, a.track_limit)
    for k, v in tr.items():
        print(f"   {k:<28}{v:.2f}" if isinstance(v, float) else f"   {k:<28}{v}")

    idd = {}
    if a.idd.exists():
        print("\n" + "-" * 78)
        print("Real Indian-road images (IDD-Lite). QUALITATIVE: there is no ground truth here.")
        print("-" * 78, flush=True)
        idd = indian_roads(det, a.idd, a.idd_limit)
        for k, v in idd.items():
            print(f"   {k:<28}{v}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "coco_map": det.coco_map,
                                 "score_threshold": a.score, "iou": a.iou,
                                 "nuscenes": ns, "tracker_integration": tr,
                                 "idd_lite_qualitative": idd,
                                 "undetectable_types": [t.value for t in UNREACHABLE_TYPES]},
                                indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
