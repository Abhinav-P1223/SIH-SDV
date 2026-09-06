"""Phase 4 step 4: replay nuScenes-mini with REAL sensor timestamps, synchronised vs temporal.

    python scripts/validate_temporal_fusion.py --root Datasets/v1.0-mini

Mode A is the pre-Phase-4 assumption: every detection in a keyframe happens at one common time.
Mode B predicts each measurement to its own timestamp. Both replays see the IDENTICAL detections
in the IDENTICAL order from the IDENTICAL scenes, so any difference is the timestamp handling and
nothing else.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT
    nuScenes annotations are the source of the controlled sensor observations here, so a small
    track-to-annotation distance is NOT evidence that a detector is accurate. What it measures is
    whether fusion put the track where the observations said the object was. The comparison is
    between two fusion strategies on the same observations, which is a fair question to ask of
    this data; "how good is our perception" is not.

    Velocity error is the metric that should move. Two sensors 66.8 ms apart looking at a moving
    object disagree by v * 0.0668 in position. Told they are simultaneous, the filter must explain
    that disagreement as noise. Told the truth, it can explain it as motion.
"""
from __future__ import annotations

import argparse
import os
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.core.config import ObjectProfiles, PerceptionConfig      # noqa: E402
from autonomy.core.types import SensorType                             # noqa: E402
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig  # noqa: E402
from dataset_adapters.nuscenes import NuScenesMini                     # noqa: E402


def motion_consistent(ds: NuScenesMini, sample: dict, dets: list) -> list:
    """Move each observation to where the object actually WAS when that sensor fired.

    THIS IS THE HONEST FIX FOR A BROKEN EXPERIMENT. `detections_for_sample` derives one detection
    per modality from the SAME annotation box, so straight out of the adapter every modality
    reports the identical position no matter what its timestamp says. Replaying that compares two
    fusion strategies on data with no inter-sensor motion in it, and unsurprisingly finds no
    difference: it is not a fair test of temporal fusion, it is a test with the effect removed.

    A real camera firing 66.8 ms after the LiDAR sees the object 66.8 ms further along. Shifting
    each observation by the ANNOTATED velocity times its own offset from the keyframe restores
    exactly that, and nothing else. The annotation is still only the source of a controlled
    observation, as this phase requires; it is not being scored against.
    """
    t_ref = ds.by_token["sample"][sample["token"]]["timestamp"] / 1e6
    vel = {}
    for ann in ds.ann_by_sample.get(sample["token"], []):
        v = ds.box_velocity(ann)
        if v is not None and np.isfinite(v).all():
            vel[ann["instance_token"]] = v
    out = []
    for d in dets:
        v = vel.get(getattr(d, "truth_id", None))
        if v is None:
            out.append(d)
            continue
        dt = d.timestamp - t_ref
        shifted = replace(d, x=d.x + v[0] * dt, y=d.y + v[1] * dt)
        shifted.truth_id = d.truth_id
        out.append(shifted)
    return out


def replay(ds: NuScenesMini, tracker_cfg: dict, max_scenes: int | None,
           motion: bool = False) -> dict:
    """One full pass over the chosen scenes with one tracker configuration."""
    profiles = ObjectProfiles.load()
    stats: Counter = Counter()
    pos_err, vel_err, cov_trace, residual, ingest_ms, spreads = [], [], [], [], [], []
    matched, expected = 0, 0
    scenes = ds.scene[:max_scenes] if max_scenes else ds.scene

    for scene in scenes:
        tracker = SensorFusionTracker(TrackerConfig(**tracker_cfg), profiles)
        for sample in ds.samples_of_scene(scene["token"]):
            dets = ds.detections_for_sample(sample["token"])
            if not dets:
                continue
            if motion:
                dets = motion_consistent(ds, sample, dets)
            stats["samples"] += 1
            stamps = [d.timestamp for d in dets]
            fusion_t = max(stamps)
            spreads.append((fusion_t - min(stamps)) * 1e3)

            ego = ds.ego_state(next(sd for sd in ds.data_by_sample[sample["token"]]
                                    if sd["is_key_frame"]))
            t0 = time.perf_counter()
            tracker.ingest(dets, fusion_t, ego)
            ingest_ms.append((time.perf_counter() - t0) * 1e3)

            tracks = tracker.get_object_states(fusion_t)
            stats["fused_objects"] += len(tracks)

            anns = ds.ann_by_sample.get(sample["token"], [])
            expected += len(anns)
            for ann in anns:
                tx, ty = ann["translation"][0], ann["translation"][1]
                if not tracks:
                    continue
                tr = min(tracks, key=lambda t: np.hypot(t.x - tx, t.y - ty))
                d = float(np.hypot(tr.x - tx, tr.y - ty))
                if d > 5.0:
                    continue                       # not the same object; not an association
                matched += 1
                pos_err.append(d)
                cov_trace.append(float(np.trace(tr.covariance)))
                v = ds.box_velocity(ann)
                if v is not None and np.isfinite(v).all():
                    vel_err.append(float(np.hypot(tr.vx - v[0], tr.vy - v[1])))
            # sensor-to-track residual: how far each raw measurement sat from its fused track
            for det in dets:
                if not tracks:
                    continue
                tr = min(tracks, key=lambda t: np.hypot(t.x - det.x, t.y - det.y))
                r = float(np.hypot(tr.x - det.x, tr.y - det.y))
                if r < 5.0:
                    residual.append(r)
        stats["scenes"] += 1

    def q(v, p):
        return float(np.percentile(v, p)) if v else float("nan")

    return {
        "scenes": stats["scenes"], "samples": stats["samples"],
        "fused_objects": stats["fused_objects"],
        "association_rate": matched / expected if expected else float("nan"),
        "matched": matched, "annotations": expected,
        "pos_err_median_m": q(pos_err, 50), "pos_err_p90_m": q(pos_err, 90),
        "vel_err_median_mps": q(vel_err, 50), "vel_err_p90_mps": q(vel_err, 90),
        "cov_trace_median": q(cov_trace, 50),
        "residual_median_m": q(residual, 50), "residual_p90_m": q(residual, 90),
        "ingest_ms_mean": float(statistics.mean(ingest_ms)) if ingest_ms else float("nan"),
        "ingest_ms_p95": q(ingest_ms, 95),
        "spread_ms_median": q(spreads, 50), "spread_ms_max": q(spreads, 100),
    }


ROWS = [
    ("keyframe sensor spread, median ms", "spread_ms_median", "{:.1f}"),
    ("keyframe sensor spread, max ms", "spread_ms_max", "{:.1f}"),
    ("annotations seen", "annotations", "{:.0f}"),
    ("associated to a track", "matched", "{:.0f}"),
    ("association rate", "association_rate", "{:.3f}"),
    ("track position error, median m", "pos_err_median_m", "{:.3f}"),
    ("track position error, p90 m", "pos_err_p90_m", "{:.3f}"),
    ("track velocity error, median m/s", "vel_err_median_mps", "{:.3f}"),
    ("track velocity error, p90 m/s", "vel_err_p90_mps", "{:.3f}"),
    ("reported covariance trace, median", "cov_trace_median", "{:.4f}"),
    ("sensor-to-track residual, median m", "residual_median_m", "{:.3f}"),
    ("sensor-to-track residual, p90 m", "residual_p90_m", "{:.3f}"),
    ("ingest latency, mean ms", "ingest_ms_mean", "{:.2f}"),
    ("ingest latency, p95 ms", "ingest_ms_p95", "{:.2f}"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("Datasets/v1.0-mini"))
    ap.add_argument("--scenes", type=int, default=0, help="0 = every scene in the mini split")
    ap.add_argument("--out", type=Path, default=Path("docs/phase4_results.json"))
    a = ap.parse_args()

    ds = NuScenesMini(a.root)
    base = dict(PerceptionConfig.load().tracker)
    max_scenes = a.scenes or None

    print("=" * 78)
    print("Phase 4: nuScenes-mini replay with real sensor timestamps")
    print("=" * 78, flush=True)

    a_cfg = dict(base, temporal_fusion=False)
    b_cfg = dict(base, temporal_fusion=True)
    results = {}
    for motion, title in ((False, "observations exactly as the adapter emits them"),
                          (True, "motion-consistent observations")):
        print(os.linesep + "--- " + title + " ---", flush=True)
        ra = replay(ds, a_cfg, max_scenes, motion)
        rb = replay(ds, b_cfg, max_scenes, motion)
        print(f"scenes {ra['scenes']}   keyframes {ra['samples']}")
        print()
        hdr = f"{'':<38s}{'A synchronised':>16s}{'B temporal':>14s}"
        print(hdr)
        print("-" * len(hdr))
        for label, key, fmt in ROWS:
            print(f"{label:<38s}{fmt.format(ra[key]):>16s}{fmt.format(rb[key]):>14s}")
        results["motion_consistent" if motion else "adapter_as_is"] = {
            "synchronised": ra, "temporal": rb}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
