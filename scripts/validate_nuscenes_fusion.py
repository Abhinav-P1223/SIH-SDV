"""Push real nuScenes observations through the EXISTING fusion tracker and report what happens.

Nothing in `autonomy/` is modified or subclassed. The tracker used here is the same
`SensorFusionTracker` the simulator uses, with the same config, so anything it reports is a
property of the real interface rather than of a test harness.

    python scripts/validate_nuscenes_fusion.py --root Datasets/v1.0-mini
    python scripts/validate_nuscenes_fusion.py --root Datasets/v1.0-mini --scenes 3 --verbose
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autonomy.core.config import ObjectProfiles, PerceptionConfig          # noqa: E402
from autonomy.core.types import SensorType                                  # noqa: E402
from autonomy.perception.tracker import SensorFusionTracker, TrackerConfig  # noqa: E402
from dataset_adapters.nuscenes import SENSOR_SYNC, NuScenesMini                     # noqa: E402


def validate(root: Path, max_scenes: int | None, verbose: bool) -> dict:
    ds = NuScenesMini(root)
    perception = PerceptionConfig.load()
    tracker = SensorFusionTracker(TrackerConfig(**perception.tracker), ObjectProfiles.load())

    stats: Counter = Counter()
    spreads_ms: list[float] = []
    ingest_ms: list[float] = []
    track_counts: list[int] = []
    pos_err: list[float] = []
    radial_seen = 0
    scenes = ds.scene[:max_scenes] if max_scenes else ds.scene

    for scene in scenes:
        stats["scenes"] += 1
        tracker = SensorFusionTracker(TrackerConfig(**perception.tracker), ObjectProfiles.load())
        for sample in ds.samples_of_scene(scene["token"]):
            stats["samples"] += 1
            dets = ds.detections_for_sample(sample["token"])
            if not dets:
                stats["samples_without_detections"] += 1
                continue

            for d in dets:
                stats[f"det_{d.sensor.value.lower()}"] += 1
                if d.sensor is SensorType.RADAR:
                    if d.radial_speed is None:
                        stats["radar_without_velocity"] += 1
                    else:
                        radial_seen += 1

            # Sensors inside one keyframe are NOT simultaneous. The tracker takes a single
            # fusion time per ingest, so the alignment strategy is stated explicitly here:
            # ingest at the LATEST sensor timestamp in the batch, which never asks the filter to
            # run backwards, and record the residual spread as association error rather than
            # hiding it. See the report note on timestamp handling.
            stamps = [d.timestamp for d in dets]
            fusion_t = max(stamps)
            spread = (fusion_t - min(stamps)) * 1e3
            spreads_ms.append(spread)
            if spread <= SENSOR_SYNC["keyframe_spread_max_ms"] + 1.0:
                stats["timestamp_associations_ok"] += 1
            else:
                stats["timestamp_associations_out_of_spec"] += 1

            ego = ds.ego_state(next(sd for sd in ds.data_by_sample[sample["token"]]
                                    if sd["is_key_frame"]))
            t0 = time.perf_counter()
            tracker.ingest(dets, fusion_t, ego)
            ingest_ms.append((time.perf_counter() - t0) * 1e3)

            tracks = tracker.get_object_states(fusion_t)
            track_counts.append(len(tracks))
            stats["fused_objects"] += len(tracks)

            # How close did fused tracks land to the annotated boxes they came from? This is an
            # interface sanity check, not a perception benchmark: the observations were derived
            # from those same boxes, so anything much above the sensor noise means the adapter
            # or the fusion path lost something.
            truth = {a["instance_token"]: a["translation"]
                     for a in ds.ann_by_sample.get(sample["token"], [])}
            for tr in tracks:
                best = min((abs(tr.x - t[0]) + abs(tr.y - t[1]), t) for t in truth.values()) \
                    if truth else None
                if best and best[0] < 10.0:
                    pos_err.append(float(np.hypot(tr.x - best[1][0], tr.y - best[1][1])))

        if verbose:
            print(f"  scene {scene['name']}: {stats['samples']} samples so far, "
                  f"{len(tracker.tracks)} live tracks at end")

    stats["radar_with_velocity"] = radial_seen
    return {
        "stats": stats, "spreads_ms": spreads_ms, "ingest_ms": ingest_ms,
        "track_counts": track_counts, "pos_err": pos_err,
    }


def report(r: dict) -> None:
    s, spreads, lat = r["stats"], r["spreads_ms"], r["ingest_ms"]
    tc, pe = r["track_counts"], r["pos_err"]
    line = "=" * 78
    print(f"\n{line}\nnuScenes -> existing fusion interface: validation\n{line}")

    print(f"\nscenes processed                    {s['scenes']}")
    print(f"keyframe samples processed          {s['samples']}")
    print(f"samples with no observation         {s['samples_without_detections']}")
    print(f"\ncamera observations                 {s['det_camera']}")
    print(f"LiDAR observations                  {s['det_lidar']}")
    print(f"radar observations                  {s['det_radar']}")
    print(f"  of which carry a range rate       {s['radar_with_velocity']}")
    print(f"  with no derivable velocity        {s['radar_without_velocity']}")

    print(f"\ntimestamp associations in spec      {s['timestamp_associations_ok']}")
    print(f"timestamp associations out of spec  {s['timestamp_associations_out_of_spec']}")
    if spreads:
        print(f"intra-keyframe sensor spread        median {statistics.median(spreads):.1f} ms, "
              f"max {max(spreads):.1f} ms")

    print(f"\nfused objects (sum over samples)    {s['fused_objects']}")
    if tc:
        print(f"live tracks per sample              mean {statistics.mean(tc):.1f}, max {max(tc)}")
    if pe:
        print(f"fused track vs annotated box        median {statistics.median(pe):.2f} m, "
              f"p95 {np.percentile(pe, 95):.2f} m")
    if lat:
        print(f"\ningest latency per sample           mean {statistics.mean(lat):.1f} ms, "
              f"p95 {np.percentile(lat, 95):.1f} ms, max {max(lat):.1f} ms")

    print(f"\nmeasured sensor rates on this data  camera {SENSOR_SYNC['camera_hz']} Hz, "
          f"lidar {SENSOR_SYNC['lidar_hz']} Hz, radar {SENSOR_SYNC['radar_hz']} Hz")
    print("simulator config for comparison     camera 20 Hz, lidar 10 Hz, radar 20 Hz "
          "(unchanged: quantified here, not fixed)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("Datasets/v1.0-mini"))
    ap.add_argument("--scenes", type=int, default=None, help="limit the number of scenes")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    report(validate(a.root, a.scenes, a.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
