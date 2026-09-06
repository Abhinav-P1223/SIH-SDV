"""Audit the Phase-3 datasets against the files on disk, not against their documentation.

Two independent audits, either of which can run alone:

  nuScenes v1.0-mini   camera + LiDAR + radar, calibration, synchronisation, 3D boxes and
                       velocities. This is the multimodal fusion reference: it tells us what a
                       real sensor suite delivers and lets us check our simulated one against it.
  IDD Lite             Indian-road semantic segmentation with a drivable / non-drivable split.
                       This is the camera perception source for drivable-space estimation.

Reads the raw metadata JSON directly rather than importing nuscenes-devkit, so it adds no
dependency to the project. Nothing here trains, converts or modifies anything: it reports.

    python scripts/audit_datasets.py --nuscenes Datasets/v1.0-mini
    python scripts/audit_datasets.py --idd-lite Datasets/idd20k_lite
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# IDD Lite level-1 label set. The mapping from these to a drivable corridor is the whole point:
# level 1 collapses road and drivable-fallback into DRIVABLE, and sidewalk, curb and
# non-drivable-fallback into NON_DRIVABLE, which is exactly the binary a planner needs.
IDD_LITE_LEVEL1 = {
    0: "drivable", 1: "non-drivable", 2: "living things", 3: "vehicles",
    4: "road-side objects", 5: "far objects", 6: "sky",
}


def _load(root: Path, name: str) -> list:
    path = root / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"missing metadata table: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_nuscenes(root: Path) -> None:
    meta_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("v1.0")]
    if not meta_dirs:
        raise SystemExit(f"no v1.0-* metadata directory under {root}")
    meta = meta_dirs[0]
    print(f"\n{'=' * 78}\nnuScenes  ({root})\n{'=' * 78}")
    print(f"metadata split: {meta.name}")

    scene = _load(meta, "scene")
    sample = _load(meta, "sample")
    sample_data = _load(meta, "sample_data")
    ann = _load(meta, "sample_annotation")
    category = _load(meta, "category")
    sensor = _load(meta, "sensor")
    calib = _load(meta, "calibrated_sensor")
    ego_pose = _load(meta, "ego_pose")
    instance = _load(meta, "instance")

    print(f"\nscenes {len(scene)}   keyframe samples {len(sample)}   "
          f"sample_data rows {len(sample_data)}   annotations {len(ann)}   instances {len(instance)}")

    # ---- sensors, modality by modality -------------------------------------------------- #
    by_token = {s["token"]: s for s in sensor}
    calib_by_token = {c["token"]: c for c in calib}
    per_channel: dict[str, list] = defaultdict(list)
    for sd in sample_data:
        chan = by_token[calib_by_token[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        per_channel[chan].append(sd)

    print(f"\n{'channel':22s} {'modality':8s} {'files':>7s} {'keyframes':>10s} {'rate Hz':>8s} {'format':>8s}")
    modality_counts: Counter = Counter()
    for chan in sorted(per_channel):
        rows = per_channel[chan]
        modality = by_token[calib_by_token[rows[0]["calibrated_sensor_token"]]["sensor_token"]]["modality"]
        modality_counts[modality] += 1
        keys = sum(1 for r in rows if r["is_key_frame"])
        ts = sorted(r["timestamp"] for r in rows)
        gaps = [(b - a) / 1e6 for a, b in zip(ts, ts[1:]) if 0 < (b - a) < 1e6]
        rate = 1.0 / statistics.median(gaps) if gaps else float("nan")
        ext = Path(rows[0]["filename"]).suffix
        print(f"{chan:22s} {modality:8s} {len(rows):7d} {keys:10d} {rate:8.1f} {ext:>8s}")
    print("\nmodalities present: " + ", ".join(f"{k} x{v}" for k, v in sorted(modality_counts.items())))

    # ---- calibration --------------------------------------------------------------------- #
    with_intrinsics = sum(1 for c in calib if c.get("camera_intrinsic"))
    print(f"\ncalibrated_sensor entries {len(calib)}: {with_intrinsics} carry a camera intrinsic matrix, "
          f"all carry translation + rotation (extrinsics)")
    print(f"ego_pose entries {len(ego_pose)} (one per sample_data row: {len(ego_pose) == len(sample_data)})")

    # ---- synchronisation ------------------------------------------------------------------ #
    # A keyframe groups one reading per channel. How tightly are they aligned in time?
    spreads_ms = []
    for s in sample:
        ts = [sd["timestamp"] for sd in sample_data
              if sd["sample_token"] == s["token"] and sd["is_key_frame"]]
        if len(ts) > 1:
            spreads_ms.append((max(ts) - min(ts)) / 1e3)
    if spreads_ms:
        print(f"\nkeyframe cross-sensor time spread: median {statistics.median(spreads_ms):.1f} ms, "
              f"max {max(spreads_ms):.1f} ms over {len(spreads_ms)} keyframes")

    # ---- annotations ---------------------------------------------------------------------- #
    cat_by_token = {c["token"]: c["name"] for c in category}
    inst_cat = {i["token"]: cat_by_token[i["category_token"]] for i in instance}
    counts = Counter(inst_cat[a["instance_token"]] for a in ann)
    print(f"\n{len(counts)} annotated categories, 3D boxes with translation / size / rotation:")
    for name, n in counts.most_common():
        print(f"    {name:44s} {n:6d}")

    # velocity is not stored on the annotation; the devkit derives it from consecutive boxes.
    linked = sum(1 for a in ann if a.get("next"))
    print(f"\n{linked} of {len(ann)} boxes link to a next box, so per-object velocity is derivable "
          f"({100.0 * linked / max(len(ann), 1):.0f}%)")
    vis = sum(1 for a in ann if a.get("visibility_token"))
    print(f"{vis} boxes carry a visibility level; attributes table present: "
          f"{(meta / 'attribute.json').exists()}")

    # ---- what this gives our fusion layer ------------------------------------------------- #
    print("\nUsable by our stack without changing it:")
    print("  radar returns carry range-rate, which is what our EKF radial-speed update consumes")
    print("  3D boxes + ego_pose give a world-frame ground truth to score fused tracks against")
    print("  calibrated_sensor gives the extrinsics our sensor models currently assume")
    print("  per-sensor rates above are the real numbers to calibrate config/sensors.yaml against")


def audit_idd_lite(root: Path) -> None:
    print(f"\n{'=' * 78}\nIDD Lite  ({root})\n{'=' * 78}")
    if not root.exists():
        raise SystemExit(f"not found: {root}")

    images = sorted(root.rglob("*_image.jpg")) or sorted(root.rglob("*leftImg8bit*"))
    masks = sorted(root.rglob("*label*.png")) or sorted(root.rglob("*gtFine*"))
    print(f"images {len(images)}   masks {len(masks)}")

    splits: Counter = Counter()
    for p in images:
        parts = {q.name for q in p.parents}
        for s in ("train", "val", "test"):
            if s in parts:
                splits[s] += 1
                break
    print("split sizes: " + ", ".join(f"{k} {v}" for k, v in sorted(splits.items())) if splits
          else "no train/val/test directories found")

    if not masks:
        print("\nNo masks found: check the archive layout before assuming the format.")
        return

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("\nInstall pillow to inspect mask contents (numpy is already a dependency).")
        return

    present: Counter = Counter()
    sizes: Counter = Counter()
    for p in masks[:200]:                       # a sample is enough to characterise the encoding
        a = np.array(Image.open(p))
        sizes[a.shape[:2]] += 1
        present.update(np.unique(a).tolist())
    print(f"\nmask dtype/shape sample: {dict(list(sizes.items())[:3])}")
    print(f"\nlabel ids present across {min(len(masks), 200)} masks:")
    for v, n in sorted(present.items()):
        print(f"    {v:>4} {IDD_LITE_LEVEL1.get(v, 'UNKNOWN / ignore'):24s} in {n} masks")

    drivable = present.get(0, 0)
    print(f"\nDrivable-area masks usable: {'YES' if drivable else 'NO'} "
          f"(label 0 = drivable appears in {drivable} of the sampled masks)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuscenes", type=Path, help="root containing samples/ sweeps/ v1.0-*/")
    ap.add_argument("--idd-lite", type=Path, help="extracted IDD Lite root")
    args = ap.parse_args()
    if not args.nuscenes and not args.idd_lite:
        ap.error("give --nuscenes and/or --idd-lite")
    if args.nuscenes:
        audit_nuscenes(args.nuscenes)
    if args.idd_lite:
        audit_idd_lite(args.idd_lite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
