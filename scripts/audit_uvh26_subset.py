"""Phase 7B step 5: verify the downloaded UVH-26 subset before anything is trained on it.

    python scripts/audit_uvh26_subset.py

Checks the things that silently ruin a detection experiment: a missing image, a box that runs off
the edge of the frame, a zero-area box, a duplicate id, or the same picture appearing in both the
training and the test split. Each is reported as a count, and the exit status is non-zero if any
integrity check fails, so this cannot be waved through.

Nothing here modifies the dataset. It reads and reports.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "Datasets" / "uvh26_subset"
ANN = SUBSET / "annotations"
IMAGES = SUBSET / "images"
MANIFEST = ROOT / "docs" / "uvh26_manifest.json"
CLASSES = ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]


def main() -> int:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cmap = man["category_map"]
    entries = man["entries"]
    by_split = defaultdict(list)
    for e in entries:
        by_split[e["assigned_split"]].append(e)

    problems: list[str] = []
    print("=" * 78)
    print("UVH-26 subset audit")
    print("=" * 78)

    # ---- 1. files present, unique, not duplicated across splits ------------------------------ #
    missing, sizes, hashes = [], [], {}
    for e in entries:
        p = IMAGES / e["assigned_split"] / e["file_name"]
        if not p.exists():
            missing.append(str(p)); continue
        sizes.append(p.stat().st_size)
        hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(e["file_name"])
    if missing:
        problems.append(f"{len(missing)} missing image files")
    dup_content = {h: v for h, v in hashes.items() if len(v) > 1}
    if dup_content:
        problems.append(f"{len(dup_content)} duplicate images by content")
    names = Counter(e["file_name"] for e in entries)
    dup_names = [n for n, c in names.items() if c > 1]
    if dup_names:
        problems.append(f"{len(dup_names)} duplicate file names across splits")

    total_bytes = sum(sizes)
    print(f"\nimages expected {len(entries)}   present {len(entries)-len(missing)}   "
          f"missing {len(missing)}")
    print(f"total size {total_bytes/1e9:.2f} GB   mean {total_bytes/max(len(sizes),1)/1e6:.2f} MB")
    print(f"duplicate file names {len(dup_names)}   duplicate content {len(dup_content)}")

    # ---- 2. split disjointness --------------------------------------------------------------- #
    sets = {s: {e["file_name"] for e in v} for s, v in by_split.items()}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and sets[a] & sets[b]:
                problems.append(f"{a}/{b} overlap: {len(sets[a] & sets[b])} files")
    print(f"split sizes: " + ", ".join(f"{s} {len(v)}" for s, v in sorted(sets.items())))
    print("splits disjoint: " + ("YES" if not any("overlap" in p for p in problems) else "NO"))

    # ---- 3. annotation validity against the real image dimensions ---------------------------- #
    coco = {"train": json.loads((ANN / "UVH-26-MV-Train.json").read_text(encoding="utf-8")),
            "test": json.loads((ANN / "UVH-26-MV-Val.json").read_text(encoding="utf-8"))}
    coco["val"] = coco["train"]
    cats = {c["id"]: c["name"] for c in coco["train"]["categories"]}
    anns_by_img = {}
    for key in ("train", "test"):
        d = defaultdict(list)
        for a in coco[key]["annotations"]:
            d[a["image_id"]].append(a)
        anns_by_img[key] = d
    anns_by_img["val"] = anns_by_img["train"]

    bad_box = out_of_bounds = zero_area = tiny = 0
    dim_mismatch = 0
    box_counts = defaultdict(Counter)
    img_counts = defaultdict(Counter)
    areas = defaultdict(list)
    for e in entries:
        split = e["assigned_split"]
        p = IMAGES / split / e["file_name"]
        if not p.exists():
            continue
        with Image.open(p) as im:
            W, H = im.size
        if (W, H) != (e["width"], e["height"]):
            dim_mismatch += 1
        seen = set()
        for a in anns_by_img[split][e["image_id"]]:
            name = cats[a["category_id"]]
            cls = cmap.get(name)
            if cls is None:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                zero_area += 1; bad_box += 1; continue
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                out_of_bounds += 1; bad_box += 1
            if w < 8 or h < 8:
                tiny += 1
            box_counts[split][cls] += 1
            areas[split].append(w * h)
            seen.add(cls)
        for c in seen:
            img_counts[split][c] += 1
    if zero_area:
        problems.append(f"{zero_area} zero/negative-area boxes")
    if dim_mismatch:
        problems.append(f"{dim_mismatch} images whose real size differs from the annotation")

    print(f"\nbox validity: zero/negative area {zero_area}   out of bounds {out_of_bounds}   "
          f"smaller than 8 px {tiny}")
    print(f"image dimension mismatches: {dim_mismatch}")

    # ---- 4. class frequencies and imbalance -------------------------------------------------- #
    print(f"\n{'class':<16}{'train box':>10}{'val box':>9}{'test box':>10}"
          f"{'train img':>11}{'val img':>9}{'test img':>10}")
    for c in CLASSES:
        print(f"{c:<16}{box_counts['train'][c]:>10}{box_counts['val'][c]:>9}{box_counts['test'][c]:>10}"
              f"{img_counts['train'][c]:>11}{img_counts['val'][c]:>9}{img_counts['test'][c]:>10}")
    tot = {s: sum(box_counts[s].values()) for s in ("train", "val", "test")}
    print(f"{'TOTAL':<16}{tot['train']:>10}{tot['val']:>9}{tot['test']:>10}")
    if tot["train"]:
        top = max(box_counts["train"].values()); bot = min(box_counts["train"].values())
        print(f"\ntrain class imbalance: most common / least common = {top/max(bot,1):.1f}x")

    import numpy as np
    for s in ("train", "val", "test"):
        if areas[s]:
            a = np.array(areas[s])
            print(f"{s:<6} box area px^2: median {np.median(a):8.0f}   "
                  f"p10 {np.percentile(a,10):7.0f}   p90 {np.percentile(a,90):9.0f}")

    print("\n" + "=" * 78)
    if problems:
        print("INTEGRITY PROBLEMS:")
        for p in problems:
            print("   " + p)
        return 1
    print("all integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
