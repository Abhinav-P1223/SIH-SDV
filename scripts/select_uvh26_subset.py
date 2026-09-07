"""Phase 7B: choose a small, DELIBERATELY NOT EASY subset of UVH-26, then fetch only those images.

    python scripts/select_uvh26_subset.py --plan       # selection only, writes the manifest
    python scripts/select_uvh26_subset.py --download   # fetch the selected images

WHY NOT JUST TAKE THE FIRST N, OR THE BUSIEST N
    Both would be worse than random. The first N are contiguous file ids, which in a dataset built
    from ~2,800 cameras means a handful of scenes. The busiest N would be crowded frames full of
    large, near, unoccluded vehicles: an artificially easy set that would flatter any model trained
    on it and prove nothing about the Phase 6 failures we are trying to fix.

SELECTION, IN THREE PASSES
    Every image is first described by two axes that matter for difficulty:

      density  = how many annotated objects it holds      sparse / medium / dense
      scale    = the median box area in the image         small / mid / large

    "small" here is genuinely small: the lowest tertile of median box area across the whole split.
    Those are the far, cluttered, partially occluded frames, and they are the ones a 320 px
    detector fails on, so the sample must contain them rather than avoid them.

    Pass 1, RARE-CLASS FLOOR. Classes are wildly imbalanced: two-wheelers are 47% of all boxes and
    mini-buses 0.3%. Sampling uniformly would leave bicycles and buses almost absent. Each target
    class is therefore given an image floor, filled rarest-first, drawing round-robin across the
    nine density x scale cells so the floor does not itself become a bias.

    Pass 2, CELL BALANCE. The remaining slots are filled round-robin over the same nine cells, so
    sparse-and-far frames get the same share as dense-and-near ones.

    Pass 3, ID SPREAD. Within any cell, candidates are ordered by a seeded shuffle rather than by
    file id, so the selection is spread across the id range instead of clustering in a few scenes.
    UVH-26 publishes no camera identifier, so this is a proxy for scene diversity and is reported
    as one rather than claimed as camera coverage.

LEAKAGE
    Train and validation are drawn from UVH-26-Train, test exclusively from UVH-26-Val. The three
    sets are disjoint by construction and the script asserts it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://huggingface.co/datasets/iisc-aim/UVH-26/resolve/main/"
SUBSET = ROOT / "Datasets" / "uvh26_subset"
ANN = SUBSET / "annotations"
IMAGES = SUBSET / "images"
MANIFEST = ROOT / "docs" / "uvh26_manifest.json"
SEED = 20260907

# --------------------------------------------------------------------------------------------- #
# Category mapping. Every merge is justified; nothing semantically different is silently combined.
# --------------------------------------------------------------------------------------------- #
CATEGORY_MAP: dict[str, str] = {
    # Our two priority classes map one-to-one.
    "Two-wheeler":     "MOTORCYCLE",     # motorcycles and scooters; our MOTORCYCLE covers both
    "Three-wheeler":   "AUTO_RICKSHAW",  # passenger and goods autos; both are auto-rickshaws
    # Passenger cars by body style. These are four sub-types of one thing, not four things.
    "Hatchback":       "CAR",
    "Sedan":           "CAR",
    "SUV":             "CAR",
    "MUV":             "CAR",
    "Van":             "CAR",            # Indian "Van" is a car-sized passenger vehicle (Omni, Eeco)
    # Passenger road transport, by size.
    "Bus":             "BUS",
    "Mini-bus":        "BUS",
    "Tempo-traveller": "BUS",            # 12-15 seat passenger minibus, not a goods vehicle
    # Goods vehicles.
    "Truck":           "TRUCK",
    "LCV":             "TRUCK",          # light commercial goods vehicle (Tata Ace and similar)
    "Bicycle":         "BICYCLE",
}
# "Others" is 0.1% of boxes and its contents are undefined. It is NOT mapped, and any image
# containing one is EXCLUDED from the pool rather than kept with an unlabelled object in it,
# because an unlabelled object teaches the detector that whatever it is counts as background.
EXCLUDE_CATEGORY = "Others"

TARGET_CLASSES = ["MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]
# Image floors, set from how rare each class is rather than from what we would like to have.
CLASS_FLOOR_TRAIN = {"BICYCLE": 70, "BUS": 90, "TRUCK": 110, "AUTO_RICKSHAW": 120, "CAR": 120}
CLASS_FLOOR_VAL = {"BICYCLE": 15, "BUS": 18, "TRUCK": 22, "AUTO_RICKSHAW": 25, "CAR": 25}
CLASS_FLOOR_TEST = {"BICYCLE": 22, "BUS": 27, "TRUCK": 33, "AUTO_RICKSHAW": 38, "CAR": 38}

SPLIT_SIZES = {"train": 500, "val": 100, "test": 150}


# --------------------------------------------------------------------------------------------- #
def load(name: str) -> dict:
    return json.loads((ANN / name).read_text(encoding="utf-8"))


def describe(coco: dict) -> tuple[dict, dict]:
    """Per-image class counts and difficulty features."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    per_img: dict[int, Counter] = defaultdict(Counter)
    areas: dict[int, list] = defaultdict(list)
    excluded: set[int] = set()
    for a in coco["annotations"]:
        name = cats[a["category_id"]]
        if name == EXCLUDE_CATEGORY:
            excluded.add(a["image_id"])
            continue
        mapped = CATEGORY_MAP.get(name)
        if mapped is None:
            excluded.add(a["image_id"])
            continue
        per_img[a["image_id"]][mapped] += 1
        w, h = a["bbox"][2], a["bbox"][3]
        areas[a["image_id"]].append(float(w) * float(h))
    meta = {}
    for im in coco["images"]:
        i = im["id"]
        if i in excluded or i not in per_img:
            continue
        ar = sorted(areas[i])
        meta[i] = {
            "file_name": im["file_name"], "width": im["width"], "height": im["height"],
            "counts": dict(per_img[i]), "n_boxes": sum(per_img[i].values()),
            "median_area": ar[len(ar) // 2],
        }
    return meta, cats


def cell_of(m: dict, area_cuts: tuple[float, float]) -> tuple[str, str]:
    n = m["n_boxes"]
    density = "sparse" if n <= 4 else ("medium" if n <= 12 else "dense")
    a = m["median_area"]
    scale = "small" if a <= area_cuts[0] else ("mid" if a <= area_cuts[1] else "large")
    return density, scale


def pick(meta: dict, size: int, floors: dict, rng: random.Random,
         taken: set[str]) -> list[int]:
    """The three passes described in the module docstring."""
    areas = sorted(m["median_area"] for m in meta.values())
    cuts = (areas[len(areas) // 3], areas[2 * len(areas) // 3])
    cells: dict[tuple, list] = defaultdict(list)
    for i, m in meta.items():
        if m["file_name"] in taken:
            continue
        cells[cell_of(m, cuts)].append(i)
    for v in cells.values():
        rng.shuffle(v)                                   # pass 3: seeded id spread

    chosen: list[int] = []
    chosen_set: set[int] = set()
    cell_keys = sorted(cells)

    def take(i):
        chosen.append(i); chosen_set.add(i)

    # pass 1: rare-class floors, rarest first, round-robin over cells
    have = Counter()
    for cls in sorted(floors, key=lambda c: floors[c]):
        k = 0
        while have[cls] < floors[cls] and len(chosen) < size:
            progressed = False
            for key in cell_keys:
                bucket = cells[key]
                for j, i in enumerate(bucket):
                    if i in chosen_set or cls not in meta[i]["counts"]:
                        continue
                    take(i)
                    for c in meta[i]["counts"]:
                        have[c] += 1
                    bucket.pop(j)
                    progressed = True
                    break
                if have[cls] >= floors[cls] or len(chosen) >= size:
                    break
            if not progressed:
                break
            k += 1

    # pass 2: fill the rest round-robin over the nine cells
    while len(chosen) < size:
        progressed = False
        for key in cell_keys:
            if len(chosen) >= size:
                break
            bucket = cells[key]
            while bucket:
                i = bucket.pop()
                if i in chosen_set:
                    continue
                take(i)
                progressed = True
                break
        if not progressed:
            break
    return chosen


def summarise(meta: dict, ids: list[int]) -> dict:
    boxes = Counter()
    imgs = Counter()
    dens = Counter()
    for i in ids:
        m = meta[i]
        for c, n in m["counts"].items():
            boxes[c] += n
            imgs[c] += 1
        n = m["n_boxes"]
        dens["sparse" if n <= 4 else ("medium" if n <= 12 else "dense")] += 1
    return {"images": len(ids), "boxes": dict(boxes), "images_with_class": dict(imgs),
            "density": dict(dens),
            "total_boxes": sum(boxes.values())}


def build_manifest() -> dict:
    rng = random.Random(SEED)
    tr = load("UVH-26-MV-Train.json")
    va = load("UVH-26-MV-Val.json")
    meta_tr, _ = describe(tr)
    meta_va, _ = describe(va)
    print(f"pool after excluding '{EXCLUDE_CATEGORY}' and unmapped: "
          f"train {len(meta_tr)} of {len(tr['images'])}, val {len(meta_va)} of {len(va['images'])}")

    taken: set[str] = set()
    train_ids = pick(meta_tr, SPLIT_SIZES["train"], CLASS_FLOOR_TRAIN, rng, taken)
    taken |= {meta_tr[i]["file_name"] for i in train_ids}
    val_ids = pick(meta_tr, SPLIT_SIZES["val"], CLASS_FLOOR_VAL, rng, taken)
    taken |= {meta_tr[i]["file_name"] for i in val_ids}
    test_ids = pick(meta_va, SPLIT_SIZES["test"], CLASS_FLOOR_TEST, rng, set())

    assert not (set(train_ids) & set(val_ids)), "train/val overlap"
    fn = lambda ids, m: {m[i]["file_name"] for i in ids}
    assert not (fn(train_ids, meta_tr) & fn(val_ids, meta_tr)), "train/val filename overlap"
    assert not (fn(train_ids, meta_tr) & fn(test_ids, meta_va)), "train/test filename overlap"
    assert not (fn(val_ids, meta_tr) & fn(test_ids, meta_va)), "val/test filename overlap"

    index = json.loads((SUBSET / "file_index.json").read_text(encoding="utf-8"))
    entries = []
    for split, ids, meta, src in (("train", train_ids, meta_tr, "UVH-26-Train"),
                                  ("val", val_ids, meta_tr, "UVH-26-Train"),
                                  ("test", test_ids, meta_va, "UVH-26-Val")):
        for i in ids:
            m = meta[i]
            entries.append({
                "image_id": i, "source_split": src, "assigned_split": split,
                "file_name": m["file_name"], "repo_path": index[m["file_name"]],
                "width": m["width"], "height": m["height"],
                "n_boxes": m["n_boxes"], "counts": m["counts"],
            })
    return {
        "dataset": "UVH-26 (AIM @ IISc)", "url": "https://huggingface.co/datasets/iisc-aim/UVH-26",
        "license": "CC BY 4.0", "annotations": "majority-vote (MV) consensus files",
        "seed": SEED, "category_map": CATEGORY_MAP, "excluded_category": EXCLUDE_CATEGORY,
        "split_sizes": SPLIT_SIZES,
        "summary": {"train": summarise(meta_tr, train_ids),
                    "val": summarise(meta_tr, val_ids),
                    "test": summarise(meta_va, test_ids)},
        "entries": entries,
    }


def download(manifest: dict) -> None:
    total = 0
    for e in manifest["entries"]:
        dst = IMAGES / e["assigned_split"] / e["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            total += dst.stat().st_size
            continue
        # Transient DNS and connection failures are normal over a 750-file fetch, so each file
        # is retried with a backoff. A partially written file is removed before retrying, because
        # a truncated PNG that survives to training time is far worse than a failed download.
        for attempt in range(6):
            try:
                r = requests.get(BASE + e["repo_path"], timeout=300, stream=True)
                r.raise_for_status()
                h = hashlib.sha256()
                with open(dst, "wb") as f:
                    for c in r.iter_content(1 << 20):
                        f.write(c); h.update(c)
                e["sha256"] = h.hexdigest()
                break
            except Exception as exc:
                if dst.exists():
                    dst.unlink()
                if attempt == 5:
                    raise
                print(f"    retry {attempt+1}/5 for {e['file_name']}: {type(exc).__name__}",
                      flush=True)
                time.sleep(2 ** attempt)
        total += dst.stat().st_size
        n = sum(1 for x in manifest["entries"] if (IMAGES / x["assigned_split"] / x["file_name"]).exists())
        if n % 50 == 0:
            print(f"  {n}/{len(manifest['entries'])} images, {total/1e9:.2f} GB", flush=True)
    print(f"downloaded {len(manifest['entries'])} images, {total/1e9:.2f} GB total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    for split, s in manifest["summary"].items():
        print(f"\n{split}: {s['images']} images, {s['total_boxes']} boxes, density {s['density']}")
        for c in TARGET_CLASSES:
            print(f"    {c:<15}{s['boxes'].get(c,0):>7} boxes  in {s['images_with_class'].get(c,0):>4} images")
    if a.download:
        download(manifest)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
