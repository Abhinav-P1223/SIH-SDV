"""Phase 7C: the UVH-26 subset as torchvision detection targets.

Reads the Phase 7B manifest, which is the single source of truth for which 750 images are in the
experiment and which split each belongs to. Nothing here re-selects images, and the loader refuses
to run if the manifest and the files on disk disagree.

CLASS IDS
    torchvision reserves 0 for background, so our six classes occupy 1 to 6 and `num_classes` is 7.
    The order is fixed here and written into every checkpoint, because a checkpoint whose class
    order is implicit is a checkpoint that will eventually be read with the wrong one.

WHAT IS DELIBERATELY NOT HERE
    No colour jitter, no scale augmentation, no mosaic. With 500 training images and three epochs
    the experiment is asking one narrow question, and augmentation choices would be another
    variable to defend. Horizontal flip is the single exception: a mirrored traffic scene is still
    a plausible traffic scene, and it is the one transform that cannot mislead.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from autonomy.core.types import ObjectType

# Fixed, ordered, and written into the checkpoint. Index 0 is torchvision's background.
CLASS_NAMES = ["__background__", "MOTORCYCLE", "AUTO_RICKSHAW", "CAR", "BUS", "TRUCK", "BICYCLE"]
NUM_CLASSES = len(CLASS_NAMES)
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
# Our own ObjectType, so a fine-tuned prediction reaches the tracker as the same enum a COCO
# prediction would. This is what makes the fine-tuned model a drop-in replacement.
ID_TO_OBJECT_TYPE = {
    1: ObjectType.MOTORCYCLE,
    2: ObjectType.AUTO_RICKSHAW,
    3: ObjectType.CAR,
    4: ObjectType.BUS,
    5: ObjectType.TRUCK,
    6: ObjectType.BICYCLE,
}

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "uvh26_manifest.json"
DEFAULT_SUBSET = ROOT / "Datasets" / "uvh26_subset"


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class UVH26Subset(Dataset):
    """One split of the Phase 7B subset, as (image tensor, target dict) pairs.

    Boxes are converted from COCO `[x, y, w, h]` to torchvision `[x1, y1, x2, y2]` and clipped to
    the real image bounds. Degenerate boxes are dropped rather than clipped into existence; the
    Phase 7B audit found none, so a non-zero drop count here means something changed on disk.
    """

    def __init__(self, split: str, manifest: dict | None = None,
                 subset_dir: Path | str = DEFAULT_SUBSET, augment: bool = False,
                 seed: int = 20260907):
        self.split = split
        self.manifest = manifest or load_manifest()
        self.dir = Path(subset_dir)
        self.augment = augment
        self.rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        self.cmap = self.manifest["category_map"]

        self.entries = [e for e in self.manifest["entries"] if e["assigned_split"] == split]
        if not self.entries:
            raise ValueError(f"no manifest entries for split {split!r}")

        ann_file = ("UVH-26-MV-Val.json" if split == "test" else "UVH-26-MV-Train.json")
        coco = json.loads((self.dir / "annotations" / ann_file).read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        by_img: dict[int, list] = defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)
        self.dropped = 0
        self.boxes: dict[int, tuple] = {}
        for e in self.entries:
            b, l = [], []
            for a in by_img[e["image_id"]]:
                name = self.cmap.get(cats[a["category_id"]])
                if name is None:
                    continue
                x, y, w, h = a["bbox"]
                x1, y1 = max(0.0, float(x)), max(0.0, float(y))
                x2, y2 = min(float(e["width"]), x1 + float(w)), min(float(e["height"]), y1 + float(h))
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    self.dropped += 1
                    continue
                b.append([x1, y1, x2, y2])
                l.append(NAME_TO_ID[name])
            self.boxes[e["image_id"]] = (b, l)

    def __len__(self) -> int:
        return len(self.entries)

    def path_of(self, i: int) -> Path:
        e = self.entries[i]
        return self.dir / "images" / e["assigned_split"] / e["file_name"]

    def __getitem__(self, i: int):
        e = self.entries[i]
        img = Image.open(self.path_of(i)).convert("RGB")
        W, H = img.size
        b, l = self.boxes[e["image_id"]]
        boxes = np.asarray(b, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(l, dtype=np.int64)

        if self.augment and self.rng.random() < 0.5 and len(boxes):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = boxes.copy()
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([e["image_id"]]),
        }
        return x, target

    # ------------------------------------------------------------------ #
    def class_counts(self) -> Counter:
        c = Counter()
        for _, l in self.boxes.values():
            for i in l:
                c[CLASS_NAMES[i]] += 1
        return c

    def ground_truth(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Boxes and labels for evaluation, without loading the image."""
        b, l = self.boxes[self.entries[i]["image_id"]]
        return np.asarray(b, dtype=np.float32).reshape(-1, 4), np.asarray(l, dtype=np.int64)


def collate(batch):
    """torchvision detection models take lists, not stacked tensors: images differ in size."""
    return tuple(zip(*batch))
