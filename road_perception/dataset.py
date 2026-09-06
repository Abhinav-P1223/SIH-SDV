"""IDD-Lite loader.

Layout, verified against the extracted archive:

    leftImg8bit/{split}/{drive}/{id}_image.jpg
    gtFine/{split}/{drive}/{id}_label.png          semantic, level-1 ids
    gtFine/{split}/{drive}/{id}_inst_label.png     instance, unused here

Level-1 ids, all seven confirmed present, with 255 as ignore:

    0 drivable   1 non-drivable   2 living things   3 vehicles
    4 road-side objects   5 far objects   6 sky

Class 0 is the drivable region; everything the planner needs comes from it. Note that level 1
folds road and drivable-fallback together, and sidewalk, curb and non-drivable-fallback together,
so there is no separate curb class to learn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NUM_CLASSES = 7
IGNORE_INDEX = 255
DRIVABLE = 0
NON_DRIVABLE = 1
CLASS_NAMES = ["drivable", "non-drivable", "living things", "vehicles",
               "road-side objects", "far objects", "sky"]

# 320x227 native; 224 is the nearest multiple of 32, which the stride-32 backbone needs.
INPUT_H, INPUT_W = 224, 320

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Pair:
    image: Path
    mask: Path | None          # None for the unlabelled test split


def find_pairs(root: Path, split: str) -> list[Pair]:
    """Every image in `split`, paired with its semantic mask when one exists.

    Raises on a mask that exists but has no image, which would mean a corrupt extraction.
    """
    img_dir = root / "leftImg8bit" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"no such split: {img_dir}")
    pairs: list[Pair] = []
    for img in sorted(img_dir.rglob("*_image.jpg")):
        rel = img.relative_to(root / "leftImg8bit")
        mask = root / "gtFine" / rel.parent / img.name.replace("_image.jpg", "_label.png")
        pairs.append(Pair(img, mask if mask.exists() else None))

    masks = {p for p in (root / "gtFine" / split).rglob("*_label.png")
             if not p.name.endswith("_inst_label.png")}
    paired = {p.mask for p in pairs if p.mask}
    orphans = masks - paired
    if orphans:
        raise ValueError(f"{len(orphans)} masks in {split} have no image, e.g. {sorted(orphans)[0]}")
    return pairs


class IDDLite(Dataset):
    """Images and level-1 masks, resized to the network's input size.

    Masks are resized with NEAREST so class ids are never interpolated into values that do not
    exist. Augmentation is a horizontal flip only: an Indian road scene flipped left-to-right is
    still a plausible scene, whereas colour or geometric distortion would misrepresent a dataset
    this small.
    """

    def __init__(self, root: Path | str, split: str, augment: bool = False, seed: int = 0):
        self.root = Path(root)
        self.split = split
        self.pairs = find_pairs(self.root, split)
        self.labelled = [p for p in self.pairs if p.mask is not None]
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.labelled) if self.labelled else len(self.pairs)

    def _item(self, i: int) -> Pair:
        return (self.labelled or self.pairs)[i]

    def __getitem__(self, i: int):
        p = self._item(i)
        img = Image.open(p.image).convert("RGB").resize((INPUT_W, INPUT_H), Image.BILINEAR)
        x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD

        if p.mask is None:                       # test split: image only
            y = np.full((INPUT_H, INPUT_W), IGNORE_INDEX, dtype=np.int64)
        else:
            m = Image.open(p.mask).resize((INPUT_W, INPUT_H), Image.NEAREST)
            y = np.asarray(m, dtype=np.int64)

        if self.augment and self.rng.random() < 0.5:
            x, y = x[:, ::-1].copy(), y[:, ::-1].copy()

        return torch.from_numpy(x.transpose(2, 0, 1)), torch.from_numpy(y)


def class_pixel_counts(ds: IDDLite, limit: int | None = None) -> np.ndarray:
    """Pixel count per class over the split, ignoring 255. Used for the loss weights."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i in range(len(ds) if limit is None else min(limit, len(ds))):
        p = ds._item(i)
        if p.mask is None:
            continue
        m = np.asarray(Image.open(p.mask))
        b = np.bincount(m.ravel(), minlength=256)
        counts += b[:NUM_CLASSES]
    return counts


def class_weights(counts: np.ndarray, cap: float = 12.0) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1 and capped.

    The audit measured non-drivable at 2.4% of pixels against drivable at 32%. Unweighted, the
    network reaches high pixel accuracy by predicting drivable almost everywhere, which is
    exactly the failure mode that makes a corridor estimate useless. The cap stops the rarest
    class from dominating the gradient and destabilising a 0.69 M-parameter model.
    """
    freq = counts.astype(np.float64) / max(counts.sum(), 1)
    w = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-8), 0.0)
    w = w / w[w > 0].mean()
    return np.minimum(w, cap).astype(np.float32)
