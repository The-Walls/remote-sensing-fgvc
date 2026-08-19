"""Dataset construction with group-aware splitting.

Domain difficulty #4 (long tail) plus the leakage finding: near-duplicate images
produced by overlapping crops of one source scene must not straddle the split, or
val accuracy measures memorisation. dedup_check.py assigns every image a group id;
here whole groups are dealt to train or val, never split.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from . import transforms as tf

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _group_ids(root: Path, samples, dup_path: str):
    """Group id per sample; identity groups when no dedup file is given."""
    if not dup_path:
        return list(range(len(samples)))
    d = json.loads(Path(dup_path).read_text(encoding="utf-8"))
    # keyed without the suffix so the groups computed on the original tree still
    # apply to the JPEG cache produced by tools/make_cache.py
    lut = {Path(f.replace("\\", "/")).with_suffix("").as_posix(): g
           for f, g in zip(d["files"], d["group_id"])}
    out, missing = [], 0
    for i, (p, _) in enumerate(samples):
        rel = Path(p).relative_to(root).with_suffix("").as_posix()
        if rel in lut:
            out.append(lut[rel])
        else:
            out.append(("__miss__", i))
            missing += 1
    if missing:
        print(f"[data] warning: {missing} files absent from dup_groups.json "
              f"(treated as unique)")
    return out


def split_indices(labels, groups, val_ratio, seed, min_val_per_class, group_aware):
    """Stratified split over groups. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    if not group_aware:
        groups = list(range(len(labels)))

    # a group is atomic; assign it the class of its first member
    by_class = defaultdict(list)
    seen = {}
    for i, (y, g) in enumerate(zip(labels, groups)):
        if g not in seen:
            seen[g] = len(seen)
            by_class[y].append(g)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    train_idx, val_idx = [], []
    for y, gs in by_class.items():
        gs = list(gs)
        rng.shuffle(gs)
        n_val = max(min_val_per_class, int(round(len(gs) * val_ratio)))
        n_val = min(n_val, max(0, len(gs) - 1))   # never leave a class with no train
        for g in gs[:n_val]:
            val_idx += members[g]
        for g in gs[n_val:]:
            train_idx += members[g]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build(cfg, smoke=False):
    root = Path(cfg["data"]["root"])
    train_ds = ImageFolder(root, transform=tf.build(cfg, True), is_valid_file=
                           lambda p: p.lower().endswith(EXTS))
    val_ds = ImageFolder(root, transform=tf.build(cfg, False), is_valid_file=
                         lambda p: p.lower().endswith(EXTS))

    labels = [y for _, y in train_ds.samples]
    groups = _group_ids(root, train_ds.samples, cfg["data"]["dup_groups"])
    tr, va = split_indices(labels, groups, cfg["data"]["val_ratio"], cfg["seed"],
                           cfg["data"]["min_val_per_class"],
                           cfg["data"]["group_aware_split"])

    if smoke:
        tr, va = tr[:50], va[:50]

    counts = np.bincount([labels[i] for i in tr], minlength=len(train_ds.classes))
    meta = {
        "n_classes": len(train_ds.classes),
        "classes": train_ds.classes,
        "n_train": len(tr), "n_val": len(va),
        "n_total_images": len(labels),
        "n_groups": len(set(map(str, groups))),
        "train_class_counts": counts.tolist(),
        "val_class_counts": np.bincount([labels[i] for i in va],
                                        minlength=len(train_ds.classes)).tolist(),
        "train_imbalance": float(counts.max() / max(counts.min(), 1)),
    }

    sampler, shuffle = None, True
    if cfg["data"]["sampler"] == "balanced":
        w = 1.0 / np.maximum(counts, 1)
        sample_w = torch.as_tensor([w[labels[i]] for i in tr], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, len(tr), replacement=True)
        shuffle = False

    def loader(ds, idx, bs, train):
        return DataLoader(
            Subset(ds, idx), batch_size=bs,
            shuffle=shuffle if train else False,
            sampler=sampler if train else None,
            num_workers=cfg["data"]["workers"], pin_memory=True,
            drop_last=train and len(idx) > bs,
            persistent_workers=cfg["data"]["workers"] > 0,
        )

    bs = cfg["data"]["batch_size"]
    return loader(train_ds, tr, bs, True), loader(val_ds, va, bs, False), meta
