"""Dataset construction with group-aware three-way splitting.

Domain difficulty #4 (long tail) plus the leakage finding: near-duplicate images
produced by overlapping crops of one source scene must not straddle a split, or
held-out accuracy measures memorisation. dedup_check.py assigns every image a
group id; here whole groups are dealt to train, val or test, never split.

val picks the epoch; test is scored once, at the val-selected epoch. Scoring on
the split that picked the epoch reports the maximum of a noisy sequence, not
the model (REPORT section 4).
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


def split_indices(labels, groups, val_ratio, test_ratio, seed, min_holdout_per_class,
                  group_aware):
    """Stratified three-way split over groups. Returns (train, val, test) indices.

    Per class, test gets max(min_holdout, round(n * test_ratio)) groups and val
    the same with val_ratio; train keeps the rest. A class never loses its last
    training group: val shrinks first, then test. So a class with 2 unique
    groups becomes test 1 / train 1 / val 0, and a class with 1 is train-only.
    """
    rng = np.random.default_rng(seed)
    if not group_aware:
        groups = list(range(len(labels)))

    # a group is atomic; it takes the class of its first member
    by_class = defaultdict(list)
    seen = set()
    for y, g in zip(labels, groups):
        if g not in seen:
            seen.add(g)
            by_class[y].append(g)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)

    train, val, test = [], [], []
    for gs in by_class.values():
        gs = list(gs)
        rng.shuffle(gs)
        n = len(gs)
        n_test = max(min_holdout_per_class, int(round(n * test_ratio)))
        n_val = max(min_holdout_per_class, int(round(n * val_ratio)))
        n_val = min(n_val, max(0, n - 1 - n_test))
        n_test = min(n_test, max(0, n - 1 - n_val))
        for g in gs[:n_test]:
            test += members[g]
        for g in gs[n_test:n_test + n_val]:
            val += members[g]
        for g in gs[n_test + n_val:]:
            train += members[g]
    for idx in (train, val, test):
        rng.shuffle(idx)
    return train, val, test


def build(cfg, smoke=False):
    """-> train_loader, val_loader, test_loader, meta."""
    d = cfg["data"]
    root = Path(d["root"])

    def is_img(p):
        return p.lower().endswith(EXTS)

    train_ds = ImageFolder(root, transform=tf.build(cfg, True), is_valid_file=is_img)
    eval_ds = ImageFolder(root, transform=tf.build(cfg, False), is_valid_file=is_img)

    labels = [y for _, y in train_ds.samples]
    groups = _group_ids(root, train_ds.samples, d["dup_groups"])
    tr, va, te = split_indices(labels, groups, d["val_ratio"], d["test_ratio"],
                               cfg["seed"], d["min_holdout_per_class"],
                               d["group_aware_split"])
    if smoke:
        tr, va, te = tr[:50], va[:50], te[:50]

    n_cls = len(train_ds.classes)

    def counts(idx):
        return np.bincount([labels[i] for i in idx], minlength=n_cls)

    tr_c, va_c, te_c = counts(tr), counts(va), counts(te)
    meta = {
        "n_classes": n_cls,
        "classes": train_ds.classes,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_total_images": len(labels),
        "n_groups": len(set(map(str, groups))),
        "train_class_counts": tr_c.tolist(),
        "val_class_counts": va_c.tolist(),
        "test_class_counts": te_c.tolist(),
        # a class with too few unique groups cannot be held out; its metric
        # simply has no support there (metrics.summarize masks it out)
        "n_classes_absent_from_val": int((va_c == 0).sum()),
        "n_classes_absent_from_test": int((te_c == 0).sum()),
        "train_imbalance": float(tr_c.max() / max(tr_c.min(), 1)),
    }

    sampler, shuffle = None, True
    if d["sampler"] == "balanced":
        w = 1.0 / np.maximum(tr_c, 1)
        sample_w = torch.as_tensor([w[labels[i]] for i in tr], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, len(tr), replacement=True)
        shuffle = False

    def loader(ds, idx, train):
        # eval loaders take half the workers so the three loaders together keep
        # the same process count as before (Windows shared-memory limit, see
        # configs/base.yaml `workers`)
        workers = d["workers"] if train else max(1, d["workers"] // 2)
        return DataLoader(
            Subset(ds, idx), batch_size=d["batch_size"],
            shuffle=shuffle if train else False,
            sampler=sampler if train else None,
            num_workers=workers, pin_memory=True,
            drop_last=train and len(idx) > d["batch_size"],
            persistent_workers=workers > 0,
        )

    return (loader(train_ds, tr, True), loader(eval_ds, va, False),
            loader(eval_ds, te, False), meta)
