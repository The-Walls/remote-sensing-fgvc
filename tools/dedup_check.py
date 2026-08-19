"""Near-duplicate / augmentation-copy detection for an ImageFolder dataset.

Why this exists: remote-sensing chip datasets are often assembled by cropping
overlapping tiles from the same source scene, and some are padded with rotated
or flipped copies of the same chip. If such copies straddle a train/val split,
val accuracy measures memorisation, not generalisation.

Method:
  - aHash (8x8 mean threshold) and pHash (32x32 DCT, top-left 8x8 vs median),
    both 64-bit.
  - For every image the hashes are also computed under rot90/180/270 and
    h/v flip, so an augmented copy still matches its source.
  - Two images are duplicates when SOME variant of one is within Hamming
    distance <= THRESH of the other's identity hash, on BOTH hashes.
  - Duplicates are unioned into groups; a group must stay inside one split.

Outputs dup_groups.json (consumed by src/data.py), a report, and a montage of
example pairs so the threshold can be eyeballed.

Usage:
    python tools/dedup_check.py <root> [--out DIR] [--thresh 5] [--val-ratio 0.25]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VARIANTS = ["id", "rot90", "rot180", "rot270", "hflip", "vflip"]


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    m = np.cos(np.pi * (x + 0.5) * k / n)
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_D32 = _dct_matrix(32)


def _bits_to_u64(bits: np.ndarray) -> np.uint64:
    return np.uint64(int("".join("1" if b else "0" for b in bits.ravel()), 2))


def _ahash(gray8: np.ndarray) -> np.uint64:
    return _bits_to_u64(gray8 > gray8.mean())


def _phash(gray32: np.ndarray) -> np.uint64:
    dct = _D32 @ gray32 @ _D32.T
    low = dct[:8, :8].copy()
    low[0, 0] = 0.0                      # drop DC: it only encodes mean brightness
    return _bits_to_u64(low > np.median(low))


def _variants(arr: np.ndarray):
    yield "id", arr
    yield "rot90", np.rot90(arr, 1)
    yield "rot180", np.rot90(arr, 2)
    yield "rot270", np.rot90(arr, 3)
    yield "hflip", arr[:, ::-1]
    yield "vflip", arr[::-1, :]


CENTER_FRAC = 0.6   # linear fraction kept by the centre crop


def _hash_one(gray):
    g8 = np.asarray(gray.resize((8, 8), Image.BILINEAR), dtype=np.float64)
    g32 = np.asarray(gray.resize((32, 32), Image.BILINEAR), dtype=np.float64)
    # rotate/flip the resized grids: cheaper than re-decoding, and identical to
    # transforming the source for the square grids the hashes are computed on.
    a = [_ahash(np.ascontiguousarray(gg)) for _, gg in _variants(g8)]
    p = [_phash(np.ascontiguousarray(gg)) for _, gg in _variants(g32)]
    return a, p


def hash_dataset(files):
    """-> full-frame and centre-crop hashes, each (a[V,N], p[V,N]) uint64.

    Two hash scales are needed because the target occupies few pixels: at 8x8 /
    32x32 the full frame is dominated by background, so a full-frame match alone
    cannot distinguish "same image" from "same background plate, different
    target". The centre crop re-weights the comparison onto the object.
    """
    V, N = len(VARIANTS), len(files)
    a = np.zeros((V, N), dtype=np.uint64)
    p = np.zeros((V, N), dtype=np.uint64)
    ac = np.zeros((V, N), dtype=np.uint64)
    pc = np.zeros((V, N), dtype=np.uint64)
    for i, f in enumerate(files):
        with Image.open(f) as im:
            g = im.convert("L")
            w, h = g.size
            cw, ch = int(w * CENTER_FRAC), int(h * CENTER_FRAC)
            gc = g.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
            av, pv = _hash_one(g)
            acv, pcv = _hash_one(gc)
        a[:, i], p[:, i], ac[:, i], pc[:, i] = av, pv, acv, pcv
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{len(files)}", flush=True)
    return a, p, ac, pc


def find_pairs(a, p, ac, pc, thresh, chunk=512):
    """Split matches into true duplicates and shared-background-only pairs.

    full match  = aHash and pHash both within thresh on the whole frame
    centre match= same, on the centre crop
    duplicate   -> full AND centre;  background plate -> full but NOT centre
    """
    n = a.shape[1]
    dup, bg = set(), set()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        hit_full = np.zeros((e - s, n), dtype=bool)
        hit_ctr = np.zeros((e - s, n), dtype=bool)
        for v in range(len(VARIANTS)):
            ha = np.bitwise_count(a[v, s:e, None] ^ a[0][None, :])
            hp = np.bitwise_count(p[v, s:e, None] ^ p[0][None, :])
            hit_full |= (ha <= thresh) & (hp <= thresh)
            hac = np.bitwise_count(ac[v, s:e, None] ^ ac[0][None, :])
            hpc = np.bitwise_count(pc[v, s:e, None] ^ pc[0][None, :])
            hit_ctr |= (hac <= thresh) & (hpc <= thresh)
        for li, gi in enumerate(range(s, e)):
            for gj in np.nonzero(hit_full[li])[0]:
                gj = int(gj)
                if gi == gj:
                    continue
                key = (min(gi, gj), max(gi, gj))
                (dup if hit_ctr[li, gj] else bg).add(key)
        print(f"  compared {e}/{n}", flush=True)
    return sorted(dup), sorted(bg - dup)


def union_groups(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n)]


def naive_split_leakage(labels, groups, pairs, val_ratio, seed=0):
    """Cross-split duplicate pairs a plain stratified split would produce."""
    rng = np.random.default_rng(seed)
    split = np.zeros(len(labels), dtype=np.int8)      # 0 train, 1 val
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        rng.shuffle(idx)
        split[idx[: max(1, int(round(len(idx) * val_ratio)))]] = 1
    cross = [(i, j) for i, j in pairs if split[i] != split[j]]
    leaked_val = {j if split[j] == 1 else i for i, j in cross}
    n_val = int(split.sum())
    return cross, leaked_val, n_val


def montage(files, pairs, out_png, k=12):
    if not pairs:
        return
    sel = pairs[:k]
    rows = len(sel)
    fig, axes = plt.subplots(rows, 2, figsize=(4.2, 2.1 * rows))
    axes = np.atleast_2d(axes)
    for r, (i, j) in enumerate(sel):
        for c, idx in enumerate((i, j)):
            ax = axes[r, c]
            with Image.open(files[idx]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{files[idx].parent.name}/{files[idx].name}", fontsize=6)
            ax.axis("off")
    fig.suptitle("detected duplicate pairs (left vs right)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresh", type=int, default=5)
    ap.add_argument("--val-ratio", type=float, default=0.25)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else Path("runs/_data") / root.name
    out.mkdir(parents=True, exist_ok=True)

    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).rglob("*")):
            if f.suffix.lower() in EXTS:
                files.append(f)
                labels.append(ci)
    labels = np.array(labels)
    print(f"hashing {len(files)} images x {len(VARIANTS)} variants x 2 scales ...")
    a, p, ac, pc = hash_dataset(files)

    print("comparing ...")
    pairs, bg_pairs = find_pairs(a, p, ac, pc, args.thresh)
    roots = union_groups(len(files), pairs)
    uniq = sorted(set(roots))
    sizes = np.bincount(np.searchsorted(uniq, roots))
    dup_members = int(sum(s for s in sizes if s > 1))
    cross_class = [(i, j) for i, j in pairs if labels[i] != labels[j]]

    cross, leaked_val, n_val = naive_split_leakage(labels, roots, pairs, args.val_ratio)
    leak_rate = 100 * len(leaked_val) / max(n_val, 1)

    bg_members = len({i for pr in bg_pairs for i in pr})
    bg_cross_class = [(i, j) for i, j in bg_pairs if labels[i] != labels[j]]

    lines = [
        f"# Duplicate / leakage report: {root.name}", "",
        f"threshold: Hamming <= {args.thresh} on **both** aHash and pHash, "
        f"over variants {VARIANTS}, at two scales "
        f"(full frame and centre {CENTER_FRAC:.0%} crop).", "",
        "A full-frame match with a **non**-matching centre crop means the two "
        "images share a background but hold different targets -- a compositing "
        "artefact, not a duplicate. Those are counted separately below and are "
        "**not** merged into duplicate groups.", "",
        f"- images scanned: **{len(files)}**",
        f"- duplicate pairs found: **{len(pairs)}**",
        f"- images belonging to a duplicate group: **{dup_members}** "
        f"({100 * dup_members / len(files):.1f}%)",
        f"- unique images after grouping: **{len(uniq)}**",
        f"- largest duplicate group: **{int(sizes.max())}** images",
        f"- groups with >1 member: {int((sizes > 1).sum())}",
        f"- **cross-class** duplicate pairs (same image, different labels): "
        f"**{len(cross_class)}**",
        "",
        "## Shared-background pairs (compositing artefact)",
        f"- pairs sharing a background but not a target: **{len(bg_pairs)}**",
        f"- images involved: **{bg_members}** ({100 * bg_members / len(files):.1f}%)",
        f"- of which **cross-class**: **{len(bg_cross_class)}** pairs -- the same "
        f"background plate carries targets of different types, i.e. the imagery "
        f"is at least partly composited rather than natively cropped",
        "",
        "## Leakage under a naive (non group-aware) stratified split",
        f"- val size: {n_val} (ratio {args.val_ratio})",
        f"- cross-split duplicate pairs: **{len(cross)}**",
        f"- val images having a twin in train: **{len(leaked_val)}**",
        f"- **leakage rate: {leak_rate:.2f}% of val**",
        "",
        f"verdict: {'**LEAKAGE > 5% -- val accuracy is inflated, flag in REPORT.md**' if leak_rate > 5 else 'leakage below the 5% threshold'}",
    ]
    if cross_class:
        lines += ["", "## Cross-class duplicate pairs (label-quality issue)", ""]
        for i, j in cross_class[:30]:
            lines.append(f"- `{classes[labels[i]]}/{files[i].name}`  ==  "
                         f"`{classes[labels[j]]}/{files[j].name}`")

    (out / "dedup_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "dup_groups.json").write_text(json.dumps({
        "root": str(root), "thresh": args.thresh, "variants": VARIANTS,
        "files": [str(f.relative_to(root)) for f in files],
        "group_id": [int(r) for r in roots],
        "n_pairs": len(pairs), "n_unique": len(uniq),
        "leak_rate_naive_split": leak_rate,
        "n_shared_background_pairs": len(bg_pairs),
        "n_shared_background_cross_class": len(bg_cross_class),
        "cross_class_pairs": [[str(files[i].relative_to(root)),
                               str(files[j].relative_to(root))] for i, j in cross_class],
        "shared_background_cross_class_pairs": [
            [str(files[i].relative_to(root)), str(files[j].relative_to(root))]
            for i, j in bg_cross_class],
    }, indent=2), encoding="utf-8")

    montage(files, pairs, out / "dup_examples.png")
    montage(files, bg_cross_class or bg_pairs, out / "shared_background.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    big = sizes[sizes > 1]
    if len(big):
        ax.hist(big, bins=range(2, int(big.max()) + 2), color="#a53b3b", align="left")
    ax.set_xlabel("images per duplicate group")
    ax.set_ylabel("groups")
    ax.set_title(f"{root.name}: duplicate group sizes ({len(uniq)} unique / {len(files)} total)")
    fig.tight_layout()
    fig.savefig(out / "dup_groups.png", dpi=150)

    print("\n".join(lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
