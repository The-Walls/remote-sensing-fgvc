"""Dataset health check for an ImageFolder-style directory.

Reports class count, per-class sample distribution, image size distribution and
long-tail ratio -- the numbers that decide input resolution and whether
class-rebalancing is needed. Writes a bar chart PNG next to the text report.

Usage:
    python tools/inspect_data.py <root> [--out runs/_data/<name>]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan(root: Path):
    """Return {class_name: [(w, h), ...]}. Reads headers only, not pixels."""
    per_class = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        sizes = []
        for f in sorted(d.rglob("*")):
            if f.suffix.lower() in EXTS:
                with Image.open(f) as im:
                    sizes.append(im.size)
        if sizes:
            per_class[d.name] = sizes
    return per_class


def pct(values, q):
    v = sorted(values)
    return v[min(int(q * len(v)), len(v) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else Path("runs/_data") / root.name
    out.mkdir(parents=True, exist_ok=True)

    per_class = scan(root)
    if not per_class:
        raise SystemExit(f"no class subdirectories with images under {root}")

    counts = {k: len(v) for k, v in per_class.items()}
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    total = sum(counts.values())
    n_max, n_min = ranked[0][1], ranked[-1][1]

    all_sizes = [s for v in per_class.values() for s in v]
    widths = [w for w, _ in all_sizes]
    heights = [h for _, h in all_sizes]
    shorter = [min(w, h) for w, h in all_sizes]
    ratios = [max(w, h) / min(w, h) for w, h in all_sizes]

    lines = []
    add = lines.append
    add(f"# Dataset report: {root.name}")
    add(f"\npath: `{root}`\n")
    add(f"- classes: **{len(counts)}**")
    add(f"- total images: **{total}**")
    add(f"- mean / median images per class: {total / len(counts):.1f} / {pct(list(counts.values()), 0.5)}")
    add(f"- largest class: `{ranked[0][0]}` ({n_max})")
    add(f"- smallest class: `{ranked[-1][0]}` ({n_min})")
    add(f"- **long-tail ratio (max/min): {n_max / n_min:.1f}x**")
    add(f"- classes with < 30 images: {sum(1 for c in counts.values() if c < 30)}")
    add(f"- classes with < 50 images: {sum(1 for c in counts.values() if c < 50)}")

    add("\n## Image size distribution")
    add(f"- width  min/p25/p50/p75/max: {min(widths)} / {pct(widths,.25)} / {pct(widths,.5)} / {pct(widths,.75)} / {max(widths)}")
    add(f"- height min/p25/p50/p75/max: {min(heights)} / {pct(heights,.25)} / {pct(heights,.5)} / {pct(heights,.75)} / {max(heights)}")
    add(f"- shorter side p05/p50/p95: {pct(shorter,.05)} / {pct(shorter,.5)} / {pct(shorter,.95)}")
    add(f"- aspect ratio (long/short) p50/p95/max: {pct(ratios,.5):.2f} / {pct(ratios,.95):.2f} / {max(ratios):.2f}")
    add(f"- images whose shorter side < 224: {sum(1 for s in shorter if s < 224)} "
        f"({100 * sum(1 for s in shorter if s < 224) / len(shorter):.1f}%)")
    add(f"- images whose shorter side < 448: {sum(1 for s in shorter if s < 448)} "
        f"({100 * sum(1 for s in shorter if s < 448) / len(shorter):.1f}%)")

    add("\n## Per-class counts (descending)")
    add("\n| rank | class | n | share | median WxH |")
    add("|---:|---|---:|---:|---|")
    for i, (name, n) in enumerate(ranked, 1):
        w = pct([s[0] for s in per_class[name]], .5)
        h = pct([s[1] for s in per_class[name]], .5)
        add(f"| {i} | {name} | {n} | {100 * n / total:.1f}% | {w}x{h} |")

    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({
        "root": str(root), "n_classes": len(counts), "n_images": total,
        "counts": counts, "imbalance_ratio": n_max / n_min,
        "shorter_side_p05_p50_p95": [pct(shorter, .05), pct(shorter, .5), pct(shorter, .95)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    ax[0].bar(range(len(ranked)), [n for _, n in ranked], color="#3b6ea5")
    ax[0].set_xticks(range(len(ranked)))
    ax[0].set_xticklabels([k for k, _ in ranked], rotation=90, fontsize=7)
    ax[0].set_ylabel("images")
    ax[0].set_title(f"{root.name}: per-class counts (imbalance {n_max / n_min:.1f}x)")
    ax[1].hist(shorter, bins=50, color="#a55b3b")
    ax[1].axvline(224, color="k", ls="--", lw=1, label="224")
    ax[1].axvline(448, color="k", ls=":", lw=1, label="448")
    ax[1].set_xlabel("shorter side (px)")
    ax[1].set_ylabel("images")
    ax[1].set_title("image size distribution")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "distribution.png", dpi=150)

    print("\n".join(lines[:30]))
    print(f"\n-> {out / 'report.md'}\n-> {out / 'distribution.png'}")


if __name__ == "__main__":
    main()
