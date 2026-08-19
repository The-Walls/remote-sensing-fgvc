"""Downscale an ImageFolder tree into a JPEG cache so training is not I/O bound.

FGSCR-42 ships as 8.4 GB of uncompressed BMP; at 60 epochs that is ~500 GB of
reads per run. Images are resized so the shorter side is at most `--max-short`
(default 512, i.e. above the largest input size in the sweep, so no experiment
sees an upsampled image it would not have seen anyway) and re-encoded as JPEG.
Images already smaller are copied through untouched.
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-short", type=int, default=512)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = [f for f in src.rglob("*") if f.suffix.lower() in EXTS]
    n_resized = n_copied = 0

    for i, f in enumerate(files):
        out = dst / f.relative_to(src).with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(f) as im:
            im = im.convert("RGB")
            w, h = im.size
            short = min(w, h)
            if short > args.max_short:
                s = args.max_short / short
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                               Image.LANCZOS)
                n_resized += 1
            else:
                n_copied += 1
            im.save(out, "JPEG", quality=args.quality, subsampling=0)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    src_mb = sum(f.stat().st_size for f in files) / 1e6
    dst_mb = sum(f.stat().st_size for f in dst.rglob("*.jpg")) / 1e6
    print(f"images     : {len(files)}  (resized {n_resized}, passthrough {n_copied})")
    print(f"size       : {src_mb:.0f} MB -> {dst_mb:.0f} MB  ({dst_mb / src_mb:.1%})")
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
