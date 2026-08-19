"""Scan runs/ and produce the comparison table, confusion matrices, the most
confusable class pairs, and a grid of misclassified examples.

    python tools/analyze.py [--runs runs] [--baseline L0] [--out runs/_analysis]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, metrics, models   # noqa: E402

LADDER = ["L0", "L0_naive_split", "L1_224", "L1_320", "L1_448", "L2",
          "L3_compact_bilinear", "L4_cbam", "L5_cb_ce",
          "L6a_convnext_tiny", "L6b_vit_base_p16"]
PARENT = {
    "L0_naive_split": "L0",
    "L1_224": "L0",
    "L1_320": "L0",
    "L1_448": "L0",
    "L2": "L1_448",
    "L3_compact_bilinear": "L2",
    "L4_cbam": "L2",
    "L5_cb_ce": "L2",
    "L6a_convnext_tiny": "L2",
    "L6b_vit_base_p16": "L2",
}


def order_key(name):
    return (LADDER.index(name), name) if name in LADDER else (len(LADDER), name)


def collect(runs: Path):
    out = {}
    for mj in sorted(runs.glob("*/seed*/metrics.json")):
        if "_smoke" in str(mj):
            continue
        r = json.loads(mj.read_text(encoding="utf-8"))
        out.setdefault(r["exp_name"], []).append(r)
    return out


def agg(rs, key):
    v = np.array([r[key] for r in rs], dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def fmt(mean, std, n, scale=100):
    if n > 1:
        return f"{mean * scale:.2f} ± {std * scale:.2f}"
    return f"{mean * scale:.2f}"


def confusion_png(cm, classes, title, path):
    n = len(classes)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.32), max(7, n * 0.30)))
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_grid(run_dir: Path, out_png: Path, k=12):
    """Re-run the best checkpoint over val and lay out k misclassified samples."""
    ck = run_dir / "best.pt"
    if not ck.exists():
        return None
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg, classes = blob["cfg"], blob["classes"]
    cfg["data"]["workers"] = 0
    _, val_loader, meta = data.build(cfg)
    model, _ = models.build(cfg, meta["n_classes"])
    model.load_state_dict(blob["model"])
    model.cuda().eval()

    mean = torch.tensor(cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["data"]["std"]).view(3, 1, 1)
    wrong = []
    with torch.no_grad():
        for x, y in val_loader:
            with torch.autocast("cuda", dtype=getattr(torch, cfg["train"]["amp_dtype"])):
                p = model(x.cuda(non_blocking=True)).float().argmax(1).cpu()
            for i in (p != y).nonzero(as_tuple=True)[0].tolist():
                wrong.append((x[i] * std + mean, int(y[i]), int(p[i])))
                if len(wrong) >= k:
                    break
            if len(wrong) >= k:
                break
    if not wrong:
        return 0

    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (img, yt, yp) in zip(np.ravel(axes), wrong):
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(f"GT:  {classes[yt]}\nPred: {classes[yp]}", fontsize=7, color="#a11")
        ax.axis("off")
    fig.suptitle(f"{run_dir.parent.name}: misclassified val samples", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return len(wrong)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline", default="L0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-error-grids", action="store_true")
    args = ap.parse_args()

    runs = Path(args.runs)
    out = Path(args.out) if args.out else runs / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    by_exp = collect(runs)
    if not by_exp:
        raise SystemExit(f"no metrics.json under {runs}")
    names = sorted(by_exp, key=order_key)
    base = by_exp.get(args.baseline)
    base_top1 = agg(base, "overall_top1")[0] if base else None
    base_mpc = agg(base, "mean_per_class_top1")[0] if base else None

    lines = ["# Experiment ladder", "",
             f"baseline = `{args.baseline}`; Δ columns are absolute percentage "
             f"points against it. `± ` appears once a rung has >1 seed.", "",
             "| id | backbone | img | aug | head | loss | seeds | overall top-1 | Δ | "
             "mean-per-class | Δ | MPC (support≥5) | macro-F1 | params (M) | "
             "s/epoch | Δ vs parent |",
             "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
             "---:|---:|---|"]

    for name in names:
        rs = by_exp[name]
        h = rs[0]["hparams"]
        t_m, t_s, n = agg(rs, "overall_top1")
        m_m, m_s, _ = agg(rs, "mean_per_class_top1")
        m5_m, m5_s, _ = agg(rs, "mean_per_class_top1_support_ge5")
        f_m, f_s, _ = agg(rs, "macro_f1")
        d_t = f"{(t_m - base_top1) * 100:+.2f}" if base_top1 is not None else "-"
        d_m = f"{(m_m - base_mpc) * 100:+.2f}" if base_mpc is not None else "-"

        parent = PARENT.get(name)
        parent_cfg = by_exp[parent][0]["hparams"] if parent in by_exp else None
        dcfg = config.diff(parent_cfg, h) if parent_cfg else {}
        dcfg.pop("exp_name", None)
        diff_s = ", ".join(f"`{k}`: {a}→{b}" for k, (a, b) in dcfg.items()) or "—"

        img = h["data"]["img_size"]
        up = " ↑up" if rs[0].get("upsampled") else ""
        lines.append(
            f"| {name} | {h['model']['name']} | {img}{up} | {h['data']['aug']} | "
            f"{h['model']['head']} | {h['loss']['name']} | {n} | "
            f"{fmt(t_m, t_s, n)} | {d_t} | {fmt(m_m, m_s, n)} | {d_m} | "
            f"{fmt(m5_m, m5_s, n)} | {fmt(f_m, f_s, n)} | "
            f"{rs[0]['params_M']:.2f} | "
            f"{np.mean([r['sec_per_epoch'] for r in rs]):.1f} | {diff_s} |")

    ooms = [(n, r["oom_events"]) for n in names for r in by_exp[n] if r["oom_events"]]
    if ooms:
        lines += ["", "## OOM backoffs (recipe no longer strictly comparable)", ""]
        lines += [f"- `{n}`: {e}" for n, e in ooms]

    lines += ["", "## Most confusable class pairs (top-15 per run)", ""]
    for name in names:
        r = by_exp[name][0]
        rd = runs / name / f"seed{r['seed']}"
        cm_path = rd / "confusion_matrix.npy"
        if not cm_path.exists():
            continue
        cm = np.load(cm_path)
        classes = r["data_meta"]["classes"]
        confusion_png(cm, classes, f"{name} (seed {r['seed']})",
                      out / f"confusion_{name}.png")
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| true | predicted | n | true support |")
        lines.append("|---|---|---:|---:|")
        for c in metrics.top_confusions(cm, classes, 15):
            lines.append(f"| {c['true']} | {c['pred']} | {c['count']} | "
                         f"{c['true_support']} |")
        lines.append("")
        if not args.no_error_grids:
            got = error_grid(rd, out / f"errors_{name}.png")
            if got:
                lines.append(f"![errors](errors_{name}.png)")
                lines.append("")

    (out / "TABLE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\n-> {out / 'TABLE.md'}")


if __name__ == "__main__":
    main()
