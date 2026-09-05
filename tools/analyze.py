"""Scan runs/ and produce the comparison table, the val->test selection gap,
a converged-window check, a per-seed appendix, confusion matrices and grids
of misclassified test examples.

    python tools/analyze.py [--runs runs] [--baseline L0] [--out runs/_analysis]
                            [--no-error-grids]

Every headline number is TEST at the val-selected epoch, mean ± std over seeds.
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

LADDER = ["L0", "L0_naive_split", "L1_320", "L1_448", "L2", "L2z_zoom_only", "L2r_rot_reflect",
          "L3_compact_bilinear", "L4_cbam", "L5_cb_ce",
          "L6a_convnext_tiny", "L6b_vit_base_p16"]
PARENT = {
    "L0_naive_split": "L0",
    "L1_320": "L0",
    "L1_448": "L0",
    "L2": "L1_448",
    "L2z_zoom_only": "L1_448",
    "L2r_rot_reflect": "L1_448",
    "L3_compact_bilinear": "L2",
    "L4_cbam": "L2",
    "L5_cb_ce": "L2",
    "L6a_convnext_tiny": "L2",
    "L6b_vit_base_p16": "L2",
}
KEYS = [("overall_top1", "overall top-1"),
        ("mean_per_class_top1", "mean-per-class"),
        ("mean_per_class_top1_support_ge5", "MPC (support≥5)"),
        ("macro_f1", "macro-F1")]
FINAL_K = 10   # epochs in the converged window


def order_key(name):
    return (LADDER.index(name), name) if name in LADDER else (len(LADDER), name)


def collect(runs: Path):
    out = {}
    for mj in sorted(runs.glob("*/seed*/metrics.json")):
        if "_smoke" in str(mj):
            continue
        r = json.loads(mj.read_text(encoding="utf-8"))
        if "test_at_best" not in r:
            print(f"[analyze] skipping {mj}: no test split (v1 layout)")
            continue
        out.setdefault(r["exp_name"], []).append(r)
    for rs in out.values():
        rs.sort(key=lambda r: r["seed"])
    return out


def agg(values):
    v = np.array(list(values), dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def fmt(mean, std, n, scale=100):
    return f"{mean * scale:.2f} ± {std * scale:.2f}" if n > 1 else f"{mean * scale:.2f}"


def final_window(r, key, k=FINAL_K):
    """Mean of history[key] over the last k epochs of one run."""
    return float(np.mean([h[key] for h in r["history"][-k:]]))


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
    """Re-run the best checkpoint over TEST and lay out k misclassified samples."""
    ck = run_dir / "best.pt"
    if not ck.exists():
        return None
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    cfg, classes = blob["cfg"], blob["classes"]
    cfg["data"]["workers"] = 0
    _, _, test_loader, meta = data.build(cfg)
    model, _ = models.build(cfg, meta["n_classes"])
    model.load_state_dict(blob["model"])
    model.cuda().eval()

    mean = torch.tensor(cfg["data"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["data"]["std"]).view(3, 1, 1)
    wrong = []
    with torch.no_grad():
        for x, y in test_loader:
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
    fig.suptitle(f"{run_dir.parent.name}: misclassified test samples", fontsize=11)
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
        raise SystemExit(f"no v2 metrics.json under {runs}")
    names = sorted(by_exp, key=order_key)

    def test_stat(rs, key):
        return agg(r["test_at_best"][key] for r in rs)

    base = by_exp.get(args.baseline)
    base_top1 = test_stat(base, "overall_top1")[0] if base else None
    base_mpc = test_stat(base, "mean_per_class_top1")[0] if base else None
    d0 = by_exp[names[0]][0]["hparams"]["data"]

    lines = ["# Experiment ladder", "",
             f"Protocol: group-aware train/val/test split "
             f"({1 - d0['val_ratio'] - d0['test_ratio']:.0%}/{d0['val_ratio']:.0%}/"
             f"{d0['test_ratio']:.0%} of unique image groups per class); val "
             f"mean-per-class top-1 picks the epoch, **test at that epoch is what is "
             f"reported**. `±` is the std over seeds (ddof=1); a rung with one seed "
             f"shows the mean only. The seed controls the split as well as the "
             f"initialisation, so the spread includes split variance.", "",
             f"baseline = `{args.baseline}`; Δ columns are absolute percentage points "
             f"against it.", "",
             "| id | backbone | img | aug | head | loss | seeds | overall top-1 | Δ | "
             "mean-per-class | Δ | MPC (support≥5) | macro-F1 | params (M) | "
             "s/epoch | Δ vs parent |",
             "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
             "---:|---:|---|"]

    for name in names:
        rs = by_exp[name]
        h = rs[0]["hparams"]
        stats = {k: test_stat(rs, k) for k, _ in KEYS}
        n = stats["overall_top1"][2]
        d_t = (f"{(stats['overall_top1'][0] - base_top1) * 100:+.2f}"
               if base_top1 is not None else "-")
        d_m = (f"{(stats['mean_per_class_top1'][0] - base_mpc) * 100:+.2f}"
               if base_mpc is not None else "-")

        parent = PARENT.get(name)
        parent_cfg = by_exp[parent][0]["hparams"] if parent in by_exp else None
        dcfg = config.diff(parent_cfg, h) if parent_cfg else {}
        dcfg.pop("exp_name", None)
        dcfg.pop("seed", None)
        diff_s = ", ".join(f"`{k}`: {a}→{b}" for k, (a, b) in dcfg.items()) or "—"

        cells = " | ".join(fmt(*stats[k]) for k, _ in KEYS)
        c = cells.split(" | ")
        lines.append(
            f"| {name} | {h['model']['name']} | {h['data']['img_size']} | "
            f"{h['data']['aug']} | {h['model']['head']} | {h['loss']['name']} | {n} | "
            f"{c[0]} | {d_t} | {c[1]} | {d_m} | {c[2]} | {c[3]} | "
            f"{rs[0]['params_M']:.2f} | "
            f"{np.mean([r['sec_per_epoch'] for r in rs]):.1f} | {diff_s} |")

    # ---- split sizes -------------------------------------------------------
    ref = by_exp.get(args.baseline) or by_exp[names[0]]
    lines += ["", "## Split per seed (group-aware rungs share it)", "",
              "| seed | train | val | test | classes absent from val | absent from test |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in ref:
        m = r["data_meta"]
        lines.append(f"| {r['seed']} | {m['n_train']} | {m['n_val']} | {m['n_test']} | "
                     f"{m['n_classes_absent_from_val']} | {m['n_classes_absent_from_test']} |")

    # ---- selection gap: val@best vs test@best -------------------------------
    lines += ["", "## Selection gap: val at its best epoch vs test at that epoch", "",
              "val@best is the maximum of a noisy sequence and is therefore optimistic; "
              "the gap to test@best is the size of that optimism. It is what a "
              "val-only protocol would have over-reported by.", "",
              "| id | best epochs | val MPC @best | test MPC @best | gap | "
              "val top-1 @best | test top-1 @best | gap |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rs = by_exp[name]
        eps = "/".join(str(r["best_epoch"]) for r in rs)
        vm = agg(r["val_at_best"]["mean_per_class_top1"] for r in rs)[0] * 100
        tm = agg(r["test_at_best"]["mean_per_class_top1"] for r in rs)[0] * 100
        vt = agg(r["val_at_best"]["overall_top1"] for r in rs)[0] * 100
        tt = agg(r["test_at_best"]["overall_top1"] for r in rs)[0] * 100
        lines.append(f"| {name} | {eps} | {vm:.2f} | {tm:.2f} | {vm - tm:+.2f} | "
                     f"{vt:.2f} | {tt:.2f} | {vt - tt:+.2f} |")

    # ---- converged-window check ----------------------------------------------
    lines += ["", f"## Converged-window check (test, mean of last {FINAL_K} epochs)", "",
              "If test@best and the converged test value disagree by more than the "
              "seed spread, the val-selected epoch is not representative of the "
              "trained model.", "",
              "| id | test MPC @best | test MPC converged | Δ | test top-1 @best | "
              "test top-1 converged | Δ |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rs = by_exp[name]
        tm = agg(r["test_at_best"]["mean_per_class_top1"] for r in rs)[0] * 100
        cm_ = agg(final_window(r, "test_mean_per_class_top1") for r in rs)[0] * 100
        tt = agg(r["test_at_best"]["overall_top1"] for r in rs)[0] * 100
        ct = agg(final_window(r, "test_overall_top1") for r in rs)[0] * 100
        lines.append(f"| {name} | {tm:.2f} | {cm_:.2f} | {tm - cm_:+.2f} | "
                     f"{tt:.2f} | {ct:.2f} | {tt - ct:+.2f} |")

    ooms = [(n, r["oom_events"]) for n in names for r in by_exp[n] if r["oom_events"]]
    if ooms:
        lines += ["", "## OOM backoffs (recipe no longer strictly comparable)", ""]
        lines += [f"- `{n}`: {e}" for n, e in ooms]

    # ---- per-seed appendix ----------------------------------------------------
    lines += ["", "## Per-seed results (test at the val-selected epoch)", "",
              "| id | seed | best epoch | overall top-1 | mean-per-class | "
              "MPC (support≥5) | macro-F1 | val MPC @best | commit |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name in names:
        for r in by_exp[name]:
            t = r["test_at_best"]
            lines.append(
                f"| {name} | {r['seed']} | {r['best_epoch']} | "
                f"{t['overall_top1'] * 100:.2f} | {t['mean_per_class_top1'] * 100:.2f} | "
                f"{t['mean_per_class_top1_support_ge5'] * 100:.2f} | "
                f"{t['macro_f1'] * 100:.2f} | "
                f"{r['val_at_best']['mean_per_class_top1'] * 100:.2f} | "
                f"{r.get('git_commit') or '-'} |")

    # ---- confusions (test, seed 0) ---------------------------------------------
    lines += ["", "## Most confusable class pairs on test (seed 0, top-15 per run)", ""]
    for name in names:
        r = by_exp[name][0]
        rd = runs / name / f"seed{r['seed']}"
        cm_path = rd / "confusion_matrix_test.npy"
        if not cm_path.exists():
            continue
        cm = np.load(cm_path)
        classes = r["data_meta"]["classes"]
        confusion_png(cm, classes, f"{name} test (seed {r['seed']})",
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
