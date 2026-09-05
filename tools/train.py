"""Single-experiment entry point.

    python tools/train.py configs/fgscr42/L0.yaml [--smoke] [--set seed=1 ...]

Writes runs/<exp>/seed<k>/: config.yaml, metrics.json, log.txt, curves.png,
confusion_matrix.npy, best.pt
"""
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, engine, losses, models   # noqa: E402


def _public_cfg(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _atomic_torch_save(obj, path):
    """Replace a checkpoint only after the new file is fully written."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def main():
    cfg = config.cli()
    out = Path(cfg["out_dir"]) / cfg["exp_name"] / f"seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    latest_path = out / "latest.pt"
    resume = cfg["_resume"] and latest_path.exists()
    if not cfg["_resume"] and latest_path.exists():
        os.replace(latest_path, out / "latest.previous.pt")
    logf = (out / "log.txt").open("a" if resume else "w", encoding="utf-8")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.deterministic = cfg["deterministic"]
    torch.backends.cudnn.benchmark = not cfg["deterministic"]
    dev = "cuda"

    if cfg["_resume"] and not resume:
        log(f"[resume] {latest_path} not found; starting from epoch 0")
    log(f"=== {cfg['exp_name']} seed={cfg['seed']} ===")
    log(f"config: {cfg['_config_path']}")

    train_loader, val_loader, meta = data.build(cfg, smoke=cfg["_smoke"])
    log(f"classes={meta['n_classes']} train={meta['n_train']} val={meta['n_val']} "
        f"groups={meta['n_groups']} train_imbalance={meta['train_imbalance']:.1f}x")

    def rebuild_loaders(bs):
        cfg["data"]["batch_size"] = bs
        tl, vl, _ = data.build(cfg, smoke=cfg["_smoke"])
        return tl, vl

    model, n_params = models.build(cfg, meta["n_classes"])
    model.to(dev)
    crit, cb_weights = losses.build(cfg, meta["train_class_counts"])
    crit.to(dev)
    log(f"model={cfg['model']['name']} head={cfg['model']['head']} "
        f"params={n_params / 1e6:.2f}M img_size={cfg['data']['img_size']}")

    opt = torch.optim.AdamW(models.param_groups(
        model, cfg["train"]["lr"], cfg["train"]["head_lr_mult"],
        cfg["train"]["weight_decay"]))
    sched_state = {
        "step": 0, "warmup": cfg["train"]["warmup_epochs"] * len(train_loader),
        "total": cfg["train"]["epochs"] * len(train_loader),
        "base_lrs": [g["lr"] for g in opt.param_groups],
    }

    hist, best, best_cm, oom_events = [], None, None, []
    prior_wall = 0.0
    ep = 0
    if resume:
        checkpoint = torch.load(latest_path, map_location=dev, weights_only=False)
        saved_cfg = checkpoint.get("cfg", {})
        diffs = config.diff(_public_cfg(saved_cfg), _public_cfg(cfg))
        incompatible = {k: v for k, v in diffs.items() if k != "data.batch_size"}
        if incompatible:
            changed = ", ".join(sorted(incompatible))
            raise ValueError(f"resume config differs from checkpoint: {changed}")

        saved_bs = checkpoint.get("batch_size", cfg["data"]["batch_size"])
        if saved_bs != cfg["data"]["batch_size"]:
            train_loader, val_loader = rebuild_loaders(saved_bs)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        sched_state = checkpoint["sched_state"]
        hist = checkpoint["history"]
        best = checkpoint["best"]
        best_cm = checkpoint["best_cm"]
        oom_events = checkpoint.get("oom_events", [])
        prior_wall = checkpoint.get("wall_sec", 0.0)
        ep = checkpoint["next_epoch"]
        _restore_rng(checkpoint["rng_state"])
        log(f"[resume] loaded {latest_path}: next_epoch={ep}, "
            f"batch_size={cfg['data']['batch_size']}")

    t_start = time.time()
    while ep < cfg["train"]["epochs"]:
        try:
            tr = engine.train_one_epoch(model, train_loader, crit, opt, dev, ep, cfg,
                                        sched_state, log)
            ev, cm, _ = engine.evaluate(model, val_loader, crit, dev, cfg,
                                        meta["n_classes"])
        except torch.cuda.OutOfMemoryError:
            # Halve the batch and redo this epoch. Recorded because it silently
            # changes the recipe -- a rung that OOM'd is no longer like-for-like.
            old = cfg["data"]["batch_size"]
            if old <= 2:
                raise
            new = max(2, old // 2)
            torch.cuda.empty_cache()
            msg = f"[OOM] epoch {ep}: batch_size {old} -> {new}, epoch restarted"
            log(msg)
            oom_events.append(msg)
            train_loader, val_loader = rebuild_loaders(new)
            sched_state["warmup"] = cfg["train"]["warmup_epochs"] * len(train_loader)
            sched_state["total"] = cfg["train"]["epochs"] * len(train_loader)
            # The epoch restarts, so rewind `step` to its start in the new
            # (longer) iteration space; otherwise the cosine schedule jumps
            # because `step` still counts batches of the old size.
            sched_state["step"] = ep * len(train_loader)
            continue
        row = {"epoch": ep, **tr, **{k: v for k, v in ev.items()
                                     if not k.startswith("per_class")}}
        hist.append(row)
        log(f"ep {ep:3d} | loss {tr['train_loss']:.4f} | val {ev['val_loss']:.4f} "
            f"| top1 {ev['overall_top1'] * 100:.2f} "
            f"| mpc {ev['mean_per_class_top1'] * 100:.2f} "
            f"| f1 {ev['macro_f1'] * 100:.2f} | {tr['sec']:.1f}s")

        if best is None or ev["mean_per_class_top1"] > best["mean_per_class_top1"]:
            best, best_cm = {**ev, "epoch": ep}, cm
            _atomic_torch_save({"model": model.state_dict(), "cfg": cfg,
                                "classes": meta["classes"]}, out / "best.pt")

        next_epoch = ep + 1
        elapsed_wall = prior_wall + time.time() - t_start
        _atomic_torch_save({
            "version": 1,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "sched_state": sched_state,
            "next_epoch": next_epoch,
            "history": hist,
            "best": best,
            "best_cm": best_cm,
            "oom_events": oom_events,
            "batch_size": cfg["data"]["batch_size"],
            "wall_sec": elapsed_wall,
            "rng_state": _rng_state(),
            "cfg": cfg,
            "classes": meta["classes"],
        }, latest_path)

        if ep == 0 and ev["overall_top1"] > 0.95 and not cfg["_smoke"]:
            log("!! val top1 > 95% at epoch 0 -- stopping per standing instruction; "
                "this indicates the split is still leaking.")
            ep = next_epoch
            break
        ep = next_epoch

    wall = prior_wall + time.time() - t_start
    result = {
        "exp_name": cfg["exp_name"], "seed": cfg["seed"], "config": cfg["_config_path"],
        "best_epoch": best["epoch"],
        "overall_top1": best["overall_top1"],
        "mean_per_class_top1": best["mean_per_class_top1"],
        "mean_per_class_top1_support_ge5": best["mean_per_class_top1_support_ge5"],
        "n_classes_support_ge5": best["n_classes_support_ge5"],
        "macro_f1": best["macro_f1"],
        "params_M": n_params / 1e6,
        "sec_per_epoch": float(np.mean([h["sec"] for h in hist])),
        "wall_sec": wall,
        "epochs_run": len(hist),
        "img_size": cfg["data"]["img_size"],
        "native_short_side_p95": cfg["data"]["native_short_side_p95"],
        "upsampled": bool(cfg["data"]["native_short_side_p95"] and
                          cfg["data"]["img_size"] > cfg["data"]["native_short_side_p95"]),
        "oom_events": oom_events,
        "final_batch_size": cfg["data"]["batch_size"],
        "deterministic": cfg["deterministic"],
        "cb_weights": cb_weights,
        "data_meta": meta,
        "per_class_top1": best["per_class_top1"],
        "per_class_support": best["per_class_support"],
        "history": hist,
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0)},
        "hparams": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(out / "confusion_matrix.npy", best_cm)
    config.save(cfg, out / "config.yaml")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], label="train")
    ax[0].plot([h["epoch"] for h in hist], [h["val_loss"] for h in hist], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot([h["epoch"] for h in hist],
               [h["overall_top1"] * 100 for h in hist], label="overall top-1")
    ax[1].plot([h["epoch"] for h in hist],
               [h["mean_per_class_top1"] * 100 for h in hist], label="mean-per-class")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("%"); ax[1].legend()
    fig.suptitle(f"{cfg['exp_name']} seed{cfg['seed']}")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=140)

    log(f"best: top1={best['overall_top1'] * 100:.2f} "
        f"mpc={best['mean_per_class_top1'] * 100:.2f} "
        f"f1={best['macro_f1'] * 100:.2f} @ep{best['epoch']} | wall {wall / 60:.1f}min")
    log(f"-> {out}")
    logf.close()


if __name__ == "__main__":
    main()
