"""Train / eval loops with bf16 autocast and OOM-triggered batch-size backoff."""
import math
import time

import numpy as np
import torch

from . import metrics


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def train_one_epoch(model, loader, crit, opt, dev, epoch, cfg, sched_state, log):
    model.train()
    amp = getattr(torch, cfg["train"]["amp_dtype"])
    tot, n, t0 = 0.0, 0, time.time()
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        lr_scale = cosine_lr(sched_state["step"], sched_state["total"],
                             sched_state["warmup"], 1.0)
        for g, base in zip(opt.param_groups, sched_state["base_lrs"]):
            g["lr"] = base * lr_scale
        with torch.autocast("cuda", dtype=amp):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["train"]["grad_clip"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        sched_state["step"] += 1
        tot += loss.item() * y.numel()
        n += y.numel()
    return {"train_loss": tot / max(n, 1), "sec": time.time() - t0,
            "lr": opt.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(model, loader, crit, dev, cfg, n_classes, keep_preds=False):
    model.eval()
    amp = getattr(torch, cfg["train"]["amp_dtype"])
    ys, ps, tot, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=amp):
            logits = model(x)
            loss = crit(logits, y)
        tot += loss.item() * y.numel()
        n += y.numel()
        ys.append(y.cpu().numpy())
        ps.append(logits.float().argmax(1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    cm = metrics.confusion(y_true, y_pred, n_classes)
    out = metrics.summarize(cm)
    out["val_loss"] = tot / max(n, 1)
    return out, cm, (y_true, y_pred) if keep_preds else None


def run_with_oom_backoff(fn, cfg, log, min_bs=2):
    """Call fn(batch_size); on CUDA OOM halve the batch size and retry.

    Reported explicitly because it silently changes the effective recipe -- an
    L-rung that OOM'd is no longer comparing like with like.
    """
    bs = cfg["data"]["batch_size"]
    events = []
    while True:
        try:
            return fn(bs), bs, events
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= min_bs:
                raise
            new = max(min_bs, bs // 2)
            msg = f"[OOM] batch_size {bs} -> {new}"
            log(msg)
            events.append(msg)
            bs = new
