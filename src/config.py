"""YAML config with single-parent inheritance and --set overrides.

A rung config names its parent via `_base_` and overrides exactly the fields the
experiment is meant to vary; the resolved dict is written next to the run so a
result is always traceable to a complete hyper-parameter set.
"""
import argparse
import copy
from pathlib import Path

import yaml


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load(path, overrides=None) -> dict:
    """Resolve `_base_` chain, then apply `key.sub=value` overrides."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = raw.pop("_base_", None)
    cfg = load((path.parent / base_ref).resolve()) if base_ref else {}
    _deep_update(cfg, raw)

    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def diff(a: dict, b: dict, prefix="") -> dict:
    """Flat {dotted_key: (a_val, b_val)} for fields that differ."""
    out = {}
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(diff(va, vb, f"{prefix}{k}."))
        elif va != vb:
            out[f"{prefix}{k}"] = (va, vb)
    return out


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], dest="overrides",
                    help="dotted overrides, e.g. --set data.img_size=448 seed=1")
    ap.add_argument("--smoke", action="store_true",
                    help="50 images per split, 2 epochs -- end-to-end shakeout")
    ap.add_argument("--resume", action="store_true",
                    help="resume exactly from runs/<exp>/seed<k>/latest.pt")
    args = ap.parse_args()
    cfg = load(args.config, args.overrides)
    cfg["_config_path"] = str(args.config)
    cfg["_smoke"] = args.smoke
    cfg["_resume"] = args.resume
    if args.smoke:
        cfg["train"]["epochs"] = 2
        cfg["train"]["warmup_epochs"] = 0
        cfg["exp_name"] = cfg["exp_name"] + "_smoke"
    return cfg


def save(cfg: dict, path):
    clean = {k: v for k, v in copy.deepcopy(cfg).items() if not k.startswith("_")}
    Path(path).write_text(yaml.safe_dump(clean, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
