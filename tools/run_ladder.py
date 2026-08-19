"""Queue ladder configs serially and keep going when one fails.

Serial by design: concurrent runs exhaust the Windows pagefile that backs
DataLoader shared memory (error 1455). Completed rungs are skipped and partial
runs holding latest.pt resume with optimizer, scheduler, and RNG intact.

    python tools/run_ladder.py configs/fgscr42/L*.yaml [--seeds 0] [--force]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    jobs = []
    for pattern in args.configs:
        for c in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if c.exists() and c.name != "base.yaml":
                jobs += [(c, s) for s in args.seeds]

    print(f"queued {len(jobs)} run(s)\n")
    done, failed, skipped = [], [], []
    for i, (cfg_path, seed) in enumerate(jobs, 1):
        cfg = config.load(cfg_path)
        out = ROOT / cfg["out_dir"] / cfg["exp_name"] / f"seed{seed}"
        if (out / "metrics.json").exists() and not args.force:
            print(f"[{i}/{len(jobs)}] SKIP {cfg['exp_name']} seed{seed} (done)")
            skipped.append(cfg["exp_name"])
            continue

        print(f"[{i}/{len(jobs)}] RUN  {cfg['exp_name']} seed{seed}", flush=True)
        t0 = time.time()
        command = [sys.executable, str(ROOT / "tools" / "train.py"), str(cfg_path),
                   "--set", f"seed={seed}"]
        if (out / "latest.pt").exists() and not args.force:
            command.append("--resume")
        p = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        captured = []
        for line in p.stdout:
            captured.append(line)
            print("        " + line.rstrip(), flush=True)
        p.wait()
        dt = (time.time() - t0) / 60
        if p.returncode == 0:
            tail = [line.strip() for line in captured if line.startswith("best:")]
            print(f"        OK  {dt:.1f}min  {tail[-1] if tail else ''}", flush=True)
            done.append(cfg["exp_name"])
        else:
            err = [line.strip() for line in captured if line.strip()]
            print(f"        FAIL {dt:.1f}min  {err[-1] if err else 'no output'}",
                  flush=True)
            (out.parent / "FAILED.log").parent.mkdir(parents=True, exist_ok=True)
            (out.parent / "FAILED.log").write_text("".join(captured),
                                                   encoding="utf-8")
            failed.append(cfg["exp_name"])

    print(f"\ndone={len(done)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        print("failed rungs:", ", ".join(failed))


if __name__ == "__main__":
    main()
