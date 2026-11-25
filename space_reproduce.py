"""
Runs the COINS'26 case-study arms over seeds on several GPUs (one job = one arm x seed),
writing results/<arm>_seed<k>.jsonl; summarize them with ``space_main.py summarize``.

    python space_reproduce.py --cache-dir CACHE --e0 'E0/*_seed{seed}/best.pt' --e2 'E2/*_seed{seed}/merged.pt' \
        --arms E1a E1b E2 E3 E4 --seeds 0 1 2 --gpus 0 1 2 3 --per-gpu 2 --out-dir results
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from spacecim.mission import ARMS, arm_missions  # noqa: E402

FIGURES = {
    "fig2": ["E1a", "E1b", "E2", "E3", "E4"],
    "fig3": ["E3_noLoRA", "E4_noLoRA", "E4_head"],
    "fig5": [f"{p}{k}Y" for p in ("P", "R", "RP", "S") for k in range(1, 5)],
    "fig6": ["D1", "D4", "RPD1", "RPD4", "SD1", "SD4", "SX1", "SX4"],
    "fig7": ["D1", "D2", "D3", "D4"],
    "fig8": ["T1", "T2", "T3", "T4"],
}


def cost(arm: str) -> float:
    """Relative cost: evaluated checkpoints, recalibration counted as 0.7 of an evaluation."""
    total = 0.0
    for spec in arm_missions(arm):
        n = 1 if spec["endpoint_only"] else len(spec["cfg"].checkpoints)
        total += n * (1.7 if spec["cfg"].recalibrate else 1.0)
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--e0", required=True)
    p.add_argument("--e2", required=True)
    p.add_argument("--arms", nargs="+", default=None, choices=sorted(ARMS))
    p.add_argument("--figures", nargs="+", default=None, choices=sorted(FIGURES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--sweep-seeds", type=int, nargs="+", default=None,
                   help="seeds for the dose / temperature sweep arms (default: --seeds)")
    p.add_argument("--figure-seeds", nargs="+", default=None, metavar="FIG=SEEDS",
                   help="seeds of a figure's arms, e.g. fig5=0,1,2,3,4 (overrides --seeds / --sweep-seeds)")
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per GPU")
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--compile", action="store_true", help="torch.compile the tile MVM (GPU)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()

    arms = list(a.arms or [])
    for f in a.figures or []:
        arms += [x for x in FIGURES[f] if x not in arms]
    if not arms:
        raise SystemExit("give --arms and/or --figures")
    os.makedirs(a.out_dir, exist_ok=True)
    sweep_seeds = a.sweep_seeds if a.sweep_seeds is not None else a.seeds
    per_arm = {}
    for item in a.figure_seeds or []:
        fig, _, seeds = item.partition("=")
        for arm in FIGURES[fig]:
            per_arm[arm] = [int(x) for x in seeds.split(",")]
    jobs = sorted({(arm, seed) for arm in arms
                   for seed in per_arm.get(arm, sweep_seeds if ARMS[arm]["sweep"] else a.seeds)})
    jobs = sorted(jobs, key=lambda j: -cost(j[0]))
    jobs = [j for j in jobs if not os.path.exists(os.path.join(a.out_dir, f"{j[0]}_seed{j[1]}.done"))]
    slots = [g for g in a.gpus for _ in range(a.per_gpu)]
    running = {}
    print(f"{len(jobs)} jobs on {len(slots)} slots", flush=True)
    while jobs or running:
        for slot in slots:
            if not jobs:
                break
            if sum(1 for s, _ in running.values() if s == slot) >= a.per_gpu:
                continue
            arm, seed = jobs.pop(0)
            stem = os.path.join(a.out_dir, f"{arm}_seed{seed}")
            cmd = [a.python, "-u", "-W", "ignore", os.path.join(HERE, "space_main.py"), "mission",
                   "--cache-dir", a.cache_dir, "--arms", arm, "--seeds", str(seed), "--e0", a.e0, "--e2", a.e2,
                   "--out", stem + ".jsonl"]
            if a.eval_batch:
                cmd += ["--eval-batch", str(a.eval_batch)]
            if a.compile:
                cmd += ["--compile"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot)
            if os.path.exists(stem + ".jsonl"):
                os.remove(stem + ".jsonl")
            proc = subprocess.Popen(cmd, env=env, stdout=open(stem + ".log", "w"), stderr=subprocess.STDOUT)
            running[proc.pid] = (slot, (proc, arm, seed, stem, time.time()))
            print(f"start {arm} seed {seed} on GPU {slot}", flush=True)
        for pid, (slot, (proc, arm, seed, stem, t0)) in list(running.items()):
            if proc.poll() is not None:
                del running[pid]
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode == 0:
                    open(stem + ".done", "w").close()
                print(f"{status:>6} {arm} seed {seed} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
