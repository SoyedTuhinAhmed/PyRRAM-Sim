"""
Analog compute-in-memory for space data centers: the COINS'26 case study in PyRRAM-Sim.

    build-cache   EuroSAT_MS GeoTIFFs -> array cache (needs rasterio, pyproj)
    finetune      E0: FP32 fine-tuning of Presto on EuroSAT
    lora          E2: noise-aware LoRA on an E0 checkpoint, merged for analog mapping
    mission       run one or more paper arms (E1a ... E4, P/R/RP/S ladders, D/T sweeps, ...)
    summarize     mean +- std over seeds of mission results, next to the paper's values
    plot          accuracy vs mission time and vs dose / temperature
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from spacecim import mission as M  # noqa: E402
from presto import eurosat as ES  # noqa: E402

# Every point of the paper's case study: 10-seed mean and std (aihwkit 1.1.0 runs).
PAPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "spacecim", "paper_coins26.json")
T_ORDER = {t: i for i, t in enumerate(["digital", "1s", "1h", "1d", "1wk", "5d", "1mo", "3mo", "6mo", "12mo"])}


def _key(arm, t, seu, env):
    return arm, t, float(seu or 0.0), None if env is None else round(float(env), 4)


def load_paper(path=PAPER_FILE):
    rows = json.load(open(path))["rows"]
    return {_key(r["arm"], r["t"], r["seu"], r["env"]): r for r in rows}


def _classifier():
    """The Presto model code needs einops; import it only for the commands that build a model."""
    from presto import classifier
    return classifier


def _device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve(pattern: str, seed: int) -> str:
    matches = sorted(glob.glob(pattern.format(seed=seed)))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matches {pattern.format(seed=seed)}")
    return matches[-1]


def cmd_build_cache(a):
    m = ES.build_cache(a.eurosat_root, a.split_dir, a.cache_dir, a.workers)
    print(f"cache: {m['n_images']} images, splits {m['split_sizes']}")


def _data(a, device):
    cache = ES.EuroSatCache(a.cache_dir)
    builder = ES.PrestoInputBuilder(cache, a.input_patch_size, device)
    return cache, builder


def cmd_finetune(a):
    PC = _classifier()
    device = _device(a.device)
    torch.manual_seed(a.seed)
    cache, builder = _data(a, device)
    model = PC.build_model(a.presto_weights).to(device)
    PC.train_model(model, builder, cache.indices("train", a.limit), cache.indices("val", a.limit),
                   max_epochs=a.max_epochs or 100, generator=torch.Generator().manual_seed(a.seed))
    print("test:", PC.evaluate(model, builder, cache.indices("test", a.limit)))
    torch.save(model.state_dict(), a.out)


def cmd_lora(a):
    PC = _classifier()
    device = _device(a.device)
    torch.manual_seed(a.seed)
    cache, builder = _data(a, device)
    model = PC.load_classifier(a.checkpoint).to(device)
    PC.noise_aware_lora(model, builder, cache.indices("train", a.limit), cache.indices("val", a.limit), a.seed,
                        max_epochs=a.max_epochs or 50)
    print("test:", PC.evaluate(model, builder, cache.indices("test", a.limit)))
    torch.save(model.state_dict(), a.out)


def cmd_mission(a):
    PC = _classifier()
    device = _device(a.device)
    if a.compile:
        from spacecim import tile as T
        T.use_compile(True)
    cache, builder = _data(a, device)
    test_rows = cache.indices("test", a.test_limit)
    val = cache.split_index["val"]
    calib_rows = ES.stratified_subset(val, cache.labels[val], a.calib_samples, 12345)
    loss_fn = nn.CrossEntropyLoss()

    def calib_batches(gen):
        return ES.iterate_batches(builder, calib_rows, 16, shuffle=True, generator=gen)

    def loss_on_batch(model, batch):
        return loss_fn(model(*builder.model_args(batch)), batch["labels"])

    def evaluate(model):
        return PC.evaluate(model, builder, test_rows, batch_images=a.eval_batch)

    out = open(a.out, "a")
    for seed in a.seeds:
        sources = {}
        for arm in a.arms:
            missions = M.arm_missions(arm, a.sweep_values)
            for spec in missions:
                src = spec["model"]
                if src not in sources:
                    path = _resolve(a.e0 if src == "e0" else a.e2, seed)
                    sources[src] = PC.load_classifier(path).to(device).eval()
                    ref = evaluate(sources[src])
                    rec = {"arm": f"{src.upper()}_digital", "seed": seed, "t_label": "digital", **ref}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"seed {seed} {src} digital acc={ref['accuracy']:.4f} ({path})")
                tag = f"{arm}" + (f"@{spec['seu_multiplier']:g}x" if spec["seu_multiplier"] != 1 else "")
                if spec["env_value"] is not None:
                    tag += f" env={spec['env_value']:g}"
                print(f"seed {seed} {tag}{' (endpoint)' if spec['endpoint_only'] else ''}")
                ms = M.mission_seed(seed, 0, spec["mission_index"])
                records = M.run_mission(sources[src], spec["cfg"], ms, evaluate, calib_batches, loss_on_batch,
                                        labels=spec["labels"])
                for r in records:
                    r.update({"arm": arm, "seed": seed, "seu_multiplier": spec["seu_multiplier"],
                              "env_value": spec["env_value"], "endpoint_only": spec["endpoint_only"],
                              "mission_seed": ms})
                    out.write(json.dumps(r) + "\n")
                out.flush()
    out.close()


def cmd_summarize(a):
    """Mean +- std over seeds per point, next to the paper's 10-seed mean +- std.

    z = (mean - paper) / (sigma * sqrt(1/n + 1/n_paper)), sigma = the paper's seed std (at least
    0.005, for points where every chip sits at chance level): two samples of the same chip
    distribution give |z| < 2 for about 95% of the points."""
    paper = load_paper(a.paper) if a.paper != "none" else {}
    groups, digital = defaultdict(list), {}
    for path in a.results:
        for line in open(path):
            r = json.loads(line)
            if a.arms and r["arm"].split("_digital")[0] not in a.arms and r["arm"] not in a.arms:
                continue
            key = _key(r["arm"], r["t_label"], r.get("seu_multiplier"), r.get("env_value"))
            if r["t_label"] == "digital":
                # every mission job re-evaluates its digital model: count each seed once
                digital.setdefault((key, r["seed"]), r["accuracy"])
            else:
                groups[key].append(r["accuracy"])
    for (key, _), acc in digital.items():
        groups[key].append(acc)

    rows = []
    for key, v in groups.items():
        p = paper.get(key)
        row = {"arm": key[0], "t": key[1], "seu": key[2], "env": key[3], "n": len(v),
               "mean": float(np.mean(v)), "std": float(np.std(v))}
        if p is not None:
            se = max(p["std"], 0.005) * np.sqrt(1 / len(v) + 1 / p["n"])
            row.update(paper=p["mean"], paper_std=p["std"], paper_n=p["n"], z=(row["mean"] - p["mean"]) / se)
        rows.append(row)
    rows.sort(key=lambda r: (not r["arm"].endswith("_digital"), r["arm"], r["seu"],
                             -1e9 if r["env"] is None else r["env"], T_ORDER.get(r["t"], 99)))

    def label(r):
        return r["arm"] + (f"@{r['seu']:g}x" if r["seu"] not in (0.0, 1.0) else "")

    print(f"{'arm':<16}{'t':>8}{'env':>8}{'n':>4}{'mean':>8}{'std':>8}{'paper':>8}{'p.std':>8}{'z':>7}")
    for r in rows:
        env = "" if r["env"] is None else f"{r['env']:g}"
        ref = (f"{r['paper']:>8.4f}{r['paper_std']:>8.4f}{r['z']:>+7.2f}" if "paper" in r else "")
        print(f"{label(r):<16}{r['t']:>8}{env:>8}{r['n']:>4}{r['mean']:>8.4f}{r['std']:>8.4f}{ref}")
    matched = [r for r in rows if "paper" in r and not r["arm"].endswith("_digital")]
    if matched:
        z = np.array([r["z"] for r in matched])
        d = np.array([r["mean"] - r["paper"] for r in matched])
        print(f"\n{len(matched)} points compared with the paper: mean difference {d.mean():+.4f}, "
              f"median |difference| {np.median(np.abs(d)):.4f}; |z| < 2 for {np.mean(np.abs(z) < 2):.0%}, "
              f"|z| < 3 for {np.mean(np.abs(z) < 3):.0%} (two samples of one distribution: ~95% and ~99.7%)")
        worst = sorted(matched, key=lambda r: -abs(r["z"]))[:a.worst]
        if worst:
            print("largest |z|:")
            for r in worst:
                env = "" if r["env"] is None else f" env={r['env']:g}"
                print(f"  {label(r)} {r['t']}{env}: {r['mean']:.3f} (n={r['n']}) vs paper {r['paper']:.3f} "
                      f"+- {r['paper_std']:.3f}, z {r['z']:+.2f}")
    if a.csv:
        cols = ["arm", "t", "seu", "env", "n", "mean", "std", "paper", "paper_std", "paper_n", "z"]
        with open(a.csv, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join("" if r.get(c) is None else str(r[c]) for c in cols) + "\n")
        print("wrote", a.csv)


def cmd_plot(a):
    """Accuracy vs mission time (timeline arms) and vs dose / temperature (sweep arms)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.out_dir, exist_ok=True)
    data = defaultdict(lambda: defaultdict(list))
    for path in a.results:
        for line in open(path):
            r = json.loads(line)
            if r["t_label"] == "digital" or r.get("endpoint_only"):
                continue
            series = (r["arm"], r.get("seu_multiplier", 1.0))
            sweep = M.ARMS.get(r["arm"], {}).get("sweep")
            if sweep is None:
                tech = {"P": "pcm", "R": "reram-single", "RP": "reram-pair", "S": "sonos"}.get(r["arm"][:-2])
                group = f"timeline_{tech}" if r["arm"].endswith("Y") and tech else "timeline_ladder"
                x = r["t_seconds"]
            else:
                group = ("dose_" if sweep["key"].startswith("tid") else "temperature_") + r["t_label"]
                x = r["env_value"]
            data[(group, series)][x].append(r["accuracy"])
    groups = defaultdict(list)
    for (group, series), pts in data.items():
        groups[group].append((series, pts))
    xlabels = {"timeline": "time since programming (s)", "dose": "total ionizing dose, krad(Si)",
               "temperature": "operating temperature (C)"}
    for name, curves in groups.items():
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for (arm, mult), pts in sorted(curves, key=lambda c: c[0]):
            xs = sorted(pts)
            mean = np.array([np.mean(pts[x]) for x in xs])
            std = np.array([np.std(pts[x]) for x in xs])
            lab = arm if mult in (0, 1, None) else f"{arm} {mult:g}x"
            ax.plot(xs, mean, marker="o", ms=3, label=lab)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.15)
        if not name.startswith("temperature"):
            ax.set_xscale("log")
        ax.set_xlabel(xlabels[name.split("_")[0]])
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_ylabel("top-1 accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        out = os.path.join(a.out_dir, f"{name}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--cache-dir", required=True, help="EuroSAT array cache (build-cache)")
        q.add_argument("--device", default="auto")
        q.add_argument("--input-patch-size", type=int, default=2, help="2 -> 1,024 patches per image")

    b = sub.add_parser("build-cache")
    b.add_argument("--eurosat-root", required=True, help="unpacked EuroSAT_MS archive")
    b.add_argument("--split-dir", required=True, help="folder with eurosat-{train,val,test}.txt")
    b.add_argument("--cache-dir", required=True)
    b.add_argument("--workers", type=int, default=16)
    b.set_defaults(fn=cmd_build_cache)

    f = sub.add_parser("finetune")
    common(f)
    f.add_argument("--presto-weights", required=True, help=f"Presto default_model.pt ({ES.PRESTO_WEIGHTS_URL})")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--limit", type=int, default=None, help="class-stratified subset per split (smoke test)")
    f.add_argument("--max-epochs", type=int, default=None, help="default 100, early stopping (patience 3)")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_finetune)

    l = sub.add_parser("lora")
    common(l)
    l.add_argument("--checkpoint", required=True, help="E0 checkpoint")
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--limit", type=int, default=None)
    l.add_argument("--max-epochs", type=int, default=None, help="default 50, early stopping (patience 3)")
    l.add_argument("--out", required=True)
    l.set_defaults(fn=cmd_lora)

    m = sub.add_parser("mission")
    common(m)
    m.add_argument("--arms", nargs="+", required=True, choices=sorted(M.ARMS))
    m.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    m.add_argument("--e0", required=True, help="E0 checkpoint path, {seed} is substituted, globs allowed")
    m.add_argument("--e2", required=True, help="E2 (LoRA merged) checkpoint path, {seed} substituted")
    m.add_argument("--sweep-values", type=float, nargs="+", default=None, help="override the dose / temperature sweep")
    m.add_argument("--test-limit", type=int, default=None)
    m.add_argument("--eval-batch", type=int, default=None, help="images per evaluation batch (throughput only)")
    m.add_argument("--compile", action="store_true", help="fuse the tile MVM with torch.compile (GPU, ~2x faster)")
    m.add_argument("--calib-samples", type=int, default=256, help="stratified to 25 per class = 250 images")
    m.add_argument("--out", required=True, help="JSONL file (appended)")
    m.set_defaults(fn=cmd_mission)

    s = sub.add_parser("summarize", help="mean +- std per point, next to the paper's values")
    s.add_argument("results", nargs="+")
    s.add_argument("--arms", nargs="+", default=None, help="only these arms")
    s.add_argument("--paper", default=PAPER_FILE, help="reference values ('none' to leave them out)")
    s.add_argument("--worst", type=int, default=10, help="list the points with the largest |z|")
    s.add_argument("--csv", default=None, help="also write the table as CSV")
    s.set_defaults(fn=cmd_summarize)

    g = sub.add_parser("plot", help="accuracy curves from mission results (needs matplotlib)")
    g.add_argument("results", nargs="+")
    g.add_argument("--out-dir", default="figures")
    g.set_defaults(fn=cmd_plot)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
