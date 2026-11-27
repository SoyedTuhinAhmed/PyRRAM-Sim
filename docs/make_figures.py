"""
Draws the figures of the documentation from the data files in docs/data/.

    pip install matplotlib numpy
    python docs/make_figures.py          # writes docs/figures/*.png

Colors are the same in every figure: blue = PyRRAM-Sim, orange = the paper.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")

# palette: categorical slots in fixed order (blue, orange, aqua), validated for color-vision deficiency
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
GRAY = "#9a9a9a"            # de-emphasized context series
INK, INK2 = "#191919", "#666666"
GRID, AXIS = "#e7e7e7", "#c2c2c2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100, "savefig.dpi": 200,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def save(fig, name, title=None, subtitle=None):
    if title:
        fig.suptitle(title, x=0.012, y=0.995, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    if subtitle:
        fig.text(0.012, 0.935 if title else 0.99, subtitle, ha="left", va="top", fontsize=10, color=INK2)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def dots(ax, x, y, color, size=7, **kw):
    """Markers >= 8 px with a 2 px white ring, so they stay legible over lines and bands."""
    return ax.plot(x, y, "o", ms=size, mfc=color, mec="white", mew=1.6, color=color, zorder=4, **kw)


# --------------------------------------------------------------------------- space data centers
def load_sdc():
    rows = list(csv.DictReader(open(os.path.join(DATA, "sdc_reproduction.csv"))))
    for r in rows:
        for k in ("seu", "env", "n", "mean", "std", "paper", "paper_std", "paper_n", "z"):
            r[k] = float(r[k]) if r[k] not in ("", None) else None
    return rows


T_SECONDS = {"1s": 1, "1h": 3600, "1d": 86400, "1wk": 604800, "1mo": 2.592e6, "3mo": 7.776e6,
             "6mo": 1.5552e7, "12mo": 3.1536e7}
T_TICKS = (["1 s", "1 hour", "1 day", "1 year"], [1, 3600, 86400, 3.1536e7])


def sdc_agreement(rows):
    pts = [r for r in rows if r["paper"] is not None and not r["config"].endswith("_digital")]
    inside = [r for r in pts if abs(r["z"]) < 2]
    outside = [r for r in pts if abs(r["z"]) >= 2]
    share = round(100 * len(inside) / len(pts))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 100], [0, 100], color=AXIS, lw=1, zorder=1, label="exactly the paper's value")
    ax.plot([100 * r["paper"] for r in inside], [100 * r["mean"] for r in inside], "o", ms=5.5, mfc=BLUE,
            mec="white", mew=0.8, alpha=0.85, zorder=3, label=f"within seed-to-seed variation ({share}%)")
    ax.plot([100 * r["paper"] for r in outside], [100 * r["mean"] for r in outside], "o", ms=6.5, mfc="white",
            mec=INK, mew=1.2, zorder=4, label=f"outside it ({100 - share}%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("accuracy reported in the paper (%)")
    ax.set_ylabel("accuracy simulated by PyRRAM-Sim (%)")
    ax.legend(loc="upper left", handletextpad=0.4)
    ax.text(99, 3, f"{100 - share}% outside is what chance alone gives\nwhen two sets of random chips are compared",
            fontsize=9, color=INK2, ha="right", va="bottom")
    save(fig, "sdc_agreement.png", f"{share}% of the paper's {len(pts)} results are reproduced",
         "Each dot is one number from the paper (one configuration, time, dose or temperature)")


def sdc_band(ax, xs, paper, paper_std, ours, log=True):
    xs, paper, paper_std, ours = map(np.asarray, (xs, paper, paper_std, ours))
    ax.fill_between(xs, paper - paper_std, paper + paper_std, color=ORANGE, alpha=0.16, lw=0, zorder=1)
    ax.plot(xs, paper, color=ORANGE, zorder=2)
    ax.plot(xs, ours, color=BLUE, lw=1.4, zorder=3, alpha=0.6)
    dots(ax, xs, ours, BLUE)
    if log:
        ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)


def legend_paper_ours(fig, ours_text, y=0.885):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], marker="o", color=BLUE, mfc=BLUE, mec="white", ms=7, lw=1.4),
               (Patch(color=ORANGE, alpha=0.16, lw=0), Line2D([], [], color=ORANGE))]
    fig.legend(handles, [ours_text, "Paper: average of 10 chips, and its chip-to-chip spread"],
               loc="upper left", bbox_to_anchor=(0.006, y), ncol=2, handlelength=2.2, columnspacing=1.6)


def sdc_mission(rows):
    by = {(r["config"], r["t"], r["seu"]): r for r in rows if r["env"] is None}
    panels = [("E1b", "No protection"), ("E3_noLoRA", "Drift compensation"),
              ("E4", "All three mitigations")]
    labels = ["1s", "1h", "1d", "1wk", "1mo", "3mo", "6mo", "12mo"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for ax, (cfg, name) in zip(axes, panels):
        pts = [by[(cfg, t, 1.0)] for t in labels]
        sdc_band(ax, [T_SECONDS[t] for t in labels], [100 * p["paper"] for p in pts],
                 [100 * p["paper_std"] for p in pts], [100 * p["mean"] for p in pts])
        ax.set_title(name, fontsize=10.5)
        ax.set_xticks(T_TICKS[1])
        ax.set_xticklabels(T_TICKS[0], fontsize=9)
        ax.minorticks_off()
    axes[0].set_ylabel("accuracy (%)")
    legend_paper_ours(fig, "PyRRAM-Sim: average of 10 simulated chips", y=0.87)
    fig.subplots_adjust(top=0.66, wspace=0.12)
    axes[1].set_xlabel("time since the chip was programmed")
    save(fig, "sdc_mission.png", "A PCM chip over one year in orbit: PyRRAM-Sim vs. the paper",
         "Mitigations: drift compensation, noise-aware LoRA training, and recalibration in orbit")


def sdc_technologies(rows):
    by = {(r["config"], r["t"]): r for r in rows if r["env"] is None and r["seu"] == 1.0}
    techs = [("S4Y", "SONOS"), ("P4Y", "PCM"), ("RP4Y", "ReRAM, two cells per weight"),
             ("R4Y", "ReRAM, one cell per weight")]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for i, (cfg, name) in enumerate(techs):
        r = by[(cfg, "12mo")]
        y = len(techs) - 1 - i
        ax.errorbar(100 * r["paper"], y - 0.12, xerr=100 * r["paper_std"], fmt="o", ms=7, color=ORANGE,
                    mfc=ORANGE, mec="white", mew=1.6, elinewidth=2, capsize=0, zorder=3)
        ax.errorbar(100 * r["mean"], y + 0.12, xerr=100 * r["std"], fmt="o", ms=7, color=BLUE,
                    mfc=BLUE, mec="white", mew=1.6, elinewidth=2, capsize=0, zorder=4)
    ax.set_yticks(range(len(techs)))
    ax.set_yticklabels([t[1] for t in techs][::-1])
    ax.set_ylim(-0.6, len(techs) - 0.4)
    ax.set_xlim(40, 100)
    ax.set_xlabel("accuracy after one year, all mitigations on (%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", color=BLUE, mec="white", ms=7, lw=2),
                Line2D([], [], marker="o", color=ORANGE, mec="white", ms=7, lw=2)],
               ["PyRRAM-Sim (5 simulated chips)", "Paper (10 chips)"], loc="upper left", ncol=2,
               bbox_to_anchor=(0.006, 0.84), handlelength=1.6)
    fig.subplots_adjust(top=0.70)
    save(fig, "sdc_technologies.png", "Which memory survives a year in orbit best?",
         "Dot = average, line = chip-to-chip spread. Same ranking as the paper: SONOS > PCM > ReRAM")


def sdc_radiation_temperature(rows):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    dose = sorted((r for r in rows if r["config"] == "D4" and r["t"] == "12mo"), key=lambda r: r["env"])
    sdc_band(a, [r["env"] for r in dose], [100 * r["paper"] for r in dose], [100 * r["paper_std"] for r in dose],
             [100 * r["mean"] for r in dose])
    a.set_xlabel("total radiation dose, krad(Si)   (the nominal mission dose is 1 krad per year)")
    a.set_xticks([0.1, 1, 10, 100, 1000])
    a.set_xticklabels(["0.1", "1", "10", "100", "1,000"])
    a.set_ylabel("accuracy after one year (%)")
    a.set_title("Radiation dose")
    a.minorticks_off()
    temp = sorted((r for r in rows if r["config"] == "T4" and r["t"] == "12mo"), key=lambda r: r["env"])
    sdc_band(b, [r["env"] for r in temp], [100 * r["paper"] for r in temp], [100 * r["paper_std"] for r in temp],
             [100 * r["mean"] for r in temp], log=False)
    b.set_xlabel("operating temperature (°C)")
    b.set_title("Temperature")
    legend_paper_ours(fig, "PyRRAM-Sim: average of 2 simulated chips", y=0.87)
    fig.subplots_adjust(top=0.70, wspace=0.08)
    save(fig, "sdc_radiation_temperature.png", "Radiation and temperature, PCM chip with all mitigations",
         "Our points fall inside the paper's chip-to-chip band; with only 2 chips per point they scatter more")


# --------------------------------------------------------------------------- HARD-DPO
def hard_dpo():
    d = json.load(open(os.path.join(DATA, "hard_dpo.json")))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    alphas = ["0.0", "0.004", "0.008"]
    x = [0, 0.4, 0.8]
    base = d["default_protocol"]["instruction_tuned"]
    hd = d["default_protocol"]["hard_dpo_paper_checkpoint"]
    a.plot(x, [base[k]["flip_rate"] for k in alphas], color=GRAY)
    dots(a, x, [base[k]["flip_rate"] for k in alphas], GRAY)
    a.plot(x, [hd[k]["flip_rate"] for k in alphas], color=BLUE)
    dots(a, x, [hd[k]["flip_rate"] for k in alphas], BLUE)
    a.text(0.82, base["0.008"]["flip_rate"], "instruction-tuned model", va="center", color=INK, fontsize=9)
    a.text(0.82, hd["0.008"]["flip_rate"] + 0.6, "HARD-DPO model", va="bottom", color=INK, fontsize=9)
    a.set_xlim(-0.05, 1.3)
    a.set_xticks(x)
    a.set_xticklabels(["0%", "0.4%", "0.8%"])
    a.set_ylim(-0.8, 25)
    a.set_xlabel("programming error of the chip (share of the conductance range)")
    a.set_ylabel("flipped preferences (%)")
    a.set_title("The noisier the chip, the more answers flip")
    a.grid(axis="x", visible=False)
    pr = d["released_protocol_alpha_0.008"]
    groups = [("instruction_tuned", "instruction-tuned"), ("hard_dpo", "HARD-DPO")]
    w = 0.22
    for i, (k, name) in enumerate(groups):
        ours = pr["pyrram_sim"][k]
        b.bar(i - w / 2 - 0.01, ours["flip_rate"], w, color=BLUE, yerr=ours["flip_rate_ci95"],
              error_kw=dict(ecolor=INK2, elinewidth=1.2, capsize=0), zorder=3)
        b.bar(i + w / 2 + 0.01, pr["paper"][k]["flip_rate"], w, color=ORANGE, zorder=3)
    b.set_xticks([0, 1])
    b.set_xticklabels([g[1] for g in groups])
    b.set_ylabel("flipped preferences (%)")
    b.set_ylim(0, 40)
    b.set_title("Same test as the paper (0.8% error)")
    b.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    b.legend([Patch(color=BLUE), Patch(color=ORANGE)], ["PyRRAM-Sim", "Paper"], loc="upper right")
    fig.subplots_adjust(top=0.74, wspace=0.28)
    save(fig, "hard_dpo.png", "HARD-DPO keeps an LLM's preferences stable on a noisy RRAM chip",
         "Share of answer pairs whose order (better vs. worse answer) the chip reverses; Qwen2.5-0.5B-Instruct.\n"
         "Thin gray line: 95% confidence range of our measurement")


# --------------------------------------------------------------------------- fault-tolerant experts
def experts():
    d = json.load(open(os.path.join(DATA, "experts_resnet18.json")))
    cond = d["conditions"]
    x = np.arange(len(cond))
    acc = d["accuracy"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [("unprotected", "no protection", GRAY), ("mof-lora", "MoF-LoRA (all experts always on)", AQUA),
              ("sparse-mof", "Sparse-MoF (experts chosen by the chip test)", BLUE)]
    for key, name, color in series:
        ax.plot(x, acc[key], color=color, label=name)
        dots(ax, x, acc[key], color)
    ax.set_xticks(x)
    ax.set_xticklabels(cond)
    ax.set_ylim(0, 100)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("problems added to the chip, one after another, from left to right")
    ax.grid(axis="x", visible=False)
    ax.axhline(10, color=AXIS, lw=1, zorder=1)
    ax.text(0, 12, "chance level (random guessing)", fontsize=9, color=INK2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.16))
    save(fig, "experts_resnet18.png", "Fault-tolerant experts rescue a network from broken memory cells",
         "ResNet18 on CIFAR-10. Stuck-at faults drop the unprotected network to chance; the experts bring it back")


def simulator_check():
    d = json.load(open(os.path.join(DATA, "simulator_check.json")))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    rows = d["rows"]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["accuracy"] for r in rows], height=0.3, color=BLUE, zorder=3)
    for yi, r in zip(y, rows):
        if r["accuracy"] > 25:   # inside the bar end, in white
            ax.text(r["accuracy"] - 1.5, yi, f"{r['accuracy']:.0f}%", va="center", ha="right", color="white",
                    fontsize=9, fontweight="semibold", zorder=4)
        else:
            ax.text(r["accuracy"] + 1.2, yi, f"{r['accuracy']:.0f}%", va="center", color=INK, fontsize=9)
    ax.axvline(d["digital_accuracy"], color=INK2, lw=1, zorder=2)
    ax.text(d["digital_accuracy"] - 1, -0.45, "digital network (no chip)", ha="right", va="center", color=INK2,
            fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r["chip"] for r in rows])
    ax.set_xlim(0, 105)
    ax.set_xlabel("accuracy on CIFAR-10 (%)")
    ax.grid(axis="y", visible=False)
    save(fig, "simulator_check.png", "A perfect chip matches the digital network; broken cells hurt fast",
         "ResNet18 on CIFAR-10, 8-bit weights")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sdc = load_sdc()
    sdc_agreement(sdc)
    sdc_mission(sdc)
    sdc_technologies(sdc)
    sdc_radiation_temperature(sdc)
    hard_dpo()
    experts()
    simulator_check()
