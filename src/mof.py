"""
Mixture of fault-tolerant low-rank experts.

MoF-LoRA [MoF-LoRA] trains one LoRA expert per RRAM non-ideality and merges all of them into
the base weights at design time. Sparse-MoF [HAT-FI] keeps the experts as separate adapters
and activates the relevant subset on the deployed chip from hardware telemetry; the active set
is merged and the chip is reprogrammed whenever it changes.

Expert training stages (paper-preset noise from train_noise.py):

    prog   programming error, truncated Gaussian sigma = alpha * std(W)      chain, stage 1
    drift  one-time tail clip of W1 = W0 + dW_prog to +-3 std, then          chain, stage 2
           state-proportional noise sigma = alpha * |W| and the
           distribution-aware regularizer R_drift [MoF-LoRA, Eq. 17]
    saf    the chip's stuck-at map (March test) masks the merged chain       chain, stage 3
           weights; the expert learns to work around the faulty weights
    read   per-inference state-proportional noise sigma = alpha * |W|        independent (on W0)
    temp   truncated Gaussian with alpha(T) = alpha0 * Arrhenius factor      independent (on W0)
"""
from dataclasses import dataclass
from itertools import cycle
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lora import (
    add_expert, expert_names, fault_context, lora_modules, noise_context,
    set_active, set_chain_clip, set_merge, set_trainable, active_names,
)
from rram_noise_model import arrhenius_factor
from train_noise import ProportionalNoise, TruncNormalNoise

EXPERT_ORDER = ("prog", "drift", "saf", "read", "temp")
CHAIN = ("prog", "drift", "saf")

# Published training settings. Values the papers do not report are marked "not reported".
RECIPES = {
    # HAT-FI, Llama-3.1-8B-Instruct: Table I, LoRA r=64, alpha=128 on q/k/v/o/gate/up/down.
    "hat-fi-llm": dict(rank=64, lora_alpha=128, prog=0.06, drift=0.03, p_saf=0.02, read=0.03, temp=0.06,
                       prog_target="effective"),
    # HAT-FI, ResNet-18 on CIFAR-10: Table I, weight-space LoRA r=4 on all Conv2d layers
    # (lora_alpha not reported; alpha = r gives scale 1).
    "hat-fi-cnn": dict(rank=4, lora_alpha=4, prog=0.50, drift=0.30, p_saf=0.15, read=0.30, temp=0.50,
                       prog_target="effective"),
    # MoF-LoRA: r=32, AdamW lr 3e-4, wd 1e-4, cosine; scale s = 1/r; 5% programming error for
    # vision (2% for language) on W0, A and B; drift stage = tail clip + R_drift (Algorithm 1).
    "mof-lora": dict(rank=32, lora_alpha=1, prog=0.05, drift=0.0, p_saf=None, read=None, temp=None,
                     prog_target="factors"),
}


class DriftRegularizer:
    """
    R_drift = alpha * sum_l sum_ij max(0, |W2_ij| - k std(W2))^2 over the merged weights W2 of
    every adapted layer [MoF-LoRA, Eq. 17; HAT-FI, Eq. 2]. The strength alpha is not reported in
    the papers. backward() computes and back-propagates the term layer by layer, so no merged
    weight is kept alive between layers.
    """

    def __init__(self, alpha: float = 1.0, k: float = 3.0):
        self.alpha, self.k = alpha, k

    def backward(self, model: nn.Module) -> float:
        total = 0.0
        if self.alpha <= 0:
            return total
        for _, m in lora_modules(model):
            W = m.effective_weight()
            tau = self.k * W.detach().std()
            r = self.alpha * F.relu(W.abs() - tau).pow(2).sum()
            if r.requires_grad:
                r.backward()
            total += r.item()
        return total


def train_expert(
    model: nn.Module,
    name: str,
    loss_fn: Callable,
    data: Iterable,
    steps: int,
    rank: int = 8,
    lora_alpha: float = 16,
    active_before: Sequence[str] = (),
    noise=None,
    faults: Optional[dict] = None,
    regularizer: Optional[DriftRegularizer] = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 50,
    train_mode: bool = False,
) -> list:
    """
    Trains expert `name` (created if missing) on top of the active experts `active_before`,
    with AdamW and cosine annealing [MoF-LoRA, Sec. V-A]. loss_fn(model, batch) returns the task
    loss; `data` is cycled. Only the expert's factors are updated. train_mode=False keeps
    BatchNorm statistics and dropout of the frozen base model fixed. Returns the loss history.
    """
    if name not in expert_names(model):
        add_expert(model, name, rank, lora_alpha)
    set_active(model, list(active_before) + [name])
    params = set_trainable(model, [name])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    model.train(train_mode)
    history, it = [], cycle(data)
    for step in range(steps):
        batch = next(it)
        opt.zero_grad(set_to_none=True)
        # Keep the perturbations installed through backward(): checkpointed layers are recomputed.
        with noise_context(model, noise), fault_context(model, faults):
            loss = loss_fn(model, batch)
            loss.backward()
        reg = regularizer.backward(model) if regularizer is not None else 0.0
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        sched.step()
        history.append({"step": step + 1, "loss": loss.item(), "reg": reg})
        if log_every and (step + 1) % log_every == 0:
            print(f"[{name}] step {step + 1}/{steps} loss={loss.item():.4f} reg={reg:.4g}")
    model.eval()
    set_trainable(model, [])
    return history


def train_prog_expert(model, loss_fn, data, steps, alpha=0.06, target="effective", name="prog", **kw):
    """Stage 1: programming-error expert on W0 [MoF-LoRA, Sec. III-C; HAT-FI, Eq. 1]."""
    return train_expert(model, name, loss_fn, data, steps, active_before=(),
                        noise=TruncNormalNoise(alpha, target), **kw)


def train_drift_expert(model, loss_fn, data, steps, alpha=0.03, reg_alpha=1.0, clip_k=3.0,
                       after=("prog",), name="drift", rank=8, lora_alpha=16, **kw):
    """
    Stage 2: drift expert on the clipped merged weights W1 = clip(W0 + dW_prog, +-clip_k std)
    [MoF-LoRA, Sec. III-D]. alpha > 0 adds state-proportional noise [HAT-FI]; alpha = 0 follows
    MoF-LoRA Algorithm 1 (clipping and regularizer only).
    """
    if name not in expert_names(model):
        add_expert(model, name, rank, lora_alpha)
    set_active(model, list(after) + [name])
    set_chain_clip(model, name, clip_k)
    return train_expert(model, name, loss_fn, data, steps, rank=rank, lora_alpha=lora_alpha,
                        active_before=after, noise=ProportionalNoise(alpha) if alpha > 0 else None,
                        regularizer=DriftRegularizer(reg_alpha, clip_k), **kw)


def saf_overlays(model: nn.Module, sim, exact: bool = False) -> dict:
    """
    Weight-domain stuck-at overlays {layer: (flat_idx, values)} from the chip's March test.

    exact=False (paper preset): every weight with a stuck cell is frozen to the value it takes
    when the current active chain is programmed [MoF-LoRA, Sec. III-E].
    exact=True: the frozen values are recomputed from the current weights on every forward
    pass, so a partially stuck differential pair follows its healthy partner as on the chip.
    """
    faults = sim.fault_map()
    if not faults:
        raise ValueError("The simulator has no stuck-at faults; call sim.apply_stuck_at_faults first.")
    out = {}
    for name, m in lora_modules(model):
        fs = faults.get(name)
        if fs is None or fs.idx.numel() == 0:
            continue
        shape = m.weight.shape
        if exact:
            out[name] = (lambda W, fs=fs, shape=shape:
                         (fs.idx.to(W.device), sim.realize_faulty(fs, W.reshape(shape)).to(W.device)))
        else:
            with torch.no_grad():
                W = m.effective_weight().reshape(shape)
            out[name] = (fs.idx.to(W.device), sim.realize_faulty(fs, W).to(W.device))
    return out


def train_saf_expert(model, sim, loss_fn, data, steps, after=("prog", "drift"), exact=False,
                     reg_alpha=1.0, clip_k=3.0, name="saf", rank=8, lora_alpha=16, **kw):
    """
    Stage 3: SAF expert trained on the weights masked by the fault map of `sim` (the target
    chip) [MoF-LoRA, Eq. 19]. The layer names of `model` must match those of the simulated model.
    Faulty weights receive no update, so the expert redistributes the computation over the
    healthy weights, which also holds after it is merged into the crossbar.
    """
    if name not in expert_names(model):
        add_expert(model, name, rank, lora_alpha)
    set_active(model, list(after))
    overlays = saf_overlays(model, sim, exact)
    return train_expert(model, name, loss_fn, data, steps, rank=rank, lora_alpha=lora_alpha,
                        active_before=after, faults=overlays,
                        regularizer=DriftRegularizer(reg_alpha, clip_k), **kw)


def train_read_expert(model, loss_fn, data, steps, alpha=0.03, name="read", **kw):
    """Stage 4: read-noise expert, trained independently on W0 [HAT-FI, Sec. II-A]."""
    return train_expert(model, name, loss_fn, data, steps, active_before=(),
                        noise=ProportionalNoise(alpha), **kw)


def train_temp_expert(model, loss_fn, data, steps, alpha0=0.06, temperature_C=None, Ea_eV=None,
                      T0_C=25.0, target="effective", name="temp", **kw):
    """
    Stage 5: temperature expert, trained independently on W0 with truncated Gaussian noise at
    an elevated temperature, alpha(T) = alpha0 * exp(Ea / kB * (1/T0 - 1/T)) [HAT-FI, Eq. 3].
    temperature_C and Ea_eV are required (HAT-FI reports neither).
    """
    if temperature_C is None or Ea_eV is None:
        raise ValueError("train_temp_expert needs temperature_C and Ea_eV.")
    alpha = alpha0 * arrhenius_factor(temperature_C, Ea_eV, T0_C)
    return train_expert(model, name, loss_fn, data, steps, active_before=(),
                        noise=TruncNormalNoise(alpha, target), **kw)


def train_experts(model, loss_fn, data, steps, experts=CHAIN, recipe="hat-fi-llm", sim=None,
                  temperature_C=None, Ea_eV=None, saf_exact=False, reg_alpha=1.0, **kw) -> dict:
    """
    Trains the requested experts in pipeline order with a recipe's coefficients (override any
    of them, e.g. prog=0.02). The SAF stage needs `sim` with stuck-at faults applied.
    """
    cfg = dict(RECIPES[recipe])
    for key in ("rank", "lora_alpha", "prog", "drift", "read", "temp", "prog_target"):
        if key in kw:
            cfg[key] = kw.pop(key)
    common = dict(rank=cfg["rank"], lora_alpha=cfg["lora_alpha"], **kw)
    unknown = set(experts) - set(EXPERT_ORDER)
    if unknown:
        raise ValueError(f"Unknown experts {sorted(unknown)}; choose from {EXPERT_ORDER}.")
    chain_done, history = [], {}
    for e in EXPERT_ORDER:
        if e not in experts:
            continue
        if e == "prog":
            history[e] = train_prog_expert(model, loss_fn, data, steps, alpha=cfg["prog"],
                                           target=cfg["prog_target"], **common)
        elif e == "drift":
            history[e] = train_drift_expert(model, loss_fn, data, steps, alpha=cfg["drift"] or 0.0,
                                            reg_alpha=reg_alpha, after=tuple(chain_done), **common)
        elif e == "saf":
            if sim is None:
                raise ValueError("The SAF expert needs the target chip: pass sim=PyRRAMSimulator(...).")
            history[e] = train_saf_expert(model, sim, loss_fn, data, steps, after=tuple(chain_done),
                                          exact=saf_exact, reg_alpha=reg_alpha, **common)
        elif e == "read":
            history[e] = train_read_expert(model, loss_fn, data, steps, alpha=cfg["read"], **common)
        elif e == "temp":
            history[e] = train_temp_expert(model, loss_fn, data, steps, alpha0=cfg["temp"],
                                           temperature_C=temperature_C, Ea_eV=Ea_eV, **common)
        if e in CHAIN:
            chain_done.append(e)
    return history


# ---------------------------------------------------------------------------- Sparse-MoF

@dataclass
class Thresholds:
    """Expert activation thresholds [HAT-FI, VTS 2026 talk]; prog is always on."""
    drift: float = 0.010       # measured relative drift alpha
    saf: float = 0.020         # measured stuck-cell rate
    read: float = 0.003        # measured relative read-noise alpha
    temp: float = 35.0         # temperature sensor, Celsius


class SparseMoF:
    """
    Telemetry-guided expert selection on a simulated chip [HAT-FI, Sec. II-A.2].

    The programming-error expert is always active; drift, SAF and read-noise experts switch on
    when the online test (sim.telemetry) exceeds their threshold, the temperature expert when the
    temperature sensor does. Lifetime conditions (drift, SAF) are latched once detected, since
    reprogramming resets the measured drift but not its cause. With deploy="merged" every change
    of the active set re-merges the experts and reprograms the chip (stuck-at map kept), followed
    by `reprogram_params` (programming error, and optionally relaxation and drift over the same
    horizon). With deploy="hybrid" the experts run digitally and nothing is reprogrammed.

    An optional accuracy probe (probe(model) -> accuracy) is the secondary trigger: if accuracy
    falls more than max_drop below `baseline`, each optional expert is toggled in turn and the
    toggle is kept when the probe improves.
    """

    def __init__(self, sim, experts: Optional[Sequence[str]] = None, always_on=("prog",),
                 thresholds: Optional[Thresholds] = None, latched=("drift", "saf"),
                 deploy: str = "merged", reprogram_params: Optional[dict] = None,
                 probe: Optional[Callable] = None, baseline: Optional[float] = None, max_drop: float = 5.0):
        if deploy not in ("merged", "hybrid"):
            raise ValueError("deploy must be 'merged' or 'hybrid'.")
        self.sim = sim
        self.model = sim.digital_model
        available = expert_names(self.model)
        self.experts = [e for e in EXPERT_ORDER if e in (experts or available) and e in available]
        self.always_on = [e for e in always_on if e in self.experts]
        self.thresholds = thresholds or Thresholds()
        self.latched_kinds = set(latched)
        self.latched = set()
        self.deploy = deploy
        self.reprogram_params = reprogram_params or {}
        self.probe, self.baseline, self.max_drop = probe, baseline, max_drop

    def reset(self):
        """Fresh deployment: only the always-on experts, nothing latched."""
        self.latched = set()
        set_active(self.model, self.always_on)
        set_merge(self.model, self.deploy == "merged")

    def select(self, telemetry: dict) -> list:
        t, on = self.thresholds, set(self.always_on)
        triggers = {
            "drift": telemetry.get("drift_alpha", 0.0) >= t.drift,
            "saf": telemetry.get("saf_rate", 0.0) >= t.saf,
            "read": telemetry.get("read_alpha", 0.0) >= t.read,
            "temp": telemetry.get("temperature_C", 25.0) >= t.temp,
        }
        for kind, fired in triggers.items():
            if fired and kind in self.experts:
                on.add(kind)
                if kind in self.latched_kinds:
                    self.latched.add(kind)
        on |= self.latched & set(self.experts)
        return [e for e in EXPERT_ORDER if e in on]

    def activate(self, names: Sequence[str], force: bool = False) -> bool:
        """Switches the active expert set on the chip; returns True if it changed."""
        names = [e for e in EXPERT_ORDER if e in names]
        if not force and names == active_names(self.model) and self.model_merge_matches():
            return False
        set_active(self.model, names)
        set_merge(self.model, self.deploy == "merged")
        if self.deploy == "merged":
            self.sim.reprogram()
            p = self.reprogram_params
            if p.get("programming_error_params"):
                self.sim.apply_programming_error(**p["programming_error_params"])
            if p.get("relaxation_params"):
                self.sim.apply_conductance_relaxation(**p["relaxation_params"])
            if p.get("drift_params"):
                self.sim.apply_relative_drift(**p["drift_params"])
        else:
            self.sim.update_adapters()
        return True

    def model_merge_matches(self) -> bool:
        mods = lora_modules(self.model)
        return not mods or mods[0][1].merge == (self.deploy == "merged")

    def step(self, n_reads: int = 16) -> dict:
        """One telemetry -> selection -> (re)deployment cycle."""
        tel = self.sim.telemetry(n_reads)
        names = self.select(tel)
        changed = self.activate(names)
        result = {"telemetry": tel, "active": names, "changed": changed}
        if self.probe is not None and self.baseline is not None:
            acc = self.probe(self.sim.analog_model)
            result["probe"] = acc
            if self.baseline - acc > self.max_drop:
                result["active"], result["probe"] = self._reevaluate(names, acc)
                result["changed"] = True
        return result

    def _reevaluate(self, names, acc):
        best, best_acc = list(names), acc
        for e in self.experts:
            if e in self.always_on:
                continue
            trial = [x for x in best if x != e] if e in best else best + [e]
            self.activate(trial, force=True)
            trial_acc = self.probe(self.sim.analog_model)
            if trial_acc > best_acc:
                best, best_acc = [x for x in EXPERT_ORDER if x in trial], trial_acc
        self.activate(best, force=True)
        return best, best_acc
