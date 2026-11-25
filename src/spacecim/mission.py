"""
Mission timeline of an analog CiM accelerator in orbit [COINS'26, Sec. IV-V].

One mission converts the digital model to analog tiles, programs it once (t = 0) and steps
through the checkpoints (1 s ... 12 months). At every checkpoint the tiles are read from
the *programmed* state at time t, so drift is measured from programming, while upsets
accumulate across checkpoints. Optional runtime mitigations at each checkpoint:

* global drift compensation (per-tile scalar from an identity probe, label-free),
* recalibration of the digital LayerNorm affine terms and/or the classifier head on a
  small labelled calibration set, restarted from the launch parameters at every
  checkpoint so that each fit is independent.

The design-time mitigation, noise-aware LoRA, is applied to the digital model before the
mission (see presto/train.py and train_noise.AbsMaxGaussianNoise).
"""
import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import tile as T
from .devices import build_device
from .environment import TID, Thermal

PAPER_CHECKPOINTS = [("1s", 1), ("1h", 3600), ("1d", 86400), ("1wk", 604800), ("1mo", 2592000),
                     ("3mo", 7776000), ("6mo", 15552000), ("12mo", 31536000)]
TECH_CHECKPOINTS = [("1s", 1), ("1h", 3600), ("1d", 86400), ("5d", 432000), ("1mo", 2592000),
                    ("3mo", 7776000), ("6mo", 15552000), ("12mo", 31536000)]
SEU_NOMINAL = 1e-6   # upsets / device / day


@dataclass
class MissionConfig:
    device: str = "pcm"
    device_kwargs: dict = field(default_factory=lambda: {"g_max": 25.0})
    periphery: T.Periphery = field(default_factory=T.Periphery)
    tid: dict = field(default_factory=lambda: {"enabled": True, "dose_rate_krad_per_year": 1.0,
                                               "k_tid": 0.05, "k_shift": 0.0, "shift_mode": "absolute"})
    thermal: dict = field(default_factory=lambda: {"enabled": False})
    seu_lambda: float = SEU_NOMINAL
    seu_mode: str = "redraw"
    seu_p_redraw: float = 0.8
    seu_discharge_max: float = 1.0
    drift_compensation: bool = False
    recalibrate: Sequence[str] = ()          # () or a subset of ("layernorm", "head")
    checkpoints: Sequence[Tuple[str, float]] = tuple(PAPER_CHECKPOINTS)
    exclude: Sequence[str] = ("head",)


def build_analog_model(digital: nn.Module, cfg: MissionConfig, seed: int) -> nn.Module:
    model = copy.deepcopy(digital)
    device = build_device(cfg.device, **cfg.device_kwargs)
    return T.convert_to_analog(model, device, cfg.periphery, TID(**cfg.tid), Thermal(**cfg.thermal),
                               seu_rate=cfg.seu_lambda, seu_mode=cfg.seu_mode, seu_p_redraw=cfg.seu_p_redraw,
                               seu_discharge_max=cfg.seu_discharge_max, drift_compensation=cfg.drift_compensation,
                               exclude=cfg.exclude, seed=seed)


def recalibration_parameters(model: nn.Module, targets: Sequence[str]) -> List[nn.Parameter]:
    params = []
    if "layernorm" in targets:
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                params += [p for p in m.parameters() if p is not None]
    if "head" in targets:
        head = getattr(model, "head", None)
        if head is not None:
            params += list(head.parameters())
    return params


def digital_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Every parameter and buffer (the analog tiles keep their state outside the module tree)."""
    return {k: v.detach().clone() for k, v in list(model.named_parameters()) + list(model.named_buffers())}


def digital_restore(model: nn.Module, snap: Dict[str, torch.Tensor]):
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    with torch.no_grad():
        for k, v in snap.items():
            tensors[k].copy_(v)


def recalibrate(model: nn.Module, batches: Callable[[torch.Generator], object], loss_on_batch: Callable,
                targets: Sequence[str] = ("layernorm", "head"), epochs: int = 5, lr: float = 1e-3,
                weight_decay: float = 0.0, generator: Optional[torch.Generator] = None) -> dict:
    """
    Fits only the digital LayerNorm affine terms and/or the head on the calibration batches,
    in eval mode, with gradients through the analog tiles (ideal backward). The analog
    weights are never written.
    """
    params = recalibration_parameters(model, targets)
    if not params:
        raise ValueError(f"No recalibration parameters for targets {targets}")
    saved = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    model.eval()
    losses = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for batch in batches(generator):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = loss_on_batch(model, batch)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1
        losses.append(total / max(n, 1))
    for name, p in model.named_parameters():
        p.requires_grad_(saved.get(name, False))
    return {"recal_losses": losses, "recal_n_params": int(sum(p.numel() for p in params))}


def mission_seed(seed: int, draw: int, mission_index: int) -> int:
    return seed * 1000 + 100 * draw + mission_index


def run_mission(digital: nn.Module, cfg: MissionConfig, seed: int, evaluate: Callable[[nn.Module], dict],
                calib_batches: Optional[Callable] = None, loss_on_batch: Optional[Callable] = None,
                recal_kwargs: Optional[dict] = None, labels: Optional[Sequence[str]] = None,
                log: Callable[[str], None] = print) -> List[dict]:
    """
    Runs one mission with mission seed ``seed`` and returns one record per evaluated
    checkpoint. ``labels`` restricts the mission to some checkpoints; upsets of the skipped
    intervals are drawn at the next read (Poisson counts add up).
    """
    torch.manual_seed(seed)
    model = build_analog_model(digital, cfg, seed)
    model.to(next(digital.parameters()).device)
    T.program(model)
    snap = digital_snapshot(model) if cfg.recalibrate else None
    records = []
    selected = [(l, s) for l, s in cfg.checkpoints if labels is None or l in labels]
    for label, seconds in selected:
        started = time.time()
        T.drift_to(model, float(seconds))
        info = {}
        if cfg.recalibrate:
            digital_restore(model, snap)
            gen = torch.Generator().manual_seed(seed * 131 + int(seconds) % 100003)
            info = recalibrate(model, calib_batches, loss_on_batch, cfg.recalibrate, generator=gen,
                               **(recal_kwargs or {}))
        result = evaluate(model)
        rec = {"t_label": label, "t_seconds": float(seconds), **result, **T.seu_totals(model),
               "wall_time": time.time() - started,
               **{k: v for k, v in info.items() if k != "recal_losses"}}
        if cfg.drift_compensation:
            alphas = [float(t.alpha) for t in T.analog_tiles(model)]
            rec["gdc_alpha_mean"] = sum(alphas) / len(alphas)
        records.append(rec)
        log(f"    t={label:>5s} acc={result.get('accuracy', float('nan')):.4f} "
            f"seu_hits={rec['seu_hits']:,} ({rec['wall_time']:.1f}s)")
    return records


# ---------------------------------------------------------------------------- paper arms

DOSES = [0.1, 0.3, 1, 3, 10, 30, 60, 100, 200, 300, 1000, 3000]
TEMPERATURES = [-100, -60, -40, -20, -10, 0, 10, 15, 20, 25, 30, 35, 40, 45, 55, 70, 85, 125]
_DOSE_SWEEP = {"key": "tid.dose_offset_krad", "values": DOSES, "fixed": {"tid.dose_rate_krad_per_year": 0.0}}
_TEMP_SWEEP = {"key": "thermal.T_op_C", "values": TEMPERATURES, "fixed": {"thermal.enabled": True}}
_PCM = {"device": "pcm", "device_kwargs": {"g_max": 25.0}}
_RERAM = {"device": "reram_cmo", "device_kwargs": {"g_max": 25.0, "g_min": 9.0}}
_RERAM_DIFF = {"device": "reram_cmo_diff", "device_kwargs": {"g_max": 25.0, "g_min": 9.0}}
_SONOS = {"device": "sonos", "device_kwargs": {"g_max": 16.0, "g_min": 0.0, "v_read": 0.1,
                                               "extrapolate": True, "retention_law": "log"},
          "seu_mode": "partial_discharge"}
_SONOS_SHIFT = {**_SONOS, "tid_extra": {"k_shift": 1e-2, "shift_mode": "absolute"}}
_LADDER = {
    1: {"model": "e0"},                                                  # naive
    2: {"model": "e2"},                                                  # + noise-aware LoRA
    3: {"model": "e2", "gdc": True},                                     # + drift compensation
    4: {"model": "e2", "gdc": True, "recal": ("layernorm", "head")},     # + recalibration
}


def _arm(model, gdc=False, recal=(), seu=(1,), endpoint=(), no_seu=False, checkpoints=PAPER_CHECKPOINTS,
         sweep=None, tech=None):
    tech = dict(tech or _PCM)
    return {"model": model, "gdc": gdc, "recal": tuple(recal), "seu_multipliers": tuple(seu),
            "endpoint_multipliers": tuple(endpoint), "include_no_seu": no_seu,
            "checkpoints": tuple(checkpoints), "sweep": sweep, **tech}


# Arms of the COINS'26 case study; names follow the original experiment codes.
ARMS = {
    # Fig. 2: mitigation ladder over the 12-month mission (PCM)
    "E1a": _arm("e0", seu=(), no_seu=True),
    "E1b": _arm("e0", seu=(1, 10, 100)),
    "E2": _arm("e2", endpoint=(10, 100)),
    "E3": _arm("e2", gdc=True, endpoint=(10, 100)),
    "E4": _arm("e2", gdc=True, recal=("layernorm", "head"), endpoint=(10, 100)),
    # Fig. 3 (leave-one-out) and Sec. V-B (deployable head-only recalibration)
    "E3_noLoRA": _arm("e0", gdc=True, endpoint=(10, 100)),
    "E4_noLoRA": _arm("e0", gdc=True, recal=("layernorm", "head"), endpoint=(10, 100)),
    "E4_head": _arm("e2", gdc=True, recal=("head",), endpoint=(10, 100)),
}
# Fig. 5: technology ladder at 12 months (P = PCM, R = ReRAM single device,
# RP = ReRAM differential pair, S = SONOS projected under the log retention law)
for prefix, tech in (("P", _PCM), ("R", _RERAM), ("RP", _RERAM_DIFF), ("S", _SONOS)):
    for k, step in _LADDER.items():
        ARMS[f"{prefix}{k}Y"] = _arm(step["model"], step.get("gdc", False), step.get("recal", ()),
                                     checkpoints=TECH_CHECKPOINTS, tech=tech)
# Fig. 7: total ionizing dose, PCM ladder, fresh (1 s) and after 12 months
_TWO = (("1s", 1), ("12mo", 31536000))
for k, step in _LADDER.items():
    ARMS[f"D{k}"] = _arm(step["model"], step.get("gdc", False), step.get("recal", ()), checkpoints=_TWO,
                         sweep=_DOSE_SWEEP)
    ARMS[f"T{k}"] = _arm(step["model"], step.get("gdc", False), step.get("recal", ()), checkpoints=_TWO,
                         sweep=_TEMP_SWEEP)
# Fig. 6: technology x dose, freshly programmed (naive and fully mitigated)
for name, tech in (("RPD", _RERAM_DIFF), ("SD", _SONOS), ("SX", _SONOS_SHIFT)):
    ARMS[f"{name}1"] = _arm("e0", checkpoints=(("1s", 1),), sweep=_DOSE_SWEEP, tech=tech)
    ARMS[f"{name}4"] = _arm("e2", True, ("layernorm", "head"), checkpoints=(("1s", 1),), sweep=_DOSE_SWEEP,
                            tech=tech)


def arm_missions(arm: str, sweep_values: Optional[Sequence[float]] = None) -> List[dict]:
    """
    Expands an arm into its missions: one per SEU multiplier (whole timeline) and per
    endpoint multiplier (final checkpoint only), crossed with the environment sweep. Each
    entry carries a MissionConfig, the evaluated labels and the mission index.
    """
    a = ARMS[arm]
    seu_plan = []
    if a["include_no_seu"]:
        seu_plan.append((0.0, False))
    seu_plan += [(float(m), False) for m in a["seu_multipliers"]]
    seu_plan += [(float(m), True) for m in a["endpoint_multipliers"]]
    sweep = a["sweep"]
    env = [None] if sweep is None else list(sweep_values if sweep_values is not None else sweep["values"])
    missions, index = [], 0
    for value in env:
        for mult, endpoint_only in seu_plan:
            cfg = MissionConfig(device=a["device"], device_kwargs=dict(a["device_kwargs"]),
                                seu_lambda=SEU_NOMINAL * mult, seu_mode=a.get("seu_mode", "redraw"),
                                drift_compensation=a["gdc"], recalibrate=a["recal"],
                                checkpoints=a["checkpoints"])
            cfg.tid.update(a.get("tid_extra", {}))
            if sweep is not None:
                for key, v in list(sweep.get("fixed", {}).items()) + [(sweep["key"], value)]:
                    block, attr = key.split(".")
                    getattr(cfg, block)[attr] = v
            labels = [a["checkpoints"][-1][0]] if endpoint_only else None
            missions.append({"cfg": cfg, "labels": labels, "mission_index": index, "seu_multiplier": mult,
                             "endpoint_only": endpoint_only, "env_value": value, "model": a["model"]})
            index += 1
    return missions
