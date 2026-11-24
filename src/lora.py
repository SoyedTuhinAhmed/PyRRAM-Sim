"""
Low-rank fault-tolerant experts for nn.Linear and nn.Conv2d layers.

Every adapted layer holds its frozen base weight W0 and a bank of named LoRA experts
(e.g. "prog", "drift", "saf", "read", "temp", "dpo"). Experts are kept as separate adapters
[HAT-FI]: any subset can be switched on, and it can either run digitally next to the crossbar
(hybrid deployment) or be merged into the weights that are programmed into the crossbar
[MoF-LoRA, HARD-DPO]. An expert's update is dW = (alpha / r) * B @ A with A in R^{r x fan_in}
and B in R^{out x r} [LoRA]; for Conv2d, dW is reshaped to the kernel (weight-space LoRA).

Active experts are applied in the order they were added. A chain expert may carry a clip
bound c > 0: the running weight is clamped to [-c, c] before the expert's update is added,
which reproduces the one-time tail clipping of the merged weights before drift adaptation
[MoF-LoRA, Sec. III-D].
"""
import math
from contextlib import contextmanager
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRAExpert(nn.Module):
    """
    One low-rank expert. Its flags are plain Python attributes, so the forward pass needs no
    device synchronization; they are saved and restored through the module's extra state.
    """

    def __init__(self, out_features: int, fan_in: int, rank: int, alpha: float, device=None):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be at least 1.")
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, fan_in, device=device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = float(alpha) / rank
        self.active = False
        self.clip = 0.0

    def get_extra_state(self):
        return {"scale": self.scale, "active": self.active, "clip": self.clip}

    def set_extra_state(self, state):
        self.scale = float(state["scale"])
        self.active = bool(state["active"])
        self.clip = float(state["clip"])

    def delta(self, A: Optional[torch.Tensor] = None, B: Optional[torch.Tensor] = None) -> torch.Tensor:
        """dW = scale * B @ A as an (out, fan_in) float32 matrix, optionally with perturbed factors."""
        A = self.A if A is None else A
        B = self.B if B is None else B
        return self.scale * (B @ A)


class _LoRAMixin:
    """Expert bank shared by LoRALinear and LoRAConv2d (installed by class swap in add_lora)."""

    def _init_lora(self):
        self.experts = nn.ModuleDict()
        # True: active experts are programmed into the crossbar together with W0 (merged deployment).
        self.merge = False
        self.noise = None   # training-time perturbation, see noise presets in mof.py
        self.fault = None   # training-time stuck-at overlay (flat indices, values) or callable

    def get_extra_state(self):
        return {"merge": self.merge}

    def set_extra_state(self, state):
        self.merge = bool(state["merge"])

    # ------------------------------------------------------------- expert bank

    def _fan_in(self) -> int:
        return self.weight[0].numel()

    def add_expert(self, name: str, rank: int, alpha: float):
        if name in self.experts:
            raise ValueError(f"Expert '{name}' already exists.")
        self.experts[name] = LoRAExpert(self.weight.shape[0], self._fan_in(), rank, alpha, self.weight.device)

    def active_experts(self):
        return [(n, e) for n, e in self.experts.items() if e.active]

    def _apply_experts(self, W: torch.Tensor, factors: Optional[dict] = None) -> torch.Tensor:
        """Runs the active expert chain on top of W (2-D float32)."""
        for name, e in self.active_experts():
            if e.clip > 0:
                W = W.clamp(-e.clip, e.clip)
            A, B = (factors or {}).get(name, (None, None))
            W = W + e.delta(A, B)
        return W

    def programmed_weight(self) -> torch.Tensor:
        """Weight that a crossbar stores: W0, plus the active experts when merge is set."""
        with torch.no_grad():
            W = self.weight.float().reshape(self.weight.shape[0], -1)
            if self.merge:
                W = self._apply_experts(W)
            return W.reshape(self.weight.shape)

    def effective_weight(self) -> torch.Tensor:
        """W0 followed by the active experts (2-D float32, differentiable w.r.t. the experts)."""
        return self._apply_experts(self.weight.float().reshape(self.weight.shape[0], -1))

    def _training_weight(self) -> torch.Tensor:
        """Effective weight with the training-time noise and stuck-at overlay applied."""
        if self.noise is not None:
            W = self.noise(self)
        else:
            W = self.effective_weight()
        if self.fault is not None:
            fault = self.fault(W.detach()) if callable(self.fault) else self.fault
            idx, values = fault
            if idx.numel():
                W = W.reshape(-1).index_put((idx,), values.to(W.dtype)).reshape(W.shape)
        return W

    def _needs_materialized_weight(self) -> bool:
        return (self.noise is not None or self.fault is not None
                or any(e.clip > 0 for _, e in self.active_experts()))


class LoRALinear(_LoRAMixin, nn.Linear):
    def forward(self, x):
        active = self.active_experts()
        if not active:
            return F.linear(x, self.weight, self.bias)
        if self._needs_materialized_weight():
            W = self._training_weight().to(x.dtype)
            return F.linear(x, W, self.bias)
        y = F.linear(x, self.weight, self.bias)
        for _, e in active:
            xa = F.linear(x.to(e.A.dtype), e.A)
            y = y + (e.scale * F.linear(xa, e.B)).to(y.dtype)
        return y


class LoRAConv2d(_LoRAMixin, nn.Conv2d):
    def forward(self, x):
        if not self.active_experts():
            return self._conv_forward(x, self.weight, self.bias)
        W = self._training_weight().reshape(self.weight.shape).to(x.dtype)
        return self._conv_forward(x, W, self.bias)


LORA_TYPES = (LoRALinear, LoRAConv2d)


def _matches(name: str, target_modules: Optional[Sequence[str]], exclude: Sequence[str]) -> bool:
    leaf = name.split(".")[-1]
    if leaf in exclude:
        return False
    return target_modules is None or leaf in target_modules


def add_lora(
    model: nn.Module,
    target_modules: Optional[Sequence[str]] = DEFAULT_TARGET_MODULES,
    exclude: Sequence[str] = ("lm_head",),
) -> list:
    """
    Turns the matching nn.Linear / nn.Conv2d layers of `model` into expert-bank layers in place
    (no weight copy). target_modules matches the last component of a module name, as in PEFT;
    None adapts every Linear and Conv2d layer except `exclude`. Returns the adapted names.
    """
    names = []
    for name, m in model.named_modules():
        if isinstance(m, LORA_TYPES):
            names.append(name)
            continue
        if type(m) not in (nn.Linear, nn.Conv2d) or not _matches(name, target_modules, exclude):
            continue
        m.__class__ = LoRALinear if type(m) is nn.Linear else LoRAConv2d
        m._init_lora()
        names.append(name)
    if not names:
        raise ValueError("add_lora found no matching nn.Linear or nn.Conv2d layer.")
    for p in model.parameters():
        p.requires_grad_(False)
    return names


def lora_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LORA_TYPES)]


def add_expert(model: nn.Module, name: str, rank: int, alpha: float):
    for _, m in lora_modules(model):
        m.add_expert(name, rank, alpha)


def expert_names(model: nn.Module) -> list:
    mods = lora_modules(model)
    return list(mods[0][1].experts.keys()) if mods else []


def set_active(model: nn.Module, names: Iterable[str]):
    """Activates exactly the experts in `names` in every adapted layer."""
    names = set(names)
    known = set(expert_names(model))
    if names - known:
        raise KeyError(f"Unknown experts {sorted(names - known)}; available: {sorted(known)}")
    for _, m in lora_modules(model):
        for n, e in m.experts.items():
            e.active = n in names


def active_names(model: nn.Module) -> list:
    mods = lora_modules(model)
    return [n for n, _ in mods[0][1].active_experts()] if mods else []


def set_merge(model: nn.Module, merge: bool):
    """merge=True: the active experts are programmed into the crossbar with W0 (merged deployment);
    False: the crossbar stores W0 and the active experts run digitally (hybrid deployment)."""
    for _, m in lora_modules(model):
        m.merge = bool(merge)


def set_trainable(model: nn.Module, names: Iterable[str]):
    """Freezes every parameter except the factors of the named experts."""
    names = set(names)
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for _, m in lora_modules(model):
        for n, e in m.experts.items():
            if n in names:
                e.A.requires_grad_(True)
                e.B.requires_grad_(True)
                params += [e.A, e.B]
    return params


@torch.no_grad()
def set_chain_clip(model: nn.Module, name: str, k: float = 3.0):
    """
    Sets the clip bound of expert `name` to k * std of the running weight just before it,
    i.e. W0 followed by the other active experts that precede it [MoF-LoRA, Eq. 15].
    """
    for _, m in lora_modules(model):
        W = m.weight.float().reshape(m.weight.shape[0], -1)
        for n, e in m.experts.items():
            if n == name:
                e.clip = k * W.std().item()
                break
            if e.active:
                if e.clip > 0:
                    W = W.clamp(-e.clip, e.clip)
                W = W + e.delta()


@contextmanager
def experts_disabled(model: nn.Module):
    """Temporarily runs the base model (all experts off), e.g. as the DPO reference policy."""
    saved = [(e, e.active) for _, m in lora_modules(model) for e in m.experts.values()]
    try:
        for e, _ in saved:
            e.active = False
        yield model
    finally:
        for e, a in saved:
            e.active = a


@contextmanager
def noise_context(model: nn.Module, noise: Optional[Callable]):
    """Installs a training-time noise preset on every adapted layer."""
    mods = lora_modules(model)
    saved = [m.noise for _, m in mods]
    try:
        for _, m in mods:
            m.noise = noise
        yield model
    finally:
        for (_, m), s in zip(mods, saved):
            m.noise = s


@contextmanager
def fault_context(model: nn.Module, faults: Optional[dict]):
    """Installs per-layer stuck-at overlays {module_name: (flat_idx, values) or callable}."""
    mods = dict(lora_modules(model))
    saved = {n: m.fault for n, m in mods.items()}
    try:
        for n, f in (faults or {}).items():
            mods[n].fault = f
        yield model
    finally:
        for n, m in mods.items():
            m.fault = saved[n]


@torch.no_grad()
def merge_experts(model: nn.Module, names: Optional[Iterable[str]] = None) -> nn.Module:
    """
    Folds experts into the base weights in place (MoF-LoRA design-time merge [MoF-LoRA, Eq. 20])
    and removes the expert banks, leaving plain nn.Linear / nn.Conv2d layers.
    names=None merges the currently active experts.
    """
    for _, m in lora_modules(model):
        if names is not None:
            for n, e in m.experts.items():
                e.active = n in set(names)
        W = m.effective_weight().reshape(m.weight.shape)
        m.weight.data.copy_(W.to(m.weight.dtype))
        cls = nn.Linear if isinstance(m, LoRALinear) else nn.Conv2d
        del m.experts
        for attr in ("noise", "fault", "merge"):
            m.__dict__.pop(attr, None)
        m.__class__ = cls
    return model


@torch.no_grad()
def prune_experts(model: nn.Module, threshold: float, names: Optional[Iterable[str]] = None) -> float:
    """Zero-shot magnitude pruning of LoRA factors [HARD-DPO, Sec. V-K]; returns the pruned fraction."""
    total = pruned = 0
    for _, m in lora_modules(model):
        for n, e in m.experts.items():
            if names is not None and n not in set(names):
                continue
            for p in (e.A, e.B):
                mask = p.abs() < threshold
                p[mask] = 0
                total += p.numel()
                pruned += int(mask.sum())
    return pruned / max(total, 1)


def expert_state_dict(model: nn.Module, names: Optional[Iterable[str]] = None) -> dict:
    """The expert factors and buffers only (what a checkpoint of trained experts needs)."""
    keep = None if names is None else set(names)
    out = {}
    for k, v in model.state_dict().items():
        if ".experts." not in k:
            continue
        expert = k.split(".experts.")[1].split(".")[0]
        if keep is None or expert in keep:
            out[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else dict(v)
    return out


def load_expert_state_dict(model: nn.Module, state: dict):
    """Creates the experts found in `state` (rank from the A factors) and loads their values."""
    mods = dict(lora_modules(model))
    for k, v in state.items():
        if not k.endswith(".A"):
            continue
        mod_name, rest = k.split(".experts.")
        expert = rest.split(".")[0]
        if mod_name not in mods:
            raise KeyError(f"Checkpoint expert for '{mod_name}', which add_lora did not adapt.")
        m = mods[mod_name]
        if expert not in m.experts:
            m.add_expert(expert, rank=v.shape[0], alpha=v.shape[0])
    missing = model.load_state_dict(state, strict=False)
    if missing.unexpected_keys:
        raise KeyError(f"Unexpected keys in expert checkpoint: {missing.unexpected_keys[:5]}")


def load_peft_adapter(model: nn.Module, state: dict, name: str, lora_alpha: float) -> int:
    """
    Imports a PEFT LoRA state dict (e.g. HARD-DPO checkpoints saved with
    get_peft_model_state_dict) as expert `name`. PEFT does not store lora_alpha, so pass the
    value used in training (16 in the HARD-DPO scripts). Returns the number of adapted layers.
    """
    state = state.get("policy_state_dict", state)
    mods = dict(lora_modules(model))
    pairs = {}
    for k, v in state.items():
        if ".lora_A." not in k and ".lora_B." not in k:
            continue
        side = "A" if ".lora_A." in k else "B"
        mod_name = k.split(".lora_" + side + ".")[0]
        for prefix in ("base_model.model.", "base_model."):
            if mod_name.startswith(prefix):
                mod_name = mod_name[len(prefix):]
                break
        pairs.setdefault(mod_name, {})[side] = v
    if not pairs:
        raise ValueError("No lora_A / lora_B tensors found in the PEFT state dict.")
    for mod_name, ab in pairs.items():
        if mod_name not in mods:
            raise KeyError(f"PEFT adapter targets '{mod_name}', which add_lora did not adapt.")
        m = mods[mod_name]
        A, B = ab["A"], ab["B"]
        r = A.shape[0]
        if name not in m.experts:
            m.add_expert(name, r, lora_alpha)
        e = m.experts[name]
        e.A.data.copy_(A.reshape(e.A.shape).float())
        e.B.data.copy_(B.reshape(e.B.shape).float())
        e.scale = lora_alpha / r
    return len(pairs)
