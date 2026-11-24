import copy
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from helper import validate, set_seed
from rram_noise_model import (
    HEALTHY, STUCK_ON, STUCK_OFF, STUCK_INT,
    arrhenius_factor, conductance_noise, sample_stuck_at_codes, intermediate_stuck_conductance,
)

_GOLDEN = 0x9E3779B97F4A7C15  # mixes an operation seed with a layer index (low-memory mode)


@dataclass
class _LayerState:
    """
    Physical state of the RRAM cells storing one layer's weight tensor.

    Conductance tensors have shape (P, S, *weight.shape): P polarity arrays (2 for a
    differential pair [G+, G-], else 1) and S cells per weight (bit-slices, LSB first).
    """
    G: torch.Tensor                # present conductance of every cell
    G_target: torch.Tensor         # programmed target: ideal level, stuck value or compensated partner
    stuck: Optional[torch.Tensor]  # bool mask of stuck cells, None if no faults were injected
    offset: float                  # w_min for affine mappings, 0 for one-sided
    scale: float                   # w_max - w_min for affine mappings, max|w| for one-sided


@dataclass
class _Slot:
    """One tensor programmed into crossbar cells: a layer weight or a LoRA factor."""
    key: str     # unique name: the module name for a layer weight, "<module>.experts.<e>.A" for a factor
    module: str  # name of the owning module
    attr: str    # "weight", or "experts.<expert>.A" / "experts.<expert>.B"


@dataclass
class FaultSet:
    """Stuck-at faults of one layer in the weight domain (the result of a March test)."""
    idx: torch.Tensor       # flat indices of weights with at least one stuck cell
    stuck: torch.Tensor     # (P, S, n) stuck flags of the cells of those weights
    G_stuck: torch.Tensor   # (P, S, n) conductances of those cells (valid where stuck)


class PyRRAMSimulator:
    def __init__(
        self,
        pretrained_model: nn.Module,
        G_min: float,
        G_max: float,
        delta_G: float = 0.1e-6,
        differential: bool = True,
        block_size: tuple = (64, 64),
        analog_bias: bool = False,
        fi_layers: tuple = (nn.Linear, nn.Conv2d),
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        n_levels: Optional[int] = None,
        cells_per_weight: int = 1,
        diff_style: str = "two_sided",
        target_modules: Optional[Sequence[str]] = None,
        exclude_modules: Sequence[str] = (),
        clip_sigma: Optional[float] = None,
        map_adapters: bool = False,
        copy_model: bool = True,
        low_memory: bool = False,
        probe_cells: int = 4096,
    ):
        """
        Args:
            delta_G:          conductance step between adjacent levels; ignored when n_levels is given.
            n_levels:         conductance levels per cell, e.g. 256 for 8-bit MLC or 2 for binary cells.
            cells_per_weight: cells per weight and polarity (bit-slicing, LSB cell first). A weight
                              is quantized to n_levels ** cells_per_weight codes, e.g. n_levels=2,
                              cells_per_weight=8 stores an 8-bit weight on 8 binary cells.
            diff_style:       differential mapping, "two_sided" (G+ + G- constant) or "one_sided"
                              (the idle cell of the pair stays at G_min).
            target_modules:   map only layers whose name ends in one of these (e.g. q_proj, ...,
                              down_proj for LLMs); None maps every fi_layers layer. A weight shared
                              by several layers (tied embeddings) is mapped once.
            exclude_modules:  layer names (last component) that are never mapped, e.g. ("lm_head",).
            clip_sigma:       clip each weight tensor to +-clip_sigma * std before mapping
                              (deployment clipping [HARD-DPO, Sec. IV-D]).
            map_adapters:     also program the factors of active, unmerged LoRA experts into
                              crossbar cells (fully analog LoRA); otherwise they run digitally.
            copy_model:       keep a private copy of pretrained_model as the digital reference;
                              False uses the model itself (saves one copy for large LLMs).
            low_memory:       store no per-cell state: every operation is replayed layer by layer
                              from per-layer seeds, so memory stays at one layer's cells. Chips are
                              statistically equivalent to, not bit-identical with, the default mode.
            probe_cells:      cells per layer sampled by telemetry() (online test).
            block_size:       kept for API compatibility. Noise and faults are i.i.d. per cell, so
                              crossbar tiling does not change their statistics.
        """
        if G_max <= G_min:
            raise ValueError("G_max must be larger than G_min.")
        if n_levels is None:
            if delta_G is None or delta_G <= 0:
                raise ValueError("Provide n_levels or a positive delta_G.")
            n_levels = int(round((G_max - G_min) / delta_G)) + 1
        if n_levels < 2:
            raise ValueError("n_levels must be at least 2.")
        if cells_per_weight < 1:
            raise ValueError("cells_per_weight must be at least 1.")
        if diff_style not in ("two_sided", "one_sided"):
            raise ValueError(f"Unknown diff_style: {diff_style}")
        if clip_sigma is not None and clip_sigma <= 0:
            raise ValueError("clip_sigma must be positive.")

        self.G_min, self.G_max = G_min, G_max
        self.n_levels = n_levels
        self.cells_per_weight = cells_per_weight
        self.delta_G = (G_max - G_min) / (n_levels - 1)
        self.n_codes = n_levels ** cells_per_weight - 1  # largest integer weight code
        if self.n_codes >= 2 ** 53:
            raise ValueError("n_levels ** cells_per_weight is too large.")
        self.differential = differential
        self.diff_style = diff_style
        self.block_size = block_size
        self.analog_bias = analog_bias
        self.fi_layers = fi_layers
        self.device = device
        self.target_modules = None if target_modules is None else tuple(target_modules)
        self.exclude_modules = tuple(exclude_modules)
        self.clip_sigma = clip_sigma
        self.map_adapters = map_adapters
        self.low_memory = low_memory
        self.probe_cells = probe_cells
        # Mapping arithmetic in float64; Apple MPS has no float64, so float32 there (exact for
        # integer codes below 2**24).
        self._fdtype = torch.float32 if torch.device(device).type == "mps" else torch.float64
        if self._fdtype == torch.float32 and self.n_codes >= 2 ** 24:
            raise ValueError("n_levels ** cells_per_weight must stay below 2**24 on MPS.")

        self.temperature_C, self.T0_C, self.Ea_eV = 25.0, 25.0, None
        self.read_alpha = 0.0
        self._read_sigma: dict = {}
        self._read_hooks: list = []

        self.digital_model = copy.deepcopy(pretrained_model).eval() if copy_model else pretrained_model
        self.analog_model = copy.deepcopy(self.digital_model).eval().to(self.device)
        self._states: dict = {}
        self._ops: list = []            # operation log (low-memory replay; stuck params for reprogram)
        self._probe: dict = {}
        self._fault_counts = None
        self.reset_analog_model()

    # ------------------------------------------------------------------ layers

    def _slots(self) -> list:
        slots, seen = [], set()
        for name, m in self.digital_model.named_modules():
            if not isinstance(m, self.fi_layers) or getattr(m, 'weight', None) is None:
                continue
            leaf = name.split(".")[-1]
            if leaf in self.exclude_modules:
                continue
            if self.target_modules is not None and leaf not in self.target_modules:
                continue
            if id(m.weight) in seen:
                continue
            seen.add(id(m.weight))
            slots.append(_Slot(name, name, "weight"))
            if self.map_adapters and hasattr(m, "experts") and not m.merge:
                for en, e in m.experts.items():
                    if e.active:
                        slots += [_Slot(f"{name}.experts.{en}.{f}", name, f"experts.{en}.{f}") for f in "AB"]
        return slots

    def _fi_modules(self, model: nn.Module):
        mods = dict(model.named_modules())
        names = dict.fromkeys(s.module for s in self._slot_list)
        return [(n, mods[n]) for n in names]

    @staticmethod
    def _param(module: nn.Module, attr: str) -> torch.Tensor:
        obj = module
        for part in attr.split("."):
            obj = obj[part] if isinstance(obj, nn.ModuleDict) else getattr(obj, part)
        return obj

    def _source(self, slot: _Slot) -> torch.Tensor:
        """Weight to program for a slot (merged experts included), after deployment clipping."""
        m = self._dmods[slot.module]
        with torch.no_grad():
            if slot.attr == "weight" and hasattr(m, "programmed_weight"):
                w = m.programmed_weight()
            else:
                w = self._param(m, slot.attr).detach()
            w = w.to(self.device, torch.float32)
            if self.clip_sigma is not None:
                lim = self.clip_sigma * w.std().item()
                w = w.clamp(-lim, lim)
        return w

    def _sync_analog(self):
        """Copies the digital model's non-mapped state (biases, norms, expert flags) to the analog model."""
        try:
            self.analog_model.load_state_dict(self.digital_model.state_dict())
        except RuntimeError:  # structure changed, e.g. experts were added after construction
            self._remove_read_hooks()
            self.analog_model = copy.deepcopy(self.digital_model).eval().to(self.device)
        self._amods = dict(self.analog_model.named_modules())
        for m in self._amods.values():
            if hasattr(m, "experts"):
                m.noise = m.fault = None
                for e in m.experts.values():
                    if m.merge and e.active:
                        e.active = False  # folded into the programmed crossbar weights
                    else:
                        e.clip = 0.0      # crossbar weights cannot be clipped digitally
        self._install_read_hooks()

    def _refresh_layout(self):
        self._dmods = dict(self.digital_model.named_modules())
        self._slot_list = self._slots()
        self._sync_analog()

    # ------------------------------------------------------------------ mapping

    def _to_digits(self, code: torch.Tensor) -> torch.Tensor:
        """Splits integer codes into base-n_levels digits, one per cell (LSB first)."""
        if self.cells_per_weight == 1:
            return code.unsqueeze(0)
        L = self.n_levels
        return torch.stack([(code // L ** k) % L for k in range(self.cells_per_weight)])

    def _affine(self, w: torch.Tensor):
        """(offset, scale) of the weight-to-code mapping of a tensor."""
        if self.differential and self.diff_style == "one_sided":
            return 0.0, (w.abs().max().item() or 1.0)
        return w.min().item(), ((w.max() - w.min()).item() or 1.0)

    def _ideal_G(self, w: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
        """Quantized target conductances (P, S, *w.shape) of weights w under a given mapping."""
        w = w.to(self._fdtype)
        Q = self.n_codes
        itype = torch.int32 if Q < 2 ** 31 else torch.int64
        if self.differential and self.diff_style == "one_sided":
            mag = torch.round(w.abs() / scale * Q).clamp_(0, Q).to(itype)
            zero = torch.zeros_like(mag)
            codes = [torch.where(w >= 0, mag, zero), torch.where(w < 0, mag, zero)]
            del mag, zero
        else:
            q = torch.round((w - offset) / scale * Q).clamp_(0, Q).to(itype)
            codes = [q, Q - q] if self.differential else [q]
            del q
        G = torch.empty((len(codes), self.cells_per_weight, *w.shape), dtype=torch.float32, device=w.device)
        for p in range(len(codes)):
            digits = self._to_digits(codes[p])
            codes[p] = None
            G[p] = (self.G_min + digits.to(self._fdtype) * self.delta_G).to(torch.float32)
            del digits
        return G

    def _map_weights(self, weights: torch.Tensor) -> _LayerState:
        """Ideal (fault- and noise-free) quantized mapping of a weight tensor to cell conductances."""
        w = weights.detach().to(self.device, self._fdtype)
        offset, scale = self._affine(w)
        G = self._ideal_G(w, offset, scale)
        return _LayerState(G=G.clone(), G_target=G, stuck=None, offset=offset, scale=scale)

    def _significance(self, ndim: int, device) -> torch.Tensor:
        S = self.cells_per_weight
        sig = torch.tensor([float(self.n_levels ** k) for k in range(S)], dtype=self._fdtype, device=device)
        return sig.view(1, S, *([1] * (ndim - 2)))

    def _G_to_weights(self, G: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
        digits = (G.to(self._fdtype) - self.G_min) / self.delta_G  # analog read, (P, S, ...)
        if self.cells_per_weight == 1:
            v = digits[:, 0]  # (P, ...)
        else:
            v = (digits * self._significance(digits.ndim, digits.device)).sum(dim=1)
        del digits
        Q = self.n_codes
        if not self.differential:
            return offset + v[0] / Q * scale
        if self.diff_style == "two_sided":
            return offset + (v[0] - v[1] + Q) / 2 / Q * scale
        return (v[0] - v[1]) / Q * scale

    def _state_to_weights(self, st: _LayerState, dtype: torch.dtype) -> torch.Tensor:
        """Reads the weight that the present cell conductances represent."""
        return self._G_to_weights(st.G, st.offset, st.scale).to(dtype)

    def _weight_sigma(self, sigma_G: torch.Tensor, scale: float) -> torch.Tensor:
        """Std of the read weight when every cell has independent Gaussian noise of std sigma_G."""
        d = 2.0 if (self.differential and self.diff_style == "two_sided") else 1.0
        sig = self._significance(sigma_G.ndim, sigma_G.device)
        var = ((sigma_G.to(self._fdtype) * sig) ** 2).sum(dim=(0, 1))
        return var.sqrt() * (scale / (d * self.n_codes * self.delta_G))

    def _snap_to_level(self, G: torch.Tensor) -> torch.Tensor:
        G = G.clamp(self.G_min, self.G_max)
        return self.G_min + torch.round((G - self.G_min) / self.delta_G) * self.delta_G

    # ------------------------------------------------------------------ per-layer operations

    def _overlay_stuck(self, G_ideal, stuck, G_stuck, compensate):
        """Programmed targets on a faulty chip: stuck cells keep their value, optionally with
        the healthy partner of a single stuck cell reprogrammed to restore G+ - G-."""
        G_t = torch.where(stuck, G_stuck, G_ideal)
        if compensate:
            target_diff = G_ideal[0] - G_ideal[1]
            pos_only, neg_only = stuck[0] & ~stuck[1], stuck[1] & ~stuck[0]
            new_neg = self._snap_to_level(G_t[0] - target_diff)
            new_pos = self._snap_to_level(G_t[1] + target_diff)
            G_t = torch.stack([
                torch.where(neg_only, new_pos, G_t[0]),
                torch.where(pos_only, new_neg, G_t[1]),
            ])
        return G_t

    def _stuck_layer(self, st: _LayerState, p, generator) -> dict:
        codes = sample_stuck_at_codes(st.G.shape, p["p_on"], p["p_off"], p["p_int"], self.device, generator)
        G_st = st.G_target.clone()
        G_st = G_st.masked_fill(codes == STUCK_ON, self.G_max)
        G_st = G_st.masked_fill(codes == STUCK_OFF, self.G_min)
        if p["p_int"] > 0:
            G_int = intermediate_stuck_conductance(
                st.G.shape, self.G_min, self.G_max, self.n_levels,
                p["int_dist"], p["int_range"], self.device, generator,
            )
            G_st = torch.where(codes == STUCK_INT, G_int.to(G_st.dtype), G_st)
        stuck = codes != HEALTHY
        st.G_target = self._overlay_stuck(st.G_target, stuck, G_st, p["compensate"])
        st.stuck, st.G = stuck, st.G_target.clone()
        return {"cells": codes.numel(), "stuck_on": int((codes == STUCK_ON).sum()),
                "stuck_off": int((codes == STUCK_OFF).sum()), "stuck_int": int((codes == STUCK_INT).sum())}

    def _noise_layer(self, st: _LayerState, p, generator=None):
        G = st.G + conductance_noise(st.G, self.G_max, p["alpha_ind"], p["alpha_prop"],
                                     p["fault_model"], p["clamp_noise"], generator)
        G = G.clamp(self.G_min, self.G_max)
        if st.stuck is not None:
            G = torch.where(st.stuck, st.G_target, G)
        st.G = G

    def _probe_index(self, i: int, st: _LayerState) -> torch.Tensor:
        n = st.G.numel()
        k = min(self.probe_cells, n)
        g = torch.Generator().manual_seed(1_000_003 + i)
        return torch.randint(0, n, (k,), generator=g).to(st.G.device)

    def _capture_probe(self, slot: _Slot, i: int, st: _LayerState, when: str):
        idx = self._probe.setdefault(slot.key, {}).get("idx")
        if idx is None:
            idx = self._probe_index(i, st)
            self._probe[slot.key]["idx"] = idx
        G = st.G.reshape(-1)[idx]
        self._probe[slot.key][when] = G.clone()
        if when == "now":
            self._probe[slot.key]["target"] = st.G_target.reshape(-1)[idx].clone()
            self._probe[slot.key]["stuck"] = (st.stuck.reshape(-1)[idx].clone() if st.stuck is not None
                                              else torch.zeros_like(G, dtype=torch.bool))

    def _seed(self, op_seed: int, i: int) -> int:
        return (op_seed + _GOLDEN * (i + 1)) % (2 ** 63)

    def _device_generator(self, seed: int) -> torch.Generator:
        try:
            g = torch.Generator(device=self.device)
        except RuntimeError:
            g = torch.Generator()
        return g.manual_seed(seed)

    def _replay(self, i: int, slot: _Slot, ops=None) -> _LayerState:
        """Low-memory mode: rebuilds one layer's cells by replaying the operation log."""
        st = self._map_weights(self._source(slot))
        for op in (self._ops if ops is None else ops):
            if op["kind"] == "stuck":
                gen = torch.Generator().manual_seed(self._seed(op["seed"], i))
                self._stuck_layer(st, op["params"], gen)
            else:
                self._noise_layer(st, op["params"], self._device_generator(self._seed(op["seed"], i)))
                if op["stage"] == "prog":
                    self._capture_probe(slot, i, st, "verify")
        return st

    # ------------------------------------------------------------------ writing the analog model

    def _write_slot(self, slot: _Slot, st: _LayerState):
        target = self._param(self._amods[slot.module], slot.attr)
        with torch.no_grad():
            target.data.copy_(self._state_to_weights(st, target.dtype).reshape(target.shape))
        if self.read_alpha > 0:
            sigma_G = self.read_alpha * self.thermal_factor() * st.G
            self._read_sigma[slot.key] = self._weight_sigma(sigma_G, st.scale).to(target.dtype).reshape(target.shape)
        else:
            self._read_sigma.pop(slot.key, None)

    def _write_weights(self):
        with torch.no_grad():
            for i, slot in enumerate(self._slot_list):
                st = self._replay(i, slot) if self.low_memory else self._states[slot.key]
                self._write_slot(slot, st)
                self._capture_probe(slot, i, st, "now")
                if self.low_memory:
                    del st

    def write_ideal_weights(self):
        """Writes the unquantized, noise-free target weights (deployment clipping included) into the
        analog model: the digital model as deployed, e.g. the clean reference for flip rates."""
        with torch.no_grad():
            for slot in self._slot_list:
                target = self._param(self._amods[slot.module], slot.attr)
                target.data.copy_(self._source(slot).to(target.dtype).reshape(target.shape))
        self._read_sigma.clear()

    # ------------------------------------------------------------------ read noise and temperature

    def thermal_factor(self) -> float:
        if self.temperature_C == self.T0_C:
            return 1.0
        return arrhenius_factor(self.temperature_C, self.Ea_eV, self.T0_C)

    def set_temperature(self, temperature_C: float, Ea_eV: Optional[float] = None, T0_C: float = 25.0):
        """
        Sets the operating temperature. Programming error applied afterwards and read noise are
        scaled by exp(Ea / kB * (1/T0 - 1/T)) [HAT-FI, Eq. 3]. Ea (eV) is required whenever
        T != T0; there is no default because it depends on the device stack.
        """
        if temperature_C != T0_C and Ea_eV is None:
            raise ValueError("Ea_eV is required when temperature_C differs from T0_C.")
        self.temperature_C, self.Ea_eV, self.T0_C = float(temperature_C), Ea_eV, float(T0_C)
        if self.read_alpha > 0:
            self._write_weights()

    def set_read_noise(self, alpha_read: float):
        """
        Per-inference read noise: on every forward pass each cell is read as G + N(0, (a * G)^2)
        with a = alpha_read * thermal factor (state-proportional [HAT-FI]). Stored conductances
        do not change. alpha_read=0 turns it off.
        """
        if alpha_read < 0:
            raise ValueError("alpha_read must be non-negative.")
        self.read_alpha = float(alpha_read)
        self._write_weights()
        self._install_read_hooks()

    def _remove_read_hooks(self):
        for h in self._read_hooks:
            h.remove()
        self._read_hooks = []

    def _install_read_hooks(self):
        self._remove_read_hooks()
        if self.read_alpha <= 0:
            return
        by_module = {}
        for slot in self._slot_list:
            by_module.setdefault(slot.module, []).append(slot)
        for name, slots in by_module.items():
            m = self._amods[name]

            def pre(module, args, slots=slots):
                module._read_stash = []
                for s in slots:
                    sigma = self._read_sigma.get(s.key)
                    if sigma is None:
                        continue
                    p = self._param(module, s.attr)
                    module._read_stash.append((p, p.data))
                    p.data = p.data + sigma * torch.randn_like(sigma)

            def post(module, args, output):
                for p, data in getattr(module, "_read_stash", []):
                    p.data = data
                module._read_stash = []

            self._read_hooks += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]

    # ------------------------------------------------------------------ state management

    def reset_analog_model(self):
        """Resets to an ideal, fault-free chip: every cell at its quantized target level."""
        self._refresh_layout()
        self._ops, self._probe, self._fault_counts = [], {}, None
        if not self.low_memory:
            with torch.no_grad():
                self._states = {s.key: self._map_weights(self._source(s)) for s in self._slot_list}
        self._write_weights()
        self.stuck_at_applied = False
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

    def _restore_programmed_state(self):
        """Returns every cell to its programmed target, keeping any stuck-at fault map."""
        self._ops = [op for op in self._ops if op["kind"] == "stuck"]
        if not self.low_memory:
            for st in self._states.values():
                st.G = st.G_target.clone()
        for pr in self._probe.values():
            pr.pop("verify", None)
        self._write_weights()
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

    def reprogram(self):
        """
        Programs the digital model's current target weights into the same chip, e.g. after a
        different set of experts was activated or merged. The permanent stuck-at map is kept
        (with fault-aware compensation if it was enabled); programming error, relaxation and
        drift are cleared and must be applied again.
        """
        old = self._states
        self._refresh_layout()
        stuck_ops = [op for op in self._ops if op["kind"] == "stuck"]
        self._ops = stuck_ops
        self._probe = {}
        if not self.low_memory:
            compensate = bool(stuck_ops) and stuck_ops[-1]["params"]["compensate"]
            states = {}
            with torch.no_grad():
                for s in self._slot_list:
                    st = self._map_weights(self._source(s))
                    prev = old.get(s.key)
                    if prev is not None and prev.stuck is not None and prev.stuck.shape == st.G.shape:
                        st.G_target = self._overlay_stuck(st.G_target, prev.stuck, prev.G_target, compensate)
                        st.stuck, st.G = prev.stuck, st.G_target.clone()
                    states[s.key] = st
            self._states = states
        self._write_weights()
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

    def update_adapters(self):
        """
        Hybrid deployment: copies the digital model's expert switches and factors to the analog
        model without touching the programmed cells. If the set of crossbar-mapped tensors
        changes (map_adapters=True or merged experts), the chip is reprogrammed instead.
        """
        old_keys = [s.key for s in self._slot_list]
        self._refresh_layout()
        if [s.key for s in self._slot_list] != old_keys or any(
                getattr(m, "merge", False) for m in self._dmods.values()):
            self.reprogram()
            return
        self._write_weights()

    def _apply_generic_noise_to_model(self, alpha_ind, alpha_prop, fault_model, clamp_noise, stage="other"):
        """Adds conductance noise to all healthy cells; stuck cells keep their stuck value."""
        params = {"alpha_ind": alpha_ind, "alpha_prop": alpha_prop,
                  "fault_model": fault_model, "clamp_noise": clamp_noise}
        if self.low_memory:
            seed = int(torch.randint(0, 2 ** 62, (1,)).item())
            self._ops.append({"kind": "noise", "stage": stage, "params": params, "seed": seed})
        else:
            with torch.no_grad():
                for i, s in enumerate(self._slot_list):
                    st = self._states[s.key]
                    self._noise_layer(st, params)
                    if stage == "prog":
                        self._capture_probe(s, i, st, "verify")
        self._write_weights()

    # ------------------------------------------------------------------ non-idealities

    def apply_stuck_at_faults(
        self,
        p_on: float = 0.0,
        p_off: float = 0.0,
        p_int: float = 0.0,
        int_dist: str = "auto",
        int_range: tuple = (0.0, 1.0),
        compensate: bool = False,
        fault_seed: Optional[int] = None,
    ) -> dict:
        """
        Injects permanent stuck-at faults into a fresh (fault-free, not yet programmed) chip.

        Every physical cell (each cell of a differential pair and each bit-slice) independently
        becomes stuck at G_max with probability p_on, at G_min with probability p_off, or at an
        intermediate conductance with probability p_int. Stuck cells ignore all later programming
        error, relaxation and drift. Apply this before apply_programming_error.

        Args:
            int_dist:   "levels" (uniform over interior levels, stuck-at-L), "continuous" (uniform
                        in the band, undefined 'U' state) or "auto" ("continuous" for binary cells,
                        "levels" otherwise).
            int_range:  (lo, hi) band for intermediate stuck values, as fractions of [G_min, G_max].
            compensate: differential pairs only: reprogram the healthy partner of a stuck cell so
                        that G+ - G- matches its target as closely as the range and levels allow.
            fault_seed: if given, the fault map comes from its own generator, so the same chip is
                        reproduced regardless of the global RNG state (e.g. across MC runs).

        Returns:
            Realized fault counts and rates over all cells.
        """
        for p in (p_on, p_off, p_int):
            if not 0.0 <= p <= 1.0:
                raise ValueError("Fault probabilities must lie in [0, 1].")
        if p_on + p_off + p_int > 1.0:
            raise ValueError("p_on + p_off + p_int must not exceed 1.")
        if compensate and not self.differential:
            raise ValueError("compensate=True requires differential=True.")
        if int_dist == "auto":
            int_dist = "continuous" if self.n_levels == 2 else "levels"

        if self.stuck_at_applied or self.programming_error_applied:
            print("Warning: stuck-at faults are injected on a fresh chip; resetting the analog model.")
        self.reset_analog_model()

        params = {"p_on": p_on, "p_off": p_off, "p_int": p_int, "int_dist": int_dist,
                  "int_range": tuple(int_range), "compensate": compensate}
        counts = {"cells": 0, "stuck_on": 0, "stuck_off": 0, "stuck_int": 0}

        with torch.no_grad():
            if self.low_memory:
                seed = fault_seed if fault_seed is not None else int(torch.randint(0, 2 ** 62, (1,)).item())
                op = {"kind": "stuck", "params": params, "seed": seed}
                for i, s in enumerate(self._slot_list):
                    st = self._map_weights(self._source(s))
                    gen = torch.Generator().manual_seed(self._seed(seed, i))
                    for k, v in self._stuck_layer(st, params, gen).items():
                        counts[k] += v
                    del st
                self._ops = [op]
            else:
                generator = torch.Generator().manual_seed(fault_seed) if fault_seed is not None else None
                for s in self._slot_list:
                    for k, v in self._stuck_layer(self._states[s.key], params, generator).items():
                        counts[k] += v
                self._ops = [{"kind": "stuck", "params": params, "seed": fault_seed}]

        self._write_weights()
        self.stuck_at_applied = True
        n = max(counts["cells"], 1)
        counts.update({f"rate_{k}": counts[k] / n for k in ("stuck_on", "stuck_off", "stuck_int")})
        self._fault_counts = counts
        return counts

    def apply_programming_error(self, alpha_ind: float = 0.03, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        """Write-and-verify error: Gaussian conductance noise (scaled by the thermal factor)."""
        if self.programming_error_applied:
            print("Warning: Programming error already applied. Reset model to apply again.")
            return
        self._restore_programmed_state()
        f = self.thermal_factor()
        self._apply_generic_noise_to_model(alpha_ind * f, alpha_prop * f, fault_model, clamp_noise, stage="prog")
        self.programming_error_applied = True

    def apply_conductance_relaxation(self, alpha_ind: float = 0.07, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        if not self.programming_error_applied:
            print("Applying default programming error first.")
            self.apply_programming_error()
        if self.conductance_relaxation_applied:
            print("Warning: Conductance relaxation already applied.")
            return
        self._apply_generic_noise_to_model(alpha_ind, alpha_prop, fault_model, clamp_noise, stage="relax")
        self.conductance_relaxation_applied = True

    def apply_relative_drift(self, relative_drift: float):
        """
        Applies drift based on the state-proportional Gaussian model from the DoRA paper.
        This drift is applied to the current state of the analog model.
        """
        if not self.conductance_relaxation_applied:
            print("Warning: Applying default programming error and relaxation first.")
            self.apply_conductance_relaxation()

        if relative_drift <= 0:
            return

        self._apply_generic_noise_to_model(alpha_ind=0.0, alpha_prop=relative_drift,
                                           fault_model="state_proportional", clamp_noise=False, stage="drift")

    # ------------------------------------------------------------------ test and telemetry

    def fault_map(self) -> dict:
        """
        March-test equivalent: the stuck-at faults of every mapped layer weight, as
        {module_name: FaultSet}. A weight is faulty if any of its cells is stuck.
        """
        out = {}
        stuck_ops = [op for op in self._ops if op["kind"] == "stuck"]
        if not stuck_ops:
            return out
        with torch.no_grad():
            for i, s in enumerate(self._slot_list):
                if s.attr != "weight":
                    continue
                st = self._replay(i, s, ops=stuck_ops) if self.low_memory else self._states[s.key]
                stuck = st.stuck.reshape(st.stuck.shape[0], st.stuck.shape[1], -1)
                idx = stuck.any(dim=(0, 1)).nonzero().squeeze(1)
                G = st.G_target.reshape(stuck.shape)
                out[s.module] = FaultSet(idx=idx, stuck=stuck[:, :, idx].clone(), G_stuck=G[:, :, idx].clone())
        return out

    def realize_faulty(self, faults: FaultSet, W: torch.Tensor) -> torch.Tensor:
        """
        Weights that a faulty layer realizes at its faulty positions when target weights W are
        programmed (stuck cells keep their value; partners follow the mapping and compensation).
        """
        with torch.no_grad():
            w = W.detach().to(self.device, torch.float32)
            if self.clip_sigma is not None:
                lim = self.clip_sigma * w.std().item()
                w = w.clamp(-lim, lim)
            offset, scale = self._affine(w.to(self._fdtype))
            sub = w.reshape(-1)[faults.idx]
            G_ideal = self._ideal_G(sub, offset, scale)
            stuck_ops = [op for op in self._ops if op["kind"] == "stuck"]
            compensate = bool(stuck_ops) and stuck_ops[-1]["params"]["compensate"]
            G = self._overlay_stuck(G_ideal, faults.stuck, faults.G_stuck, compensate)
            return self._G_to_weights(G, offset, scale).to(torch.float32)

    def telemetry(self, n_reads: int = 16) -> dict:
        """
        Online test of the simulated chip, from a fixed sample of probe_cells cells per layer:

            saf_rate      fraction of stuck cells (March test over all cells)
            drift_alpha   sqrt(sum dG^2 / sum G^2) of healthy cells since the last programming
                          (write-verify) step, or since the targets if none was simulated; equals
                          the relative drift for state-proportional drift
            read_alpha    relative std of n_reads repeated reads (alpha_read * thermal factor)
            temperature_C on-chip temperature sensor reading
        """
        num = den = rnum = rden = 0.0
        for pr in self._probe.values():
            now = pr.get("now")
            if now is None:
                continue
            healthy = ~pr["stuck"]
            ref = pr.get("verify", pr.get("target"))
            if ref is not None:
                v = ref[healthy].to(self._fdtype)
                num += ((now[healthy].to(self._fdtype) - v) ** 2).sum().item()
                den += (v ** 2).sum().item()
            if self.read_alpha > 0 and n_reads > 1:
                g = now.to(self._fdtype)
                reads = g + self.read_alpha * self.thermal_factor() * g * torch.randn(
                    n_reads, *g.shape, dtype=self._fdtype, device=g.device)
                rnum += reads.var(dim=0).sum().item()
                rden += (g ** 2).sum().item()
        fc = self._fault_counts
        return {
            "saf_rate": 0.0 if fc is None else (fc["stuck_on"] + fc["stuck_off"] + fc["stuck_int"]) / max(fc["cells"], 1),
            "drift_alpha": math.sqrt(num / den) if den > 0 else 0.0,
            "read_alpha": math.sqrt(rnum / rden) if rden > 0 else 0.0,
            "temperature_C": self.temperature_C,
        }

    def weight_noise_moments(self, W: torch.Tensor, alpha_ind: float = 0.0, alpha_prop: float = 0.0):
        """
        (mean, std) of the weight error that programming error causes for target weights W under
        this mapping, per weight: independent Gaussian cell noise clamped to [G_min, G_max]
        (censored normal per cell, combined over polarities and bit-slices). The mean is the
        clamping bias of cells at a bound, e.g. binary cells. Used by the simulator-exact
        training noise preset.
        """
        with torch.no_grad():
            w = W.detach().to(self.device, torch.float32)
            offset, scale = self._affine(w.to(self._fdtype))
            G = self._ideal_G(w, offset, scale).to(self._fdtype)
            f = self.thermal_factor()
            sigma = ((alpha_ind * f * self.G_max) ** 2 + (alpha_prop * f * G) ** 2).sqrt()
            s = sigma.clamp(min=1e-30)
            a, b = (self.G_min - G) / s, (self.G_max - G) / s
            pdf = lambda z: torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            cdf = lambda z: 0.5 * (1 + torch.erf(z / math.sqrt(2)))
            # c = clamp(G + e, G_min, G_max) - G with e ~ N(0, s^2): first and second moments
            m1 = s * (pdf(a) - pdf(b) + a * cdf(a) + b * (1 - cdf(b)))
            m2 = s ** 2 * (cdf(b) - cdf(a) - b * pdf(b) + a * pdf(a) + a * a * cdf(a) + b * b * (1 - cdf(b)))
            var = (m2 - m1 ** 2).clamp(min=0)
            m1 = torch.where(sigma > 0, m1, torch.zeros_like(m1))
            var = torch.where(sigma > 0, var, torch.zeros_like(var))

            d = 2.0 if (self.differential and self.diff_style == "two_sided") else 1.0
            k = scale / (d * self.n_codes * self.delta_G)
            sig = self._significance(G.ndim, G.device)
            pol = torch.tensor([1.0, -1.0][:G.shape[0]], dtype=self._fdtype, device=G.device)
            pol = pol.view(-1, *([1] * (G.ndim - 1)))
            mean = k * (pol * sig * m1).sum(dim=(0, 1))
            std = k * ((sig ** 2) * var).sum(dim=(0, 1)).sqrt()
            return mean.to(torch.float32), std.to(torch.float32)

    def weight_noise_std(self, W: torch.Tensor, alpha_ind: float = 0.0, alpha_prop: float = 0.0) -> torch.Tensor:
        """Root-mean-square weight error of programming error, sqrt(mean^2 + std^2) of
        weight_noise_moments."""
        mean, std = self.weight_noise_moments(W, alpha_ind, alpha_prop)
        return (mean ** 2 + std ** 2).sqrt()

    # ------------------------------------------------------------------ evaluation

    def simulate_stuck_at_faults(self, dataloader, criterion, **stuck_at_params):
        self.apply_stuck_at_faults(**stuck_at_params)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def simulate_programming_error(self, dataloader, criterion, alpha_ind: float = 0.03, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        self.apply_programming_error(alpha_ind, alpha_prop, fault_model, clamp_noise)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def simulate_conductance_relaxation(self, dataloader, criterion, alpha_ind: float = 0.07, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        self.apply_conductance_relaxation(alpha_ind, alpha_prop, fault_model, clamp_noise)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def simulate_relative_drift(self, dataloader, criterion, relative_drift: float):
        self.apply_relative_drift(relative_drift)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def program(self, stuck_at_params=None, programming_error_params=None,
                relaxation_params=None, drift_params=None):
        """Builds a fresh chip: stuck-at -> programming -> relaxation -> drift."""
        self.reset_analog_model()
        if stuck_at_params:
            self.apply_stuck_at_faults(**stuck_at_params)
        if programming_error_params:
            self.apply_programming_error(**programming_error_params)
        if relaxation_params:
            self.apply_conductance_relaxation(**relaxation_params)
        if drift_params:
            self.apply_relative_drift(**drift_params)
        return self.analog_model

    def simulate(self, dataloader, criterion, stuck_at_params=None, programming_error_params=None,
                 relaxation_params=None, drift_params=None):
        """Runs the full pipeline (stuck-at -> programming -> relaxation -> drift) on a fresh chip."""
        self.program(stuck_at_params, programming_error_params, relaxation_params, drift_params)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def MC_simulate(self, mode: str, dataloader, criterion, number_runs: int = 100, seed: int = 42, **kwargs):
        """
        Monte-Carlo simulation: calls self.apply_<mode> each run, seeds RNG,
        and returns (mean_acc, std_acc).
        """
        set_seed(seed)
        accs = []

        if mode == "relative_drift":
            apply_fn = self.apply_relative_drift
            params = {"relative_drift": kwargs.get("relative_drift", 0.0)}
        elif mode == "programming_error":
            apply_fn = self.apply_programming_error
            params = {k: v for k, v in kwargs.items() if k in ["alpha_ind", "alpha_prop", "fault_model", "clamp_noise"]}
        elif mode == "conductance_relaxation":
            apply_fn = self.apply_conductance_relaxation
            params = {k: v for k, v in kwargs.items() if k in ["alpha_ind", "alpha_prop", "fault_model", "clamp_noise"]}
        elif mode == "stuck_at":
            apply_fn = self.apply_stuck_at_faults
            params = {k: v for k, v in kwargs.items() if k in ["p_on", "p_off", "p_int", "int_dist", "int_range", "compensate", "fault_seed"]}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for run in range(number_runs):
            set_seed(seed + run)
            self.reset_analog_model()
            apply_fn(**params)

            _, acc = validate(self.analog_model, dataloader, criterion, self.device)
            accs.append(acc)

        accs = torch.tensor(accs, device=self.device)
        return accs.mean().item(), accs.std().item()
