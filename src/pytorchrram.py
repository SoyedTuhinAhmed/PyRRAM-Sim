import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from helper import validate, set_seed
from rram_noise_model import (
    HEALTHY, STUCK_ON, STUCK_OFF, STUCK_INT,
    conductance_noise, sample_stuck_at_codes, intermediate_stuck_conductance,
)


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


class PyRRAMSimulator:
    def __init__(
        self,
        pretrained_model: nn.Module,
        G_min: float,
        G_max: float,
        delta_G: float = 0.1e-6,
        differential: bool = True,
        block_size: tuple[int, int] = (64, 64),
        analog_bias: bool = False,
        fi_layers: tuple[type, ...] = (nn.Linear, nn.Conv2d),
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        n_levels: Optional[int] = None,
        cells_per_weight: int = 1,
        diff_style: str = "two_sided",
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

        self.digital_model = copy.deepcopy(pretrained_model).eval()
        self.analog_model = copy.deepcopy(self.digital_model).eval().to(self.device)
        self._states: dict[str, _LayerState] = {}
        self.reset_analog_model()

    # ------------------------------------------------------------------ mapping

    def _fi_modules(self, model: nn.Module):
        return [
            (name, m) for name, m in model.named_modules()
            if isinstance(m, self.fi_layers) and getattr(m, 'weight', None) is not None
        ]

    def _to_digits(self, code: torch.Tensor) -> torch.Tensor:
        """Splits integer codes into base-n_levels digits, one per cell (LSB first)."""
        L = self.n_levels
        return torch.stack([(code // L ** k) % L for k in range(self.cells_per_weight)])

    def _map_weights(self, weights: torch.Tensor) -> _LayerState:
        """Ideal (fault- and noise-free) quantized mapping of a weight tensor to cell conductances."""
        w = weights.detach().to(self.device, torch.float64)
        Q = self.n_codes

        if self.differential and self.diff_style == "one_sided":
            offset, scale = 0.0, (w.abs().max().item() or 1.0)
            mag = torch.round(w.abs() / scale * Q).to(torch.int64)
            zero = torch.zeros_like(mag)
            codes = [torch.where(w >= 0, mag, zero), torch.where(w < 0, mag, zero)]
        else:
            offset, scale = w.min().item(), ((w.max() - w.min()).item() or 1.0)
            q = torch.round((w - offset) / scale * Q).to(torch.int64)
            codes = [q, Q - q] if self.differential else [q]

        levels = torch.stack([self._to_digits(c) for c in codes])  # (P, S, *shape)
        G = (self.G_min + levels.to(torch.float64) * self.delta_G).to(torch.float32)
        return _LayerState(G=G.clone(), G_target=G, stuck=None, offset=offset, scale=scale)

    def _state_to_weights(self, st: _LayerState, dtype: torch.dtype) -> torch.Tensor:
        """Reads the weight that the present cell conductances represent."""
        S = self.cells_per_weight
        digits = (st.G.to(torch.float64) - self.G_min) / self.delta_G  # analog read, (P, S, *shape)
        sig = torch.tensor([float(self.n_levels ** k) for k in range(S)],
                           dtype=torch.float64, device=digits.device)
        sig = sig.view(1, S, *([1] * (digits.ndim - 2)))
        v = (digits * sig).sum(dim=1)  # (P, *shape)
        Q = self.n_codes

        if not self.differential:
            w = st.offset + v[0] / Q * st.scale
        elif self.diff_style == "two_sided":
            w = st.offset + (v[0] - v[1] + Q) / 2 / Q * st.scale
        else:
            w = (v[0] - v[1]) / Q * st.scale
        return w.to(dtype)

    def _write_weights(self):
        with torch.no_grad():
            for name, m in self._fi_modules(self.analog_model):
                m.weight.data = self._state_to_weights(self._states[name], m.weight.dtype)

    def _snap_to_level(self, G: torch.Tensor) -> torch.Tensor:
        G = G.clamp(self.G_min, self.G_max)
        return self.G_min + torch.round((G - self.G_min) / self.delta_G) * self.delta_G

    # ------------------------------------------------------------------ state management

    def reset_analog_model(self):
        """Resets to an ideal, fault-free chip: every cell at its quantized target level."""
        with torch.no_grad():
            for name, m in self._fi_modules(self.digital_model):
                self._states[name] = self._map_weights(m.weight.data)
        self._write_weights()
        self.stuck_at_applied = False
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

    def _restore_programmed_state(self):
        """Returns every cell to its programmed target, keeping any stuck-at fault map."""
        for st in self._states.values():
            st.G = st.G_target.clone()
        self._write_weights()
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

    def _apply_generic_noise_to_model(self, alpha_ind, alpha_prop, fault_model, clamp_noise):
        """Adds conductance noise to all healthy cells; stuck cells keep their stuck value."""
        with torch.no_grad():
            for st in self._states.values():
                G = st.G + conductance_noise(st.G, self.G_max, alpha_ind, alpha_prop, fault_model, clamp_noise)
                G = G.clamp(self.G_min, self.G_max)
                if st.stuck is not None:
                    G = torch.where(st.stuck, st.G_target, G)
                st.G = G
        self._write_weights()

    # ------------------------------------------------------------------ non-idealities

    def apply_stuck_at_faults(
        self,
        p_on: float = 0.0,
        p_off: float = 0.0,
        p_int: float = 0.0,
        int_dist: str = "auto",
        int_range: tuple[float, float] = (0.0, 1.0),
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

        generator = torch.Generator().manual_seed(fault_seed) if fault_seed is not None else None
        counts = {"cells": 0, "stuck_on": 0, "stuck_off": 0, "stuck_int": 0}

        with torch.no_grad():
            for st in self._states.values():
                codes = sample_stuck_at_codes(st.G.shape, p_on, p_off, p_int, self.device, generator)
                G_t = st.G_target.clone()
                G_t = G_t.masked_fill(codes == STUCK_ON, self.G_max)
                G_t = G_t.masked_fill(codes == STUCK_OFF, self.G_min)
                if p_int > 0:
                    G_int = intermediate_stuck_conductance(
                        st.G.shape, self.G_min, self.G_max, self.n_levels,
                        int_dist, int_range, self.device, generator,
                    )
                    G_t = torch.where(codes == STUCK_INT, G_int.to(G_t.dtype), G_t)
                stuck = codes != HEALTHY

                if compensate:
                    target_diff = st.G_target[0] - st.G_target[1]
                    pos_only, neg_only = stuck[0] & ~stuck[1], stuck[1] & ~stuck[0]
                    new_neg = self._snap_to_level(G_t[0] - target_diff)
                    new_pos = self._snap_to_level(G_t[1] + target_diff)
                    G_t = torch.stack([
                        torch.where(neg_only, new_pos, G_t[0]),
                        torch.where(pos_only, new_neg, G_t[1]),
                    ])

                st.G_target, st.stuck, st.G = G_t, stuck, G_t.clone()
                counts["cells"] += codes.numel()
                counts["stuck_on"] += int((codes == STUCK_ON).sum())
                counts["stuck_off"] += int((codes == STUCK_OFF).sum())
                counts["stuck_int"] += int((codes == STUCK_INT).sum())

        self._write_weights()
        self.stuck_at_applied = True
        n = max(counts["cells"], 1)
        counts.update({f"rate_{k}": counts[k] / n for k in ("stuck_on", "stuck_off", "stuck_int")})
        return counts

    def apply_programming_error(self, alpha_ind: float = 0.03, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        if self.programming_error_applied:
            print("Warning: Programming error already applied. Reset model to apply again.")
            return
        self._restore_programmed_state()
        self._apply_generic_noise_to_model(alpha_ind, alpha_prop, fault_model, clamp_noise)
        self.programming_error_applied = True

    def apply_conductance_relaxation(self, alpha_ind: float = 0.07, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        if not self.programming_error_applied:
            print("Applying default programming error first.")
            self.apply_programming_error()
        if self.conductance_relaxation_applied:
            print("Warning: Conductance relaxation already applied.")
            return
        self._apply_generic_noise_to_model(alpha_ind, alpha_prop, fault_model, clamp_noise)
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

        self._apply_generic_noise_to_model(alpha_ind=0.0, alpha_prop=relative_drift, fault_model="state_proportional", clamp_noise=False)

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

    def simulate(self, dataloader, criterion, stuck_at_params=None, programming_error_params=None,
                 relaxation_params=None, drift_params=None):
        """Runs the full pipeline (stuck-at -> programming -> relaxation -> drift) on a fresh chip."""
        self.reset_analog_model()
        if stuck_at_params:
            self.apply_stuck_at_faults(**stuck_at_params)
        if programming_error_params:
            self.apply_programming_error(**programming_error_params)
        if relaxation_params:
            self.apply_conductance_relaxation(**relaxation_params)
        if drift_params:
            self.apply_relative_drift(**drift_params)
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
