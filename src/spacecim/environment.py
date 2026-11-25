"""
Space-environment effects on a programmed crossbar [COINS'26, Sec. IV-C]:

* Single-event upsets (SEU): between two reads separated by dt days, each of a tile's
  N conductances is hit at rate lambda (upsets/device/day), N_hits ~ Poisson(lambda N dt).
  A hit redraws the conductance uniformly in [G_min, G_max] with probability 0.8 and
  otherwise leaves it stuck at one of the two bounds ("redraw"), or removes a random
  fraction of its charge ("partial_discharge", charge-trap cells). Upsets accumulate: a
  corrupted device is never reprogrammed.
* Total ionizing dose (TID): dose(t) = offset + rate * t / 1 yr. Broadening widens the
  read (and optionally programming) sigmas by 1 + k_tid * dose; charge loss shifts the
  stored conductance, as a fraction (``multiplicative``, ``toward_gmin``) or as a fixed
  quantum k_shift * dose * G_max per cell clamped at G_min (``absolute``).
* Temperature: drift exponent nu(T) = nu * T_op / T_ref, read noise * (T_op/T_ref)^0.5,
  and the state-dependent conduction activation G(T) = G exp(-E_a/k_B (1/T_op - 1/T_ref))
  with E_a interpolated between 0.30 eV (amorphous, G = G_min) and 0.02 eV (crystalline,
  G = G_max), clamped at the room-temperature full scale.

Port of the case-study code of the COINS'26 paper (space_noise.py).
"""
from typing import List, Optional

import torch
from torch import Tensor

SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY
BOLTZMANN_EV = 8.617333262e-5


class TID:
    SHIFT_MODES = ("multiplicative", "toward_gmin", "absolute")

    def __init__(self, enabled: bool = True, dose_rate_krad_per_year: float = 1.0, dose_offset_krad: float = 0.0,
                 dose_at_programming_krad: float = 0.0, k_tid: float = 0.05, k_shift: float = 0.0,
                 shift_mode: str = "absolute", apply_to_programming: bool = True, apply_to_read: bool = True):
        if shift_mode not in self.SHIFT_MODES:
            raise ValueError(f"shift_mode must be one of {self.SHIFT_MODES}")
        self.enabled = bool(enabled)
        self.rate, self.offset = float(dose_rate_krad_per_year), float(dose_offset_krad)
        self.at_programming = float(dose_at_programming_krad)
        self.k_tid, self.k_shift, self.shift_mode = float(k_tid), float(k_shift), shift_mode
        self.apply_to_programming, self.apply_to_read = bool(apply_to_programming), bool(apply_to_read)

    def dose(self, t: float) -> float:
        return self.offset + self.rate * float(t) / SECONDS_PER_YEAR

    def read_sigma_scale(self, t: float) -> float:
        return 1.0 + self.k_tid * self.dose(t) if (self.enabled and self.apply_to_read) else 1.0

    def programming_sigma_scale(self) -> float:
        return 1.0 + self.k_tid * self.at_programming if (self.enabled and self.apply_to_programming) else 1.0

    def shift(self, g: Tensor, t: float, g_min: float, g_max: float) -> Tensor:
        if not self.enabled or self.k_shift == 0.0:
            return g
        if self.shift_mode == "absolute":
            return (g - self.k_shift * self.dose(t) * g_max).clamp(min=g_min)
        factor = max(0.0, 1.0 - self.k_shift * self.dose(t))
        if factor == 1.0:
            return g
        if self.shift_mode == "toward_gmin":
            return g_min + (g - g_min) * factor
        return g * factor


class Thermal:
    def __init__(self, enabled: bool = False, T_ref_C: float = 25.0, T_op_C: float = 25.0,
                 drift_activation: str = "linear", read_noise_exponent: float = 0.5, state_activation: bool = True,
                 E_a_amorphous_eV: float = 0.30, E_a_crystalline_eV: float = 0.02, clamp_to_g_max: bool = True):
        if drift_activation not in ("linear", "none"):
            raise ValueError("drift_activation must be linear or none")
        self.enabled = bool(enabled)
        self.T_ref, self.T_op = float(T_ref_C) + 273.15, float(T_op_C) + 273.15
        if self.T_ref <= 0 or self.T_op <= 0:
            raise ValueError("Temperatures must be above absolute zero")
        self.drift_activation, self.read_noise_exponent = drift_activation, float(read_noise_exponent)
        self.state_activation = bool(state_activation)
        self.E_a_amorphous, self.E_a_crystalline = float(E_a_amorphous_eV), float(E_a_crystalline_eV)
        self.clamp_to_g_max = bool(clamp_to_g_max)

    @property
    def active(self) -> bool:
        return self.enabled and abs(self.T_op - self.T_ref) > 1e-9

    def drift_scale(self) -> float:
        return self.T_op / self.T_ref if (self.active and self.drift_activation == "linear") else 1.0

    def read_noise_scale(self) -> float:
        return (self.T_op / self.T_ref) ** self.read_noise_exponent if self.active else 1.0

    def activate(self, g: Tensor, g_state: Tensor, g_min: float, g_max: float) -> Tensor:
        """State-dependent gain; the activation energy follows the programmed state."""
        if not self.active or not self.state_activation:
            return g
        frac = ((g_state - g_min) / max(g_max - g_min, 1e-12)).clamp(0.0, 1.0)
        e_a = self.E_a_amorphous + frac * (self.E_a_crystalline - self.E_a_amorphous)
        out = g * torch.exp(-(e_a / BOLTZMANN_EV) * (1.0 / self.T_op - 1.0 / self.T_ref))
        return out.clamp(0.0, g_max) if self.clamp_to_g_max else out.clamp(min=0.0)


class SEU:
    """Accumulated upsets of one tile; one boolean mask and value vector per slice."""

    MODES = ("redraw", "partial_discharge")

    def __init__(self, rate_per_device_day: float = 0.0, p_redraw: float = 0.8, mode: str = "redraw",
                 discharge_max: float = 1.0, seed: int = 0):
        if mode not in self.MODES:
            raise ValueError(f"SEU mode must be one of {self.MODES}")
        self.rate, self.p_redraw, self.mode, self.discharge_max = float(rate_per_device_day), float(p_redraw), mode, float(discharge_max)
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.masks: List[Tensor] = []
        self.values: List[Tensor] = []
        self.hits = 0
        self.last_t = 0.0
        self._gen: Optional[torch.Generator] = None

    def _generator(self, device) -> torch.Generator:
        if self._gen is None or self._gen.device != torch.device(device):
            try:
                self._gen = torch.Generator(device=device).manual_seed(self.seed)
            except RuntimeError:
                self._gen = torch.Generator().manual_seed(self.seed)
        return self._gen

    def n_devices(self) -> int:
        return sum(int(m.numel()) for m in self.masks)

    @torch.no_grad()
    def accumulate(self, slices: List[Tensor], t: float, g_min: float, g_max: float):
        """Samples the upsets of the interval (last read, t]."""
        if not self.masks:
            self.masks = [torch.zeros(g.numel(), dtype=torch.bool, device=g.device) for g in slices]
            self.values = [torch.zeros(g.numel(), dtype=g.dtype, device=g.device) for g in slices]
        dt_days = (float(t) - self.last_t) / SECONDS_PER_DAY
        self.last_t = max(self.last_t, float(t))
        if self.rate <= 0.0 or dt_days <= 0.0:
            return
        device = slices[0].device
        gen = self._generator(device)
        n = self.n_devices()
        lam = torch.tensor([self.rate * n * dt_days], dtype=torch.float64, device=gen.device)
        k = int(torch.poisson(lam, generator=gen).item())
        if k <= 0:
            return
        idx = torch.randint(0, n, (k,), generator=gen, device=gen.device).to(device)
        if self.mode == "partial_discharge":
            struck = torch.cat([g.reshape(-1) for g in slices])[idx]
            frac = (self.discharge_max * torch.rand(k, generator=gen, device=gen.device)).to(device)
            new = g_min + (struck.float() - g_min) * (1.0 - frac)
        else:
            u = torch.rand(k, generator=gen, device=gen.device).to(device)
            redraw = u < self.p_redraw
            new = torch.empty(k, device=device)
            uniform = torch.rand(int(redraw.sum()), generator=gen, device=gen.device).to(device)
            new[redraw] = g_min + (g_max - g_min) * uniform
            n_stuck = int((~redraw).sum())
            if n_stuck:
                at_max = (torch.rand(n_stuck, generator=gen, device=gen.device) < 0.5).to(device)
                new[~redraw] = torch.where(at_max, torch.full_like(new[~redraw], g_max),
                                           torch.full_like(new[~redraw], g_min))
        offset = 0
        for mask, values in zip(self.masks, self.values):
            size = mask.numel()
            sel = (idx >= offset) & (idx < offset + size)
            if bool(sel.any()):
                local = idx[sel] - offset
                mask[local] = True
                values[local] = new[sel].to(values.dtype)
            offset += size
        self.hits += k

    def apply(self, slices: List[Tensor]) -> List[Tensor]:
        if not self.masks:
            return slices
        out = []
        for g, mask, values in zip(slices, self.masks, self.values):
            flat = g.reshape(-1).clone()
            flat[mask] = values[mask].to(flat.dtype)
            out.append(flat.view_as(g))
        return out
