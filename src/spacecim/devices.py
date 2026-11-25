"""
Time-dependent NVM device models for analog compute-in-memory missions.

Each model maps normalized tile weights (max |w| = 1 after per-tile weight scaling) to
cell conductances in uS, draws programming error, and evaluates drift and accumulated
read noise at a time t after programming. The equations are ports of measured fits:

* PCM: IBM aihwkit 1.1.0 ``PCMLikeNoiseModel`` (Nandakumar et al., ICECS 2019):
  programming noise polynomial in G/G_max, per-device drift exponent nu, and 1/f read
  noise accumulated since programming.
* ReRAM (CMO/HfOx): aihwkit 1.1.0 ``ReRamCMONoiseModel``: linear programming noise for a
  0.2 % or 2 % program-and-verify acceptance range, absolute log(t) conductance decay
  with growing spread, and read noise. ``mapping="single"`` uses one device per weight
  (offset-binary affine map), ``mapping="differential"`` a pair with the 9 uS floor on
  both legs.
* SONOS: Sandia CrossSim ``SONOS.py`` (Agrawal et al., IEDM 2022, 128K 40 nm devices),
  measured for 0-5 days, with an opt-in retention-law projection beyond the window.

The aihwkit models are MIT-licensed (c) IBM; the equations and coefficients are
reproduced here with attribution. As in aihwkit, conductances are re-derived from the
programmed weights at every read, so the idle leg of a differential pair reads back at
the floor and the programming error of both legs is folded into the stored weight.
"""
import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

SECONDS_PER_DAY = 86400.0
_ZERO_CLIP = 1e-7


# ---------------------------------------------------------------------------- converters

class PairConverter:
    """Differential pair: G+ carries max(w, 0), G- carries max(-w, 0), both above g_min
    (aihwkit ``SinglePairConductanceConverter``)."""

    devices_per_weight = 2

    def __init__(self, g_max: float, g_min: float = 0.0):
        self.g_max, self.g_min = float(g_max), float(g_min)

    def to_conductances(self, w: Tensor) -> Tuple[List[Tensor], dict]:
        ratio = (self.g_max - self.g_min) / w.abs().max().clamp(min=_ZERO_CLIP)
        s = w * ratio
        return [s.clamp(0.0, self.g_max) + self.g_min, (-s).clamp(0.0, self.g_max) + self.g_min], {"ratio": ratio}

    def to_weights(self, g: List[Tensor], params: dict) -> Tensor:
        return ((g[0] - self.g_min) - (g[1] - self.g_min)) / params["ratio"]


class SingleDeviceConverter:
    """One device per weight, offset-binary affine map of [w_min, w_max] onto
    [g_min, g_max] (aihwkit ``SingleDeviceConductanceConverter``)."""

    devices_per_weight = 1

    def __init__(self, g_max: float, g_min: float):
        self.g_max, self.g_min = float(g_max), float(g_min)

    def to_conductances(self, w: Tensor) -> Tuple[List[Tensor], dict]:
        w_min, w_max = w.min(), w.max()
        ratio = (self.g_max - self.g_min) / (w_max - w_min).clamp(min=_ZERO_CLIP)
        return [(w - w_min) * ratio + self.g_min], {"ratio": ratio, "min": w_min}

    def to_weights(self, g: List[Tensor], params: dict) -> Tensor:
        return params["min"] + (g[0] - self.g_min) / params["ratio"]


# ---------------------------------------------------------------------------- devices

class DeviceModel:
    """Interface: programming noise, drift parameters and drift + read noise at t."""

    name = "device"

    def __init__(self, converter, prog_noise_scale=1.0, read_noise_scale=1.0, drift_scale=1.0):
        self.converter = converter
        self.prog_noise_scale = float(prog_noise_scale)
        self.read_noise_scale = float(read_noise_scale)
        self.drift_scale = float(drift_scale)

    @property
    def g_min(self) -> float:
        return self.converter.g_min

    @property
    def g_max(self) -> float:
        return self.converter.g_max

    def program_conductance(self, g: Tensor) -> Tensor:
        raise NotImplementedError

    def drift_parameters(self, g: Tensor) -> Optional[Tensor]:
        return None

    def drift_conductance(self, g: Tensor, nu: Optional[Tensor], t: float) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def program(self, w: Tensor) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Programmed weights and per-slice drift parameters (fixed at programming)."""
        g_target, params = self.converter.to_conductances(w)
        g_prog = [self.program_conductance(g) for g in g_target]
        nus = [self.drift_parameters(g) for g in g_target]
        return self.converter.to_weights(g_prog, params), nus


class PCM(DeviceModel):
    """Phase-change memory, aihwkit ``PCMLikeNoiseModel`` (Nandakumar et al. 2019)."""

    name = "pcm"

    def __init__(self, g_max: float = 25.0, t_read: float = 250e-9, t_0: float = 20.0, **scales):
        super().__init__(PairConverter(g_max, 0.0), **scales)
        self.prog_coeff = [0.26348, 1.9650, -1.1731]  # uS, fitted at g_max = 25 uS
        self.prog_coeff_g_max_reference = 25.0
        self.t_read, self.t_0 = float(t_read), float(t_0)

    def program_conductance(self, g: Tensor) -> Tensor:
        x = g / self.g_max
        sig = self.prog_coeff[0] + self.prog_coeff[1] * x + self.prog_coeff[2] * x * x
        sig = sig * (self.g_max / self.prog_coeff_g_max_reference)
        return (g + self.prog_noise_scale * sig * torch.randn_like(g)).clamp(min=0.0)

    def drift_parameters(self, g: Tensor) -> Tensor:
        g_rel = (g / self.g_max).abs().clamp(min=_ZERO_CLIP)
        mu = (-0.0155 * torch.log(g_rel) + 0.0244).clamp(0.049, 0.1)
        sig = (-0.0125 * torch.log(g_rel) - 0.0059).clamp(0.008, 0.045)
        return (mu + sig * torch.randn_like(g_rel)).abs().clamp(min=0.0) * self.drift_scale

    def drift_conductance(self, g: Tensor, nu: Tensor, t: float) -> Tensor:
        tt = float(t) + self.t_0
        g_drift = g * (tt / self.t_0) ** (-nu) if tt > self.t_0 else g
        q_s = (0.0088 / ((g.abs() / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
        sig = q_s * math.sqrt(math.log((tt + self.t_read) / (2 * self.t_read)))
        return (g_drift + g_drift.abs() * self.read_noise_scale * sig * torch.randn_like(g)).clamp(min=0.0)


class ReRAMCMO(DeviceModel):
    """CMO/HfOx ReRAM, aihwkit ``ReRamCMONoiseModel`` (IBM Research Zurich)."""

    name = "reram_cmo"
    PROG_COEFF = {0.2: [0.00081107, 0.00106879], 2.0: [0.0112185391, 0.01129027418]}  # c0 + c1 G, uS
    DECAY_MEAN, DECAY_STD = -0.08900206, (0.04201137, 0.41183342)
    READ_K, READ_T = 0.0277316483, 1e-6

    def __init__(self, g_max: float = 25.0, g_min: float = 9.0, mapping: str = "single",
                 acceptance_range: float = 0.2, **scales):
        if mapping not in ("single", "differential"):
            raise ValueError("mapping must be 'single' or 'differential'")
        if float(acceptance_range) not in self.PROG_COEFF:
            raise ValueError(f"acceptance_range must be one of the fitted ranges {sorted(self.PROG_COEFF)} (%)")
        conv = SingleDeviceConverter(g_max, g_min) if mapping == "single" else PairConverter(g_max, g_min)
        super().__init__(conv, **scales)
        self.mapping = mapping
        self.coeff = self.PROG_COEFF[float(acceptance_range)]

    def program_conductance(self, g: Tensor) -> Tensor:
        sig = self.coeff[0] + self.coeff[1] * g
        return g + sig * torch.randn_like(g) * self.prog_noise_scale

    def drift_conductance(self, g: Tensor, nu, t: float) -> Tensor:
        if t == 0:
            return g
        lt = math.log(t)
        g_mean = g + self.DECAY_MEAN * lt * self.drift_scale
        sigma = self.DECAY_STD[0] * lt + self.DECAY_STD[1]
        g_drift = g_mean + torch.randn_like(g) * sigma * self.drift_scale
        sigma_read = self.READ_K * torch.log10(g_drift) * math.sqrt(math.log((t + self.READ_T) / (2 * self.READ_T)))
        g_final = g_drift + sigma_read * torch.randn_like(g) * self.read_noise_scale
        return g_final.clamp(min=self.g_min, max=self.g_max)


# SONOS tables, verbatim from Sandia CrossSim SONOS.py (cell current in nA, t in days).
_SONOS_DRIFT_POLY = [
    [0.0] * 11,
    [0.172684, 0.0572552, -0.000685715, 3.50271e-06, -9.37682e-09, 1.40575e-11,
     -1.19803e-14, 5.41293e-18, -8.85448e-22, -1.66869e-25, 5.9518e-29],
    [0.250712, 0.0662195, -0.000683218, 2.99442e-06, -6.78255e-09, 7.70482e-12,
     -3.04983e-15, -2.15919e-18, 2.94617e-21, -1.23359e-24, 1.85329e-28],
    [-0.01753, 0.0976883, -0.000963791, 4.55551e-06, -1.22391e-08, 1.93786e-11,
     -1.85899e-14, 1.07659e-17, -3.58211e-21, 5.96785e-25, -3.3041e-29],
    [0.0403338, 0.103111, -0.0010474, 4.83961e-06, -1.28132e-08, 1.99484e-11,
     -1.85139e-14, 1.00008e-17, -2.84944e-21, 3.01008e-25, 1.18747e-29],
    [0.319927, 0.106596, -0.00104054, 4.5029e-06, -1.10352e-08, 1.56477e-11,
     -1.27592e-14, 5.45513e-18, -7.26841e-22, -2.43893e-25, 7.14608e-29],
]
_SONOS_AB = [[19.88665, 176.3115], [25.59102, 186.85544], [28.72753, 193.56834],
             [31.04318, 198.90332], [33.76214, 204.77023], [34.57909, 207.34309]]
_SONOS_READ_AB = (12.58037, 215.36557)
_SONOS_LOG_T0 = {"mean": 3.70, "sigma_a": 1.00, "sigma_b": 1.75}     # days, fitted to the window
_SONOS_POWER = {"mean": 0.755, "sigma_a": 0.555, "sigma_b": 0.635}


def _sat_sigma(i: Tensor, a: float, b: float) -> Tensor:
    return (a - a * torch.exp(-i / b)).clamp(min=0.0)


def _interp(table, t_days: float):
    lo = int(t_days)
    if lo >= len(table) - 1:
        return list(table[-1])
    f = t_days - lo
    return [x + f * (y - x) for x, y in zip(table[lo], table[lo + 1])]


class SONOS(DeviceModel):
    """
    40 nm SONOS charge-trap memory, CrossSim fit (Agrawal et al., IEDM 2022). Measured for
    0-5 days; ``extrapolate=True`` projects the retention past the window under
    ``retention_law`` ("log", "power" or "linear"): the current dependence stays at its
    5-day form and only the time axis is scaled, s(5 d) = 1. Programming error is frozen at
    programming and only the spread accumulated since then is added at a read.
    """

    name = "sonos"
    MEASURED_DAYS = 5.0

    def __init__(self, g_max: float = 16.0, g_min: float = 0.0, v_read: float = 0.1,
                 extrapolate: bool = False, retention_law: str = "log", max_days: float = 400.0, **scales):
        super().__init__(PairConverter(g_max, g_min), **scales)
        if retention_law not in ("log", "power", "linear"):
            raise ValueError("retention_law must be log, power or linear")
        self.v_read = float(v_read)
        self.extrapolate, self.retention_law, self.max_days = bool(extrapolate), retention_law, float(max_days)
        if self.to_current(g_max) >= 1800.0:
            raise ValueError("g_max * v_read must stay below CrossSim's 1800 nA drift-model limit")

    def to_current(self, g):
        return g * self.v_read * 1e3

    def to_conductance(self, i):
        return i / (self.v_read * 1e3)

    def _scale(self, t_days: float, term: str) -> float:
        if self.retention_law == "linear":
            return t_days / self.MEASURED_DAYS
        if self.retention_law == "power":
            return (t_days / self.MEASURED_DAYS) ** _SONOS_POWER[term]
        t0 = _SONOS_LOG_T0[term]
        return math.log1p(t_days / t0) / math.log1p(self.MEASURED_DAYS / t0)

    def program_conductance(self, g: Tensor) -> Tensor:
        i = self.to_current(g)
        sig = _sat_sigma(i, *_SONOS_AB[0])
        return self.to_conductance((i + self.prog_noise_scale * sig * torch.randn_like(i)).clamp(min=0.0))

    def drift_conductance(self, g: Tensor, nu, t: float) -> Tensor:
        t_days = float(t) / SECONDS_PER_DAY
        projecting = t_days > self.MEASURED_DAYS
        if projecting and not self.extrapolate:
            raise ValueError(f"SONOS is measured for 0-5 days; t = {t_days:.3g} d needs extrapolate=True")
        if projecting and t_days > self.max_days:
            raise ValueError(f"SONOS projection is capped at {self.max_days:g} days")
        i = self.to_current(g)
        sigma_0 = _sat_sigma(i, *_SONOS_AB[0])
        if t_days > 0:
            poly = _interp(_SONOS_DRIFT_POLY, t_days)
            term, drift = torch.ones_like(i), torch.zeros_like(i)
            for c in poly:
                drift = drift + c * term
                term = term * i
            if projecting:
                drift = drift * self._scale(t_days, "mean")
            i = i + self.drift_scale * drift
            a_t, b_t = _interp(_SONOS_AB, t_days)
            if projecting:
                a0, b0 = _SONOS_AB[0]
                a_t = a0 + (a_t - a0) * self._scale(t_days, "sigma_a")
                b_t = b0 + (b_t - b0) * self._scale(t_days, "sigma_b")
            extra = (_sat_sigma(i, a_t, b_t).square() - sigma_0.square()).clamp(min=0.0).sqrt()
            i = i + self.prog_noise_scale * extra * torch.randn_like(i)
        i = i + self.read_noise_scale * _sat_sigma(i, *_SONOS_READ_AB) * torch.randn_like(i)
        return self.to_conductance(i.clamp(min=0.0))


DEVICES = {
    "pcm": lambda **kw: PCM(**kw),
    "reram_cmo": lambda **kw: ReRAMCMO(mapping="single", **kw),
    "reram_cmo_diff": lambda **kw: ReRAMCMO(mapping="differential", **kw),
    "sonos": lambda **kw: SONOS(**kw),
}


def build_device(name: str, **kwargs) -> DeviceModel:
    if name not in DEVICES:
        raise ValueError(f"Unknown device {name!r}; choose one of {sorted(DEVICES)}")
    return DEVICES[name](**kwargs)
