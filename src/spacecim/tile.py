"""
Analog crossbar layers with an IBM-aihwkit-style periphery [COINS'26, Sec. IV-A/B].

``AnalogLinear`` replaces an ``nn.Linear``: the weight is split into tiles of at most
256 x 256 (balanced splits, as aihwkit's ``TileModuleArray``), each tile is scaled so its
largest |w| maps to 1 (``weight_scaling_omega``), and the bias stays digital. A forward
pass through one tile follows aihwkit's torch tile (``AnalogMVM``, bound management off):

    x_s = x / max|x|                              ABS_MAX noise management, per input row
    x_q = clip(round(x_s / d_in) d_in, +-1)       DAC, d_in = 2 / (2^dac_bits - 2)
    y   = x_q W^T + w_noise ||x_q|| e1 + out_noise e2      per-read weight and output noise
          (drawn as one Gaussian of variance w_noise^2 ||x_q||^2 + out_noise^2)
    y_q = clip(round(y / d_out) d_out, +-12)      ADC, d_out = 24 / (2^adc_bits - 2)
    out = y_q * max|x| * s_tile * alpha_tile      s_tile: weight scale, alpha: drift comp.

Global drift compensation applies the identity matrix as a probe right after programming
and at every later read, and scales the tile's outputs by alpha = ref / cur, where each
readout is the mean |response| (aihwkit ``GlobalDriftCompensation``). Gradients through a
tile (for on-orbit recalibration of digital parameters) are those of the ideal product
with the tile's present weights, as in aihwkit's inference tiles (straight-through for
the converters).
"""
import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .devices import DeviceModel
from .environment import SEU, TID, Thermal


@dataclass
class Periphery:
    dac_bits: int = 6
    adc_bits: int = 8
    inp_bound: float = 1.0
    out_bound: float = 12.0
    w_noise: float = 0.02          # additive weight read noise, fraction of the tile's w_max
    out_noise: float = 0.02
    tile_size: Tuple[int, int] = (256, 256)   # (max inputs, max outputs)
    weight_scaling_omega: float = 1.0

    @property
    def d_in(self) -> float:
        return 2.0 * self.inp_bound / (2 ** self.dac_bits - 2)

    @property
    def d_out(self) -> float:
        return 2.0 * self.out_bound / (2 ** self.adc_bits - 2)


def _mvm(x: Tensor, W: Tensor, gain: Tensor, d_in: float, inp_bound: float, w_noise2: float, out_noise2: float,
         d_out: float, out_bound: float, adc: bool) -> Tensor:
    """IO-managed analog MVM. The weight read noise and the output noise are independent
    Gaussians and are drawn as one with the combined variance."""
    nm = x.abs().amax(dim=-1, keepdim=True)
    all_zero = (nm == 0).all()          # aihwkit returns exact zeros for an all-zero input
    nm = torch.where(nm > 0, nm, torch.ones_like(nm))
    xq = (torch.round(x / (nm * d_in)) * d_in).clamp(-inp_bound, inp_bound)
    y = xq @ W.T
    if w_noise2 > 0 or out_noise2 > 0:
        sigma = torch.sqrt(w_noise2 * xq.square().sum(dim=-1, keepdim=True) + out_noise2)
        y = y + torch.randn_like(y) * sigma
    if adc:
        y = torch.round(y / d_out) * d_out
    y = y.clamp(-out_bound, out_bound) * (nm * gain)
    return torch.where(all_zero, torch.zeros_like(y), y)


_MVM = _mvm


def use_compile(enabled: bool = True):
    """Fuses the tile MVM with torch.compile (about 2x faster on GPU; the first calls
    compile). Statistically identical to the eager path."""
    global _MVM
    _MVM = torch.compile(_mvm, dynamic=True) if enabled else _mvm


def split_sizes(size: int, max_size: int) -> List[int]:
    n = (size + max_size - 1) // max_size
    base, extra = divmod(size, n)
    return [base + (i < extra) for i in range(n)]


class Tile:
    """One physical crossbar tile: device model, environment state and periphery."""

    def __init__(self, weight: Tensor, device: DeviceModel, periphery: Periphery, tid: TID, thermal: Thermal,
                 seu: SEU, drift_compensation: bool):
        w = weight.detach().float()
        self.scale = (w.abs().max() / periphery.weight_scaling_omega).clamp(min=1e-12)
        self.reference = w / self.scale
        self.device, self.periphery, self.tid, self.thermal, self.seu = device, periphery, tid, thermal, seu
        self.drift_compensation = drift_compensation
        self._base = (device.prog_noise_scale, device.read_noise_scale, device.drift_scale)
        self.programmed: Optional[Tensor] = None
        self.nu: List[Optional[Tensor]] = []
        self.weight = self.reference.clone()
        self.alpha = torch.ones((), device=w.device)
        self.baseline: Optional[Tensor] = None

    def apply_fn(self, fn):
        """Applies a tensor function (e.g. a device move from Module.to) to the tile state."""
        for name in ("scale", "reference", "weight", "alpha", "programmed", "baseline"):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                setattr(self, name, fn(value))
        self.nu = [None if v is None else fn(v) for v in self.nu]
        self.seu.masks = [fn(m) for m in self.seu.masks]
        self.seu.values = [fn(v) for v in self.seu.values]

    # ---------------------------------------------------------------- periphery

    @torch.no_grad()
    def raw_forward(self, x: Tensor, gain=None) -> Tensor:
        """Analog MVM in normalized tile units, times ``gain`` (weight scale and drift
        compensation)."""
        p = self.periphery
        g = torch.ones((), device=x.device, dtype=x.dtype) if gain is None else gain
        return _MVM(x, self.weight, g, p.d_in, p.inp_bound, p.w_noise ** 2, p.out_noise ** 2,
                    p.d_out, p.out_bound, p.adc_bits > 0)

    def effective_weight(self) -> Tensor:
        return self.weight * (self.scale * self.alpha)

    def forward(self, x: Tensor) -> Tensor:
        out = self.raw_forward(x, self.scale * self.alpha)
        if torch.is_grad_enabled() and x.requires_grad:
            ideal = x @ self.effective_weight().T
            out = ideal + (out - ideal).detach()
        return out

    @torch.no_grad()
    def _probe(self) -> Tensor:
        eye = torch.eye(self.weight.shape[1], device=self.weight.device, dtype=self.weight.dtype)
        return self.raw_forward(eye).abs().mean().clamp(min=1e-4)

    # ---------------------------------------------------------------- programming and time

    @torch.no_grad()
    def program(self):
        """Writes the reference weights once (t = 0): programming error and drift
        coefficients are drawn here; upsets are cleared."""
        prog, read, drift = self._base
        self.device.prog_noise_scale = prog * self.tid.programming_sigma_scale()
        self.device.read_noise_scale = read
        self.device.drift_scale = drift * self.thermal.drift_scale()
        self.programmed, self.nu = self.device.program(self.reference)
        self.weight = self.programmed.clone()
        self.seu.reset()
        self.alpha = torch.ones((), device=self.weight.device)
        if self.drift_compensation:
            self.baseline = self._probe()

    @torch.no_grad()
    def drift_to(self, t: float):
        """Weights read at time t after programming: drift and read noise, temperature,
        dose, then the accumulated upsets."""
        if self.programmed is None:
            self.program()
        _, read, _ = self._base
        self.device.read_noise_scale = read * self.tid.read_sigma_scale(t) * self.thermal.read_noise_scale()
        dev = self.device
        g_slices, params = dev.converter.to_conductances(self.programmed)
        self.seu.accumulate(g_slices, t, dev.g_min, dev.g_max)
        out = []
        for g, nu in zip(g_slices, self.nu):
            gd = dev.drift_conductance(g, nu, t)
            gd = self.thermal.activate(gd, g, dev.g_min, dev.g_max)
            gd = self.tid.shift(gd, t, dev.g_min, dev.g_max)
            out.append(gd)
        out = self.seu.apply(out)
        self.weight = dev.converter.to_weights(out, params).to(self.reference.dtype)
        if self.drift_compensation:
            self.alpha = self.baseline / self._probe()


class AnalogLinear(nn.Module):
    """Tiled analog replacement of an ``nn.Linear`` with a digital bias."""

    def __init__(self, linear: nn.Linear, tiles_factory, periphery: Periphery, first_tile_index: int):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(),
                                                                   requires_grad=linear.bias.requires_grad)
        W = linear.weight.detach()
        self.in_sizes = split_sizes(self.in_features, periphery.tile_size[0])
        self.out_sizes = split_sizes(self.out_features, periphery.tile_size[1])
        self.tiles: List[List[Tile]] = []
        index, i0 = first_tile_index, 0
        for n_in in self.in_sizes:
            row, o0 = [], 0
            for n_out in self.out_sizes:
                row.append(tiles_factory(W[o0:o0 + n_out, i0:i0 + n_in], index))
                index += 1
                o0 += n_out
            self.tiles.append(row)
            i0 += n_in
        self.n_tiles = index - first_tile_index

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for t in self.all_tiles():
            t.apply_fn(fn)
        return self

    def all_tiles(self) -> List[Tile]:
        return [t for row in self.tiles for t in row]

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        out = None
        for chunk, row in zip(torch.split(x2, self.in_sizes, dim=-1), self.tiles):
            y = torch.cat([t.forward(chunk) for t in row], dim=-1)
            out = y if out is None else out + y
        out = out.reshape(*shape[:-1], self.out_features)
        return out if self.bias is None else out + self.bias

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, tiles={len(self.in_sizes)}x{len(self.out_sizes)}"


def _set_submodule(root: nn.Module, name: str, module: nn.Module):
    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


def convert_to_analog(model: nn.Module, device: DeviceModel, periphery: Optional[Periphery] = None,
                      tid: Optional[TID] = None, thermal: Optional[Thermal] = None, seu_rate: float = 0.0,
                      seu_mode: str = "redraw", seu_p_redraw: float = 0.8, seu_discharge_max: float = 1.0,
                      drift_compensation: bool = False, exclude: Sequence[str] = ("head",), seed: int = 0) -> nn.Module:
    """
    Replaces every ``nn.Linear`` of ``model`` (in place) except those whose name starts with
    an entry of ``exclude`` by ``AnalogLinear``. Every tile gets its own copy of the device
    model and its own upset stream, seeded from (seed, tile index).
    """
    periphery = periphery or Periphery()
    tid = tid or TID(enabled=False)
    thermal = thermal or Thermal(enabled=False)

    def factory(W, index):
        seu_seed = (seed * 1_000_003 + index * 7919 + 17) % (2 ** 63)
        return Tile(W, copy.deepcopy(device), periphery, tid, thermal,
                    SEU(seu_rate, seu_p_redraw, seu_mode, seu_discharge_max, seu_seed), drift_compensation)

    targets = [(n, m) for n, m in model.named_modules()
               if type(m) is nn.Linear and not any(n == e or n.startswith(e + ".") for e in exclude)]
    index = 0
    for name, lin in targets:
        analog = AnalogLinear(lin, factory, periphery, index)
        index += analog.n_tiles
        _set_submodule(model, name, analog)
    return model


def analog_layers(model: nn.Module) -> List[AnalogLinear]:
    return [m for m in model.modules() if isinstance(m, AnalogLinear)]


def analog_tiles(model: nn.Module) -> List[Tile]:
    return [t for layer in analog_layers(model) for t in layer.all_tiles()]


def program(model: nn.Module):
    for t in analog_tiles(model):
        t.program()


def drift_to(model: nn.Module, t: float):
    for tile in analog_tiles(model):
        tile.drift_to(t)


def seu_totals(model: nn.Module) -> dict:
    tiles = analog_tiles(model)
    return {"seu_hits": sum(t.seu.hits for t in tiles), "n_devices": sum(t.seu.n_devices() for t in tiles)}


def analog_weight_count(model: nn.Module) -> int:
    return sum(t.reference.numel() for t in analog_tiles(model))
