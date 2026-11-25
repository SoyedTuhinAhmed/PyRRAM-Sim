"""
Training-time weight perturbations for hardware-aware LoRA training.

Each preset is a callable that receives an adapted layer (lora.LoRALinear / LoRAConv2d) and
returns its perturbed effective weight as a 2-D float32 tensor (out, fan_in). Presets are
installed with lora.noise_context and sampled anew on every forward pass. They follow the
published training surrogates; evaluation always uses the conductance-domain models of
PyRRAMSimulator. SimulatorNoise is the exception: it injects the exact weight-domain
equivalent of the simulator's programming error.
"""
from typing import Optional

import torch


def _std(t: torch.Tensor) -> float:
    return t.detach().float().std().item() if t.numel() > 1 else 0.0


def truncated_normal(shape, sigma: float, k: float, device) -> torch.Tensor:
    """Samples N(0, sigma^2) truncated to [-k sigma, k sigma]."""
    if sigma <= 0:
        return torch.zeros(shape, device=device)
    out = torch.empty(shape, device=device, dtype=torch.float32)
    return torch.nn.init.trunc_normal_(out, mean=0.0, std=sigma, a=-k * sigma, b=k * sigma)


def _base(m) -> torch.Tensor:
    return m.weight.float().reshape(m.weight.shape[0], -1)


class LogNormalNoise:
    """
    HARD-DPO training noise [HARD-DPO, Eqs. 25-26]: W~ = W * exp(eps), eps ~ N(0, sigma^2),
    clipped to +-clip_k * std(W). Sign-preserving, magnitude-scaled.

    target="merged": perturbs the adapted weight W0 + dW, as written in the paper.
    target="base":   perturbs only W0 and adds the clean LoRA update, as the released
                     noise_robust_dpo.py does (fused_noise=False).
    """

    def __init__(self, sigma: float = 0.008, clip_k: Optional[float] = 3.0, target: str = "merged"):
        if target not in ("merged", "base"):
            raise ValueError(f"Unknown target: {target}")
        self.sigma, self.clip_k, self.target = sigma, clip_k, target

    def _perturb(self, W: torch.Tensor) -> torch.Tensor:
        lim = None if self.clip_k is None else self.clip_k * _std(W)
        W = W * torch.exp(torch.randn_like(W) * self.sigma)
        return W if lim is None else W.clamp(-lim, lim)

    def __call__(self, m) -> torch.Tensor:
        if self.target == "merged":
            return self._perturb(m.effective_weight())
        return m._apply_experts(self._perturb(_base(m)).detach())


class TruncNormalNoise:
    """
    Programming-error expert noise: additive truncated Gaussian, sigma = alpha * std(X),
    truncated at +-k sigma.

    target="effective": X = W0 + sum dW, the effective weight [HAT-FI, Eq. 1].
    target="factors":   independent noise on W0 and on the A and B factors of every active
                        expert, W~ = W~0 + s A~ B~ [MoF-LoRA, Eqs. 10-12].
    """

    def __init__(self, alpha: float, target: str = "effective", k: float = 3.0):
        if target not in ("effective", "factors"):
            raise ValueError(f"Unknown target: {target}")
        self.alpha, self.target, self.k = alpha, target, k

    def _noise(self, X: torch.Tensor) -> torch.Tensor:
        return truncated_normal(X.shape, self.alpha * _std(X), self.k, X.device)

    def __call__(self, m) -> torch.Tensor:
        if self.target == "effective":
            W = m.effective_weight()
            return W + self._noise(W)
        W0 = _base(m)
        factors = {n: (e.A + self._noise(e.A), e.B + self._noise(e.B)) for n, e in m.active_experts()}
        return m._apply_experts(W0 + self._noise(W0), factors)


class ProportionalNoise:
    """State-proportional noise sigma = alpha * |W| on the effective weight (drift and read-noise
    experts [HAT-FI, Sec. II-A])."""

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, m) -> torch.Tensor:
        W = m.effective_weight()
        return W * (1.0 + self.alpha * torch.randn_like(W))


class SimulatorNoise:
    """
    Simulator-exact programming error: a weight-domain Gaussian with the per-weight mean and
    std that PyRRAMSimulator's conductance-domain programming error induces for the current
    effective weight (mapping, differential style, bit-slicing, levels, clamping at the
    conductance bounds and deployment clipping of `sim`).
    """

    def __init__(self, sim, alpha_ind: float = 0.0, alpha_prop: float = 0.0):
        self.sim, self.alpha_ind, self.alpha_prop = sim, alpha_ind, alpha_prop

    def __call__(self, m) -> torch.Tensor:
        W = m.effective_weight()
        if self.sim.clip_sigma is not None:
            lim = self.sim.clip_sigma * _std(W)
            W = W.clamp(-lim, lim)
        mean, std = self.sim.weight_noise_moments(W, self.alpha_ind, self.alpha_prop)
        return W + mean.to(W.device) + std.to(W.device) * torch.randn_like(W)


class AbsMaxGaussianNoise:
    """
    Additive Gaussian noise on the effective weight with sigma = std * max|W_eff| of the
    tensor, i.e. relative to the weight that the tile normalization maps to G_max
    (noise-aware LoRA of the COINS'26 space-CiM case study, std = 0.1; hardware-aware
    training practice of IBM aihwkit).
    """

    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, m) -> torch.Tensor:
        W = m.effective_weight()
        return W + torch.randn_like(W) * (self.std * W.detach().abs().max())
