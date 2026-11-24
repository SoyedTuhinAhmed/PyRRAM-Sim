import math

import torch

# Per-cell stuck-at fault codes.
HEALTHY, STUCK_ON, STUCK_OFF, STUCK_INT = 0, 1, 2, 3

BOLTZMANN_EV = 8.617333262e-5  # Boltzmann constant in eV/K


def _rand(shape, device, generator=None):
    """Uniform [0, 1) samples; drawn on the CPU when a dedicated generator is given."""
    if generator is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=generator).to(device)


def _randint(low, high, shape, device, generator=None):
    if generator is None:
        return torch.randint(low, high, shape, device=device)
    return torch.randint(low, high, shape, generator=generator).to(device)


def _randn_like(G, generator=None):
    if generator is None:
        return torch.randn_like(G)
    if generator.device != G.device:
        return torch.randn(G.shape, generator=generator, dtype=G.dtype).to(G.device)
    return torch.randn(G.shape, generator=generator, device=G.device, dtype=G.dtype)


def arrhenius_factor(temperature_C: float, Ea_eV: float, T0_C: float = 25.0) -> float:
    """
    Thermal acceleration factor exp(Ea / kB * (1/T0 - 1/T)) of a noise coefficient,
    alpha(T) = alpha0 * arrhenius_factor(T, Ea, T0) [HAT-FI, Eq. 3]. Temperatures in Celsius.
    """
    if Ea_eV is None:
        raise ValueError("An activation energy Ea_eV is required for temperatures other than T0.")
    T, T0 = temperature_C + 273.15, T0_C + 273.15
    return math.exp(Ea_eV / BOLTZMANN_EV * (1.0 / T0 - 1.0 / T))


def conductance_noise(
    G: torch.Tensor,
    G_max: float,
    alpha_ind: float,
    alpha_prop: float,
    fault_model: str,
    clamp_noise: bool,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Gaussian conductance noise for every cell of G.

    Args:
        G:           conductance tensor of any shape (Siemens).
        G_max:       maximum device conductance.
        alpha_ind:   fraction of G_max used as sigma of the state-independent noise.
        alpha_prop:  fraction of G used as sigma of the state-proportional noise.
        fault_model: one of {"state_independent", "state_proportional", "both"}.
        clamp_noise: clamp the noise at +-3 sigma (the state-independent sigma when both are on).
        generator:   optional generator on G's device (default: the global RNG).

    Returns:
        Noise tensor with the same shape as G.
    """
    if fault_model not in ("state_independent", "state_proportional", "both"):
        raise ValueError(f"Unknown fault_model: {fault_model}")

    noise = torch.zeros_like(G)
    clamp_limit = None

    if fault_model in ("state_independent", "both"):
        if alpha_ind is None:
            raise ValueError("alpha_ind is required for state_independent noise.")
        sigma_ind = alpha_ind * G_max
        noise = noise + _randn_like(G, generator) * sigma_ind
        if clamp_noise:
            clamp_limit = 3.0 * sigma_ind

    if fault_model in ("state_proportional", "both"):
        if alpha_prop is None:
            raise ValueError("alpha_prop is required for state_proportional noise.")
        sigma_prop = alpha_prop * G
        noise = noise + _randn_like(G, generator) * sigma_prop
        if clamp_noise and clamp_limit is None:
            clamp_limit = 3.0 * sigma_prop

    if clamp_limit is not None:
        noise = torch.clamp(noise, min=-clamp_limit, max=clamp_limit)
    return noise


def sample_stuck_at_codes(
    shape,
    p_on: float,
    p_off: float,
    p_int: float,
    device: torch.device,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Draws an i.i.d. stuck-at fault map, one independent draw per physical cell.

    The three fault types are mutually exclusive: a single uniform draw u per cell gives
    STUCK_ON if u < p_on, STUCK_OFF if u < p_on + p_off, STUCK_INT if u < p_on + p_off + p_int,
    and HEALTHY otherwise.

    Returns:
        uint8 tensor of fault codes (HEALTHY, STUCK_ON, STUCK_OFF, STUCK_INT).
    """
    u = _rand(shape, device, generator)
    codes = torch.full(shape, HEALTHY, dtype=torch.uint8, device=device)
    codes = codes.masked_fill(u < p_on, STUCK_ON)
    codes = codes.masked_fill((u >= p_on) & (u < p_on + p_off), STUCK_OFF)
    codes = codes.masked_fill((u >= p_on + p_off) & (u < p_on + p_off + p_int), STUCK_INT)
    return codes


def intermediate_stuck_conductance(
    shape,
    G_min: float,
    G_max: float,
    n_levels: int,
    int_dist: str,
    int_range: tuple[float, float],
    device: torch.device,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Conductances for cells stuck at an intermediate state, one sample per cell of `shape`.

    Args:
        int_dist:  "levels"     -> uniform over the discrete interior levels (stuck-at-L, for MLC);
                   "continuous" -> uniform in the conductance band (undefined 'U' state, for binary).
        int_range: (lo, hi) band as fractions of [G_min, G_max], e.g. (0.0, 1.0) for the full range.
    """
    lo, hi = int_range
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"int_range must satisfy 0 <= lo <= hi <= 1, got {int_range}")
    G_range = G_max - G_min

    if int_dist == "continuous":
        return G_min + G_range * (lo + (hi - lo) * _rand(shape, device, generator))

    if int_dist == "levels":
        top = n_levels - 1
        k_lo = max(1, math.ceil(lo * top - 1e-9))
        k_hi = min(top - 1, math.floor(hi * top + 1e-9))
        if k_lo > k_hi:
            raise ValueError(
                f"No interior level of a {n_levels}-level cell lies in int_range={int_range}; "
                "use int_dist='continuous'."
            )
        k = _randint(k_lo, k_hi + 1, shape, device, generator)
        return G_min + k.to(torch.float32) * (G_range / top)

    raise ValueError(f"Unknown int_dist: {int_dist}")
