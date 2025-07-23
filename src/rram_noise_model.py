import torch

def _apply_rram_effects(
    G_target: torch.Tensor,
    G_min: float,
    G_max: float,
    delta_G: float,
    alpha_ind_param: float,
    alpha_prop_param: float,
    fault_model: str,
    clamp_noise: bool,
    block_size: tuple[int,int]
) -> torch.Tensor:
    """
    Applies quantization and RRAM noise to a target conductance tensor, _block-wise_.

    Args:
        G_target:        torch.Tensor of any shape (>=2D) representing target conductances.
        G_min, G_max:    floats, device conductance range.
        delta_G:         float, quantization step size.
        alpha_ind_param: float or None, fraction of G_max for state-independent noise.
        alpha_prop_param:float or None, fraction of G for state-proportional noise.
        fault_model:     one of {"state_independent","state_proportional","both"}.
        clamp_noise:     bool, whether to clamp noise at ±3σ.
        block_size:      (m, m) tuple: size of each square block for noise injection.

    Returns:
        torch.Tensor of same shape as G_target: quantized + noisy conductances, clamped to [G_min, G_max].
    """
    # Quantize to nearest RRAM level
    levels = torch.round((G_target - G_min) / delta_G)
    G_q = G_min + levels * delta_G

    # Prepare an empty noise tensor
    noise = torch.zeros_like(G_q)

    # If last two dims are at least block-sized, do block-wise injection
    if G_q.ndim >= 2 and block_size is not None:
        # Dimensions of blocks
        block_h, block_w = block_size

        # collapse leading dims so we can loop over 2D slices
        *lead_dims, H, W = G_q.shape
        G_flat = G_q.reshape(-1, H, W)
        noise_flat = torch.zeros_like(G_flat)

        for idx in range(G_flat.size(0)):
            slice_q = G_flat[idx]
            slice_noise = torch.zeros_like(slice_q)

            # tile over non-overlapping blocks
            for i in range(0, H, block_h):
                for j in range(0, W, block_w):
                    block_q = slice_q[i:i+block_h, j:j+block_w]
                    local_noise = torch.zeros_like(block_q)
                    local_clamp = None

                    # State-independent noise
                    if fault_model in ("state_independent", "both"):
                        if alpha_ind_param is None:
                            raise ValueError("alpha_ind is required for state_independent noise.")
                        alpha_ind = alpha_ind_param * G_max
                        local_noise = local_noise + torch.randn_like(block_q) * alpha_ind
                        if clamp_noise:
                            # scalar clamp limit for this block
                            local_clamp = 3.0 * alpha_ind

                    # State-proportional noise
                    if fault_model in ("state_proportional", "both"):
                        if alpha_prop_param is None:
                            raise ValueError("alpha_prop is required for state_proportional noise.")
                        alpha_prop = alpha_prop_param * block_q
                        local_noise = local_noise + torch.randn_like(block_q) * alpha_prop
                        if clamp_noise and local_clamp is None:
                            # per-element clamp limits
                            local_clamp = 3.0 * alpha_prop

                    # Clamp this block’s noise if requested
                    if clamp_noise and local_clamp is not None:
                        local_noise = local_noise.clamp(-local_clamp, local_clamp)

                    # write back
                    slice_noise[i:i+block_h, j:j+block_w] = local_noise

            noise_flat[idx] = slice_noise

        # reshape noise back to original
        noise = noise_flat.reshape_as(G_q)

    else:
        # Fallback for 1D or scalar: original element-wise behavior
        clamp_limit = None

        if fault_model in ("state_independent", "both"):
            if alpha_ind_param is None:
                raise ValueError("alpha_ind is required for state_independent noise.")
            alpha_ind = alpha_ind_param * G_max
            noise = noise + torch.randn_like(G_q) * alpha_ind
            if clamp_noise:
                clamp_limit = 3.0 * alpha_ind

        if fault_model in ("state_proportional", "both"):
            if alpha_prop_param is None:
                raise ValueError("alpha_prop is required for state_proportional noise.")
            alpha_prop = alpha_prop_param * G_q
            noise = noise + torch.randn_like(G_q) * alpha_prop
            if clamp_noise and clamp_limit is None:
                clamp_limit = 3.0 * alpha_prop

        if clamp_noise and clamp_limit is not None:
            noise = noise.clamp(-clamp_limit, clamp_limit)

    # Inject noise and clamp final conductance
    G_noisy = G_q + noise
    return G_noisy.clamp(min=G_min, max=G_max)
