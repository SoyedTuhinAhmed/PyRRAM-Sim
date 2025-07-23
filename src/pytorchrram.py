import torch, torch.nn as nn
import copy
from helper import validate, set_seed
from rram_noise_model import _apply_rram_effects

class PyRRAMSimulator:
    def __init__(
        self,
        pretrained_model: nn.Module,
        G_min: float,
        G_max: float,
        delta_G: float = 0.1e-6,
        differential: bool = True,
        block_size: tuple[int,int] = (64,64),
        analog_bias: bool = False,
        fi_layers: tuple[type, ...] = (nn.Linear, nn.Conv2d),
        device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ):
        self.G_min, self.G_max, self.delta_G = G_min, G_max, delta_G
        self.differential = differential
        self.block_size = block_size
        self.analog_bias = analog_bias
        self.fi_layers = fi_layers
        self.device = device
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

        self.digital_model = copy.deepcopy(pretrained_model).eval()
        self.analog_model = self._to_analog(self.digital_model)
        self.analog_model.to(self.device)

    def _weights_to_conductance(self, weights: torch.Tensor):
        """Helper to convert a weight tensor to target conductance tensor(s)."""
        w_min, w_max = weights.min(), weights.max()
        w_range = (w_max - w_min).item()
        if w_range == 0:
            return (torch.full_like(weights, (self.G_min + self.G_max) / 2),), (w_min, w_range)

        G_range = self.G_max - self.G_min
        if self.differential:
            weights_norm = 2 * (weights - w_min) / w_range - 1
            G_diff_target = weights_norm * G_range
            G_p = (self.G_min + self.G_max + G_diff_target) / 2
            G_n = (self.G_min + self.G_max - G_diff_target) / 2
            return (G_p, G_n), (w_min, w_range)
        else:
            G_target = self.G_min + (weights - w_min) / w_range * G_range
            return (G_target,), (w_min, w_range)

    def _conductance_to_weights(self, conductances: tuple[torch.Tensor, ...], w_min: float, w_range: float):
        """Helper to convert conductance tensor(s) back to a weight tensor."""
        if w_range == 0:
            return torch.full_like(conductances[0], w_min.item())

        G_range = self.G_max - self.G_min
        if self.differential:
            G_p, G_n = conductances
            G_diff = G_p - G_n
            w_norm = G_diff / G_range
            return (w_norm + 1) / 2 * w_range + w_min
        else:
            G_target, = conductances
            return w_min + (G_target - self.G_min) / G_range * w_range
    
    def _to_analog(self, source_model: nn.Module) -> nn.Module:
        """Quantizes a model's weights to their ideal analog representation without noise."""
        analog = copy.deepcopy(source_model).eval().to(self.device)
        with torch.no_grad():
            for m in analog.modules():
                if isinstance(m, self.fi_layers) and hasattr(m, 'weight'):
                    conductances, (w_min, w_range) = self._weights_to_conductance(m.weight.data)
                    quantized_conductances = tuple(
                        _apply_rram_effects(g, self.G_min, self.G_max, self.delta_G, 0.0, 0.0, "both", False, self.block_size)
                        for g in conductances
                    )
                    m.weight.data = self._conductance_to_weights(quantized_conductances, w_min, w_range)
        return analog
    
    def reset_analog_model(self):
        """Resets the analog model to its ideal quantized state."""
        self.analog_model = self._to_analog(self.digital_model)
        self.programming_error_applied = False
        self.conductance_relaxation_applied = False

    def _apply_generic_noise_to_model(self, alpha_ind, alpha_prop, fault_model, clamp_noise):
        """Applies a generic noise model to all relevant layers."""
        with torch.no_grad():
            for m in self.analog_model.modules():
                if isinstance(m, self.fi_layers) and hasattr(m, 'weight'):
                    conductances, (w_min, w_range) = self._weights_to_conductance(m.weight.data)
                    noisy_conductances = tuple(
                        _apply_rram_effects(g, self.G_min, self.G_max, self.delta_G, alpha_ind, alpha_prop, fault_model, clamp_noise, self.block_size)
                        for g in conductances
                    )
                    m.weight.data = self._conductance_to_weights(noisy_conductances, w_min, w_range)

    def apply_programming_error(self, alpha_ind: float = 0.03, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        if self.programming_error_applied:
            print("Warning: Programming error already applied. Reset model to apply again.")
            return
        self.reset_analog_model()
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

    def simulate_programming_error(self, dataloader, criterion, alpha_ind: float = 0.03, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        self.apply_programming_error(alpha_ind, alpha_prop, fault_model, clamp_noise)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def simulate_conductance_relaxation(self, dataloader, criterion, alpha_ind: float = 0.07, alpha_prop: float = 0.0, fault_model: str = "state_independent", clamp_noise: bool = False):
        self.apply_conductance_relaxation(alpha_ind, alpha_prop, fault_model, clamp_noise)
        return validate(self.analog_model, dataloader, criterion, self.device)[-1]

    def simulate_relative_drift(self, dataloader, criterion, relative_drift: float):
        self.apply_relative_drift(relative_drift)
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
