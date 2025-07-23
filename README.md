# PyRRAM-Sim: A Toolkit for RRAM Non-Ideality Simulation

PyRRAM-Sim is a lightweight, PyTorch-based simulation tool designed for researchers and engineers working on Deep Neural Networks (DNNs) with RRAM-based Compute-in-Memory (CIM) architectures. It provides an efficient, modular framework to simulate key hardware non-idealities, allowing for the evaluation of model robustness before hardware deployment.

This toolkit simulates the sequential, cumulative effects of programming errors, short-term conductance relaxation, and long-term, state-proportional conductance drift. The models are based on phenomena described in leading academic research to ensure hardware-accurate simulation.

---

## Key Features

* **Efficient State Management**: Simulates the physical state of RRAM by operating directly on conductance values, avoiding inefficient back-and-forth conversions.
* **Sequential Non-Idealities**: Accurately models the workflow of applying one-time programming errors, followed by short-term relaxation and long-term drift.
* **Hardware-Accurate Drift Model**: Implements a state-proportional drift model based on findings from recent literature.
* **Differential Pair Support**: Natively supports the simulation of differential RRAM cells (`W = Gp - Gn`) for signed weight representation.
* **PyTorch Integration**: Seamlessly integrates with existing PyTorch models and validation pipelines.

---

## Installation

The toolkit is contained within the `FastPyTorchRRAM` class. Simply add the class file to your project. Ensure you have the required dependencies installed:

```bash
pip install torch
```

## Usage
The following example demonstrates the standard workflow for simulating all non-idealities on a pretrained PyTorch model.

```bash
import torch
import torch.nn as nn

# Assume 'pretrained_model', 'dataloader', and 'criterion' are already defined
# Assume helper functions '_apply_rram_effects' and 'validate' are available

# 1. Initialize the simulator with your model and hardware specs
rram_sim = FastPyTorchRRAM(
    pretrained_model=my_model,
    G_min=1e-6,         # Minimum conductance (Siemens)
    G_max=100e-6,       # Maximum conductance (Siemens)
    differential=True
)

# 2. Apply a one-time programming error to the ideal conductances
# This simulates the initial write operation to the chip.
prog_error_params = {'alpha_ind': 0.03, 'fault_model': 'state_independent'}
rram_sim.apply_programming_error(**prog_error_params)

# 3. Apply short-term conductance relaxation
# This simulates the rapid drift that occurs shortly after programming.
relax_params = {'alpha_ind': 0.07, 'fault_model': 'state_independent'}
rram_sim.apply_conductance_relaxation(**relax_params)

# 4. Apply long-term relative drift
# This uses the hardware-accurate model from the DoRA paper.
# The paper suggests drift is typically < 20% of the target conductance.
drift_params = {'relative_drift': 0.20}
rram_sim.apply_relative_drift(**drift_params)

# 5. Get the final noisy model and run inference
# This is the only point where conductances are converted back to weights.
final_noisy_model = rram_sim.get_noisy_model_for_inference()
accuracy = validate(final_noisy_model, dataloader, criterion, device)

print(f"Final accuracy after all non-idealities: {accuracy:.2f}%")
```

## Model Accuracy and Based On
The non-ideality models implemented in this toolkit are phenomenological, designed to capture the behavior described in academic literature for efficient simulation.

* Conductance Drift: The primary long-term drift model is based on [1]. It models drift as additive Gaussian noise where the standard deviation is directly proportional to the target conductance (Relative Drift = σ / Gt).

* Relaxation vs. Drift: The concept of a distinct, rapid initial relaxation followed by a slower, long-term drift is based on characterization studies of RRAM devices.

* Thermal Noise: The general approach of modeling device noise and variations as Gaussian perturbations is consistent with methods used to simulate thermal noise in PIM devices.
