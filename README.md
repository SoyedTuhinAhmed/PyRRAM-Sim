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

# Initialize the simulator with your model and hardware specs
rram_sim = FastPyTorchRRAM(
    pretrained_model=my_model,
    G_min=1e-6,         # Minimum conductance (Siemens)
    G_max=100e-6,       # Maximum conductance (Siemens)
    differential=True
)

# Apply a one-time programming error to the ideal conductances
# This simulates the initial write operation to the chip.
prog_error_params = {'alpha_ind': 0.03, 'fault_model': 'state_independent'}
rram_sim.apply_programming_error(**prog_error_params)

# Apply short-term conductance relaxation
# This simulates the rapid drift that occurs shortly after programming.
relax_params = {'alpha_ind': 0.07, 'fault_model': 'state_independent'}
rram_sim.apply_conductance_relaxation(**relax_params)

# Apply long-term relative drift
# This uses the hardware-accurate model from the DoRA paper.
# The paper suggests drift is typically < 20% of the target conductance.
drift_params = {'relative_drift': 0.20}
rram_sim.apply_relative_drift(**drift_params)

# Get the final noisy model and run inference
# This is the only point where conductances are converted back to weights.
final_noisy_model = rram_sim.get_noisy_model_for_inference()
accuracy = validate(final_noisy_model, dataloader, criterion, device)

print(f"Final accuracy after all non-idealities: {accuracy:.2f}%")
```

---

## Supported Non-idealities Models

The non-ideality models implemented in this toolkit are phenomenological, designed to capture the behavior described in academic literature for efficient simulation.

### Conductance Mapping

* **Explanation**: This is the process of linearly scaling a floating-point weight from a pre-trained neural network to a physical conductance value within the RRAM's operational range `[G_min, G_max]`.
* **Model**: The mapping from the weight domain `[w_min, w_max]` to the conductance domain is performed using the following equation:
    ```
    G_target = G_min + ((weight - w_min) / (w_max - w_min)) * (G_max - G_min)
    ```
* **Basis**: This approach is based on standard hardware mapping procedures where whe weights are linearly scaled to align with the full conductance range Gmax of the hardware [1].

### Differential Conductance Mapping

* **Explanation**: To represent both positive and negative weights using RRAM cells that only have positive conductance, a single weight `W` is mapped to the difference between two cells, `G_p` and `G_n`. This is a common and robust technique in CIM hardware.
* **Model**: A weight `W` is represented by the differential conductance `G_p - G_n`. To map a given weight, `G_p` and `G_n` are calculated assuming a constant sum, which keeps both values within the device's valid range:
    ```
    G_diff = (W / w_max) * G_max
    G_p = (G_min + G_max + G_diff) / 2
    G_n = (G_min + G_max - G_diff) / 2
    ```
* **Basis**: This method is based on hardware designs where a signed kernel weight is mapped to the differential conductance of a pair of memristors and programmed as the differential conductance (G_r+ - G_r-) between two RRAM devices [1, 2].

### Quantization

* **Explanation**: Real RRAM devices often have a finite number of stable conductance levels rather than being perfectly continuous. This simulation models quantization by mapping a target conductance value to the nearest available discrete level, which is determined by the step size `delta_G`.
* **Model**:
    ```
    Levels = round((G_target - G_min) / delta_G)
    G_quantized = G_min + Levels * delta_G
    ```
* **Basis**: This is based on hardware implementations where weights are quantized from 32-bit floating type to, e.g., 15-level fixed-point type as shown in [2].

### Programming Error

* **Explanation**: This initial error occurs when a target conductance value is written to an RRAM cell. Due to the inherent stochasticity of the filament formation process, the achieved conductance will deviate from the target. This is a one-time error that happens during device programming. We model two components of this error.
* **State-Independent Model**: This models a baseline random noise that is independent of the target conductance state.
    * **Equation**: `Noise ~ N(0, σ²)` where `σ = alpha_ind * G_max`
* **State-Proportional Model**: This models the observation that programming variation can be greater for higher conductance states.
    * **Equation**: `Noise ~ N(0, σ²)` where `σ = alpha_prop * G_target`
* **Basis**: The existence of these errors is due to nonideal characteristics and reliability issues [4] such as device variations that carry errors into the weight stored in the RRAM [4].

### Conductance Relaxation

* **Explanation**: Immediately after programming, the conductive filament in an RRAM cell is in a quasi-stable state and will rapidly "relax" towards a more stable, but slightly different, conductance value. This toolkit models relaxation as a distinct, one-time noise event that occurs after the initial programming error.
* **Model**: `G_relaxed = G_programmed + Noise_relaxation`. This is typically simulated using a state-independent noise model with a larger standard deviation than the initial programming error.
* **Basis**: This effect is well-documented, where "the conductance undergoes a relaxation process, drifting over time" [1], with the largest drift "appearing immediately at the end of program and verify algorithms" [3] before it "stabilizes over time" [1].

### Relative Drift

* **Explanation**: This models the long-term, continuous change in conductance that occurs after the initial relaxation period has stabilized. The model is based on the findings that this long-term drift is random and its magnitude is proportional to the cell's current conductance.
* **Model**: Drift is modeled as additive Gaussian noise where the standard deviation (`σ`) is directly proportional to the current conductance (`G_current`).
    ```
    G_drift ~ N(0, σ²)
    σ = RelativeDrift * G_current
    G_final = G_current + G_drift
    ```
* **Basis**: This model is based on [1], which states, "To model this drift, we assume that the deviation in conductance, G\_drift, follows a Gaussian distribution". The same paper explicitly defines the term as `Relative Drift = σ / Gt` [1].

---


## Citing This Work
If you use PyRRAM-Sim in your research, we kindly ask that you cite the following paper:

```bash
@article{YourLastName2025PyRRAMSim,
  title   = {Accurate Mapping and Inferencing of Large Vision and Language Models Based on RRAM-based Compute-in-Memory Architecture using Robust Multi-domain Low-Rank Adaptation},
  author  = {Your Name and Co-authors},
  journal = {Journal/Conference Name},
  year    = {2025},
  volume  = {XX},
  number  = {YY},
  pages   = {ZZZ--ZZZ}
}
```


## Citations

[1] W. Dong, K. Zhou, Z. Kong, Q. Cheng, J. Huang, Z. Yang, M. Hashimoto, and L. Lin, "Efficient Calibration for RRAM-based In-Memory Computing using DoRA," *arXiv preprint arXiv:2504.03763*, 2025.

[2] P. Yao, H. Wu, B. Gao, J. Tang, Q. Zhang, W. Zhang, J. J. Yang, and H. Qian, "Fully hardware-implemented memristor convolutional neural network," *Nature*, vol. 577, pp. 641–646, 2020.

[3] A. Baroni, A. Glukhov, E. Pérez, C. Wenger, D. Ielmini, P. Olivo, and C. Zambelli, "Low Conductance State Drift Characterization and Mitigation in Resistive Switching Memories (RRAM) for Artificial Neural Networks," *IEEE Transactions on Device and Materials Reliability*, vol. 22, no. 3, pp. 340-348, 2022.

[4] Y. Liu, B. Gao, J. Tang, H. Wu, and H. Qian, "Architecture-circuit-technology co-optimization for resistive random access memory-based computation-in-memory chips," *Science China Information Sciences*, vol. 66, no. 10, 2023.
