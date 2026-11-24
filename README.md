# PyRRAM-Sim: A Toolkit for RRAM Non-Ideality Simulation

PyRRAM-Sim is a lightweight, PyTorch-based simulation tool designed for researchers and engineers working on Deep Neural Networks (DNNs) with RRAM-based Compute-in-Memory (CIM) architectures. It provides an efficient, modular framework to simulate key hardware non-idealities, allowing for the evaluation of model robustness before hardware deployment.

This toolkit simulates the sequential, cumulative effects of permanent stuck-at faults, programming errors, short-term conductance relaxation, and long-term, state-proportional conductance drift. The models are based on phenomena described in leading academic research to ensure hardware-accurate simulation.

---

## What's New

* **Stuck-at fault injection** (`apply_stuck_at_faults`): cells stuck at `G_on`, `G_off` or an intermediate state, for binary and multi-level cells. Each cell of a differential pair fails independently, the healthy partner of a stuck cell can optionally be reprogrammed to compensate, and `fault_seed` reproduces the same faulty chip.
* **Multi-level and bit-sliced cells**: `n_levels` conductance levels per cell and `cells_per_weight` cells per weight, e.g. 8-bit weights on one 256-level cell or on eight binary cells.
* **One-sided differential mapping** (`diff_style="one_sided"`) next to the constant-sum two-sided mapping.
* **Per-cell conductance state**: the simulator now stores the conductance of every cell between steps, so faults persist and noise accumulates on the conductances instead of being re-derived from the weights after each step. As a consequence, programming error, relaxation and drift give slightly different numbers than earlier versions for the same seed.
* **Pipeline helpers**: `simulate()` runs the full chain on a fresh chip, and `MC_simulate("stuck_at", ...)` runs Monte-Carlo simulations over fault maps.
* **Command-line evaluation**: `main.py` evaluates a CIFAR-10-trained ResNet18 with every non-ideality configurable from flags (see [Validation Results](#validation-results)).

---

## Key Features

* **Efficient State Management**: Simulates the physical state of RRAM by storing the conductance of every cell and applying all non-idealities directly to it.
* **Sequential Non-Idealities**: Accurately models the workflow of stuck-at faults from fabrication, followed by one-time programming errors, short-term relaxation and long-term drift.
* **Stuck-at Fault Injection**: Cells stuck at `G_on`, `G_off` or an intermediate state, for binary and multi-level cells, with optional fault-aware programming of differential pairs.
* **Hardware-Accurate Drift Model**: Implements a state-proportional drift model based on findings from recent literature.
* **Differential Pair Support**: Natively supports the simulation of differential RRAM cells (`W = Gp - Gn`) for signed weight representation, in two-sided and one-sided styles.
* **Multi-Level and Bit-Sliced Cells**: Any number of levels per cell, and weights bit-sliced across several cells.
* **PyTorch Integration**: Seamlessly integrates with existing PyTorch models and validation pipelines.

---

## Installation

The toolkit is contained within the `PyRRAMSimulator` class in `src/`. Simply add the `src` folder to your project. It requires Python 3.9+ and PyTorch; `torchvision` is only needed for the command-line script `main.py`:

```bash
pip install torch torchvision
```

| File | Contents |
|---|---|
| `src/pytorchrram.py` | `PyRRAMSimulator`: weight-to-conductance mapping, simulation pipeline, Monte-Carlo runs |
| `src/rram_noise_model.py` | Conductance noise and stuck-at fault models |
| `src/helper.py` | `validate` (loss, accuracy) and `set_seed` |
| `main.py` | Command-line evaluation of a CIFAR-10-trained ResNet18 |

---

## How to Run

### Command Line

`main.py` evaluates ResNet18 [13] trained on CIFAR-10 [14] and reports the digital baseline and the accuracy after the selected non-idealities. On the first run it downloads the CIFAR-10 test set into `--dataset-path` and a pretrained checkpoint [15] into the PyTorch hub cache (integrity-checked by its SHA-256 hash).

```bash
# Ideal 8-bit chip: 256 levels per cell, differential pairs, no non-idealities
python main.py --dataset-path ./data --n-levels 256

# Stuck-at faults: 9.04% of cells stuck at G_max, 1.75% at G_min, fixed fault map
python main.py --dataset-path ./data --n-levels 256 \
    --apply-stuck-at --p-on 0.0904 --p-off 0.0175 --fault-seed 1

# Same faults, with fault-aware compensation of the differential pairs
python main.py --dataset-path ./data --n-levels 256 \
    --apply-stuck-at --p-on 0.0904 --p-off 0.0175 --fault-seed 1 --compensate

# Full pipeline: stuck-at faults, programming error, relaxation and drift
python main.py --dataset-path ./data --n-levels 256 \
    --apply-stuck-at --p-on 0.0904 --p-off 0.0175 --fault-seed 1 --compensate \
    --apply-prog-error --apply-relaxation --apply-drift

# Your own CIFAR-10 ResNet18, 8-bit weights bit-sliced over eight binary cells
python main.py --dataset-path ./data --checkpoint my_resnet18.pth \
    --n-levels 2 --cells-per-weight 8 --apply-stuck-at --p-on 0.01 --p-off 0.002
```

| Flag | Default | Description |
|---|---|---|
| `--dataset-path` | required | CIFAR-10 location (downloaded if missing) |
| `--checkpoint` | pretrained [15] | State dict of your own CIFAR-10 ResNet18 (3x3 stem, no max-pool, same normalization) |
| `--batch-size` | 64 | Evaluation batch size |
| `--g-min`, `--g-max` | 1e-6, 100e-6 | Device conductance range (Siemens) |
| `--n-levels` | 991 (`delta_G` = 0.1 µS) | Conductance levels per cell |
| `--cells-per-weight` | 1 | Cells per weight and polarity (bit-slicing) |
| `--no-differential` | off | Single-ended cells instead of differential pairs |
| `--diff-style` | `two_sided` | Differential mapping: `two_sided` or `one_sided` |
| `--apply-stuck-at` | off | Inject stuck-at faults before programming |
| `--p-on`, `--p-off`, `--p-int` | 0, 0, 0 | Probability of a cell stuck at `G_max`, `G_min`, an intermediate state |
| `--compensate` | off | Reprogram the healthy partner of a stuck differential cell |
| `--fault-seed` | none | Dedicated seed for the fault map |
| `--apply-prog-error` | off | Programming error, with `--prog-alpha-ind` (0.03) and `--prog-alpha-prop` (0.0) |
| `--apply-relaxation` | off | Conductance relaxation, with `--relax-alpha-ind` (0.07) |
| `--apply-drift` | off | Relative drift, with `--relative-drift` (0.2) |
| `--seed` | 42 | Global random seed |

### Python API

The following example demonstrates the standard workflow for simulating all non-idealities on a pretrained PyTorch model.

```python
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn

from pytorchrram import PyRRAMSimulator
from helper import validate

# Assume 'my_model', 'dataloader', 'criterion' and 'device' are already defined

# Initialize the simulator with your model and hardware specs
rram_sim = PyRRAMSimulator(
    pretrained_model=my_model,
    G_min=1e-6,         # Minimum conductance (Siemens)
    G_max=100e-6,       # Maximum conductance (Siemens)
    n_levels=256,       # 8-bit weights on 256-level cells
    differential=True
)

# Inject permanent stuck-at faults into the fresh chip (fabrication defects)
# This must come before programming: stuck cells ignore all later noise.
saf_params = {'p_on': 0.0904, 'p_off': 0.0175}
fault_summary = rram_sim.apply_stuck_at_faults(**saf_params)

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

# Run inference on the final noisy model
# Its weights are read from the cell conductances after every step.
_, accuracy = validate(rram_sim.analog_model, dataloader, criterion, device)

print(f"Final accuracy after all non-idealities: {accuracy:.2f}%")
```

`apply_stuck_at_faults` takes the following parameters and returns the realized fault counts and rates:

| Parameter | Default | Description |
|---|---|---|
| `p_on`, `p_off`, `p_int` | 0.0 | Probability of a cell stuck at `G_max`, `G_min`, an intermediate state (mutually exclusive, sum <= 1) |
| `int_dist` | `"auto"` | Intermediate value: `"levels"` (random interior level), `"continuous"` (uniform in the band); `"auto"` uses `"continuous"` for binary cells and `"levels"` otherwise |
| `int_range` | `(0.0, 1.0)` | Band for intermediate values, as fractions of `[G_min, G_max]` |
| `compensate` | `False` | Reprogram the healthy partner of a single stuck cell in a differential pair |
| `fault_seed` | `None` | Seed of a dedicated generator for the fault map |

Monte-Carlo runs and the full pipeline on a fresh chip:

```python
# Monte-Carlo over 20 faulty chips (a new fault map per run)
mean_acc, std_acc = rram_sim.MC_simulate("stuck_at", dataloader, criterion, number_runs=20,
                                         p_on=0.0904, p_off=0.0175, compensate=True)

# Full pipeline: stuck-at faults -> programming error -> relaxation -> drift
accuracy = rram_sim.simulate(dataloader, criterion,
                             stuck_at_params={'p_on': 0.0904, 'p_off': 0.0175},
                             programming_error_params={'alpha_ind': 0.03},
                             relaxation_params={'alpha_ind': 0.07},
                             drift_params={'relative_drift': 0.2})
```

---

## Validation Results

The simulator was validated end-to-end with `main.py` on ResNet18 [13] trained on CIFAR-10 [14, 15], evaluated on the full CIFAR-10 test set (10,000 images).

**Setup**: `G_min` = 1 µS, `G_max` = 100 µS, 8-bit weights, two-sided differential pairs, all `Conv2d` and `Linear` layers mapped to RRAM, global seed 42, batch size 250, PyTorch 2.8.0 and torchvision 0.23.0 on CPU. Rows A-D store each weight on one 256-level cell per polarity (fault seed 1); rows E-F bit-slice it over eight binary cells per polarity.

| Configuration | Top-1 accuracy |
|---|---|
| Digital (floating-point) baseline | 94.98% |
| A. RRAM, 256-level cells, no non-idealities | 94.99% |
| B. A + stuck-at faults (`p_on` = 9.04%, `p_off` = 1.75%) | 10.00% |
| C. B with fault-aware compensation | 10.00% |
| D. C + programming error (`alpha_ind` = 0.03), relaxation (`alpha_ind` = 0.07) and relative drift (0.2) | 10.00% |
| E. RRAM, 8 binary cells per weight, no non-idealities | 94.99% |
| F. E + stuck-at faults (`p_on` = 1%, `p_off` = 0.2%) | 91.59% |

* **A** and **E** match the digital baseline, which validates the conductance mapping, 8-bit quantization, bit-slicing and read-back.
* **F** shows a gradual degradation at a lower fault rate: with 1.2% stuck cells, accuracy drops by 3.4 points.
* **B-D** use the stuck-at fault rates measured on a 4 Mb HfO2 test chip [5], about 10.8% of all cells. Without fault tolerance, the network drops to chance level (10% for ten classes). With full-range mapping, a cell stuck at `G_max` or `G_min` shifts its weight by a sizeable fraction of the layer's weight range, and with two cells per weight about 20% of the weights are affected. Compensation corrects roughly half of the weights with a single stuck cell, but not the remaining ones nor pairs with both cells stuck. Studies of stuck-at fault tolerance therefore evaluate unmitigated networks at fault rates of 0.1-1% [9], and rely on fault-tolerant training, remapping or redundancy at chip-level fault rates [6, 10, 11].

Rows A-D are reproduced by the corresponding commands in [Command Line](#command-line) with `--batch-size 250`. Rows E-F use `--n-levels 2 --cells-per-weight 8`, without and with `--apply-stuck-at --p-on 0.01 --p-off 0.002`.

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
* **Model**: A weight `W` is represented by the differential conductance `G_p - G_n`. Two styles are supported (`diff_style`); the names follow CrossSim's balanced core:
    * `two_sided` (default): `G_p` and `G_n` keep a constant sum, which keeps both values within the device's valid range. With `W_norm = 2 (W - w_min) / (w_max - w_min) - 1`:
        ```
        G_diff = W_norm * (G_max - G_min)
        G_p = (G_min + G_max + G_diff) / 2
        G_n = (G_min + G_max - G_diff) / 2
        ```
    * `one_sided`: the active cell carries `|W|` and the idle cell stays at `G_min`:
        ```
        G_p = G_min + (max(W, 0) / max|W|) * (G_max - G_min)
        G_n = G_min + (max(-W, 0) / max|W|) * (G_max - G_min)
        ```
* **Basis**: This method is based on hardware designs where a signed kernel weight is mapped to the differential conductance of a pair of memristors and programmed as the differential conductance (G_r+ - G_r-) between two RRAM devices [1, 2]. The one-sided style, with the idle cell at the minimum conductance, is used for example in [10, 11].

### Quantization

* **Explanation**: Real RRAM devices often have a finite number of stable conductance levels rather than being perfectly continuous. Each cell has `n_levels` equally spaced levels (given directly, or derived from the step size `delta_G`). A weight can also be bit-sliced across `cells_per_weight` cells per polarity, e.g. 8-bit weights on 256-level cells (`n_levels=256`), on four 2-bit cells (`n_levels=4, cells_per_weight=4`), or on eight binary cells (`n_levels=2, cells_per_weight=8`).
* **Model**: The weight is quantized to an integer code `q` in `[0, Q]` with `Q = n_levels^cells_per_weight - 1`, split into base-`n_levels` digits `d_k` (LSB cell first), and each cell is programmed to its level:
    ```
    delta_G = (G_max - G_min) / (n_levels - 1)
    G_k = G_min + d_k * delta_G
    q = sum_k n_levels^k * d_k
    ```
    For `two_sided` pairs, `G_n` stores the complement code `Q - q`; for `one_sided` pairs, the code of `|W|` is stored in the cell of the weight's sign.
* **Basis**: This is based on hardware implementations where weights are quantized from 32-bit floating type to, e.g., 15-level fixed-point type as shown in [2]. Bit-slicing with digit significance `n_levels^k` follows [10].

### Stuck-at Faults

* **Explanation**: Fabrication defects and limited endurance leave some cells permanently stuck: their conductance no longer responds to programming. Over-forming and short defects pin a cell at the low-resistance state (stuck at `G_on = G_max`), while broken word lines and open switches pin it at the high-resistance state (stuck at `G_off = G_min`) [5, 6]. Defect-level analysis also shows cells stuck in an undefined state between the two resistance states [7], and in multi-level cells a stuck cell can sit at any level [8]. Stuck-at faults are permanent, so they are injected on the fresh chip, before programming, and stuck cells are unaffected by programming error, relaxation and drift.
* **Model**: Each physical cell (each cell of a differential pair and each bit-slice) independently draws one fault state:
    ```
    P(stuck at G_max) = p_on
    P(stuck at G_min) = p_off
    P(stuck at G_int) = p_int
    G_int ~ uniform over interior levels  {G_min + k * delta_G, 1 <= k <= n_levels - 2}   (int_dist="levels", multi-level cells)
    G_int ~ uniform in (G_min, G_max)                                                     (int_dist="continuous", binary cells)
    ```
    `int_range=(lo, hi)` restricts `G_int` to a band of the range. A stuck cell in bit-slice `k` corrupts the weight in proportion to its significance `n_levels^k` [10].
* **Differential Pairs**: `G_p` and `G_n` fail independently, so the impact of a fault depends on the mapping style. In the one-sided style, a stuck-off active cell acts like pruning, while a stuck-on idle cell causes the largest error [9, 11]. With `compensate=True`, the healthy partner of a single stuck cell is reprogrammed (fault-aware mapping) so that `G_p - G_n` matches its target as closely as the range and levels allow [11]:
    ```
    G_p stuck:  G_n = clip(G_p_stuck - (G_p_target - G_n_target), G_min, G_max)
    G_n stuck:  G_p = clip(G_n_stuck + (G_p_target - G_n_target), G_min, G_max)
    ```
    Pairs with both cells stuck cannot be compensated.
* **Basis**: Measurements on a 4 Mb HfO2 RRAM test chip found about 10% stuck cells [5, 6], commonly cited as 9.04% stuck at LRS and 1.75% stuck at HRS, a ratio of about 5:1 [9]. Stuck cells in analog/multi-level arrays usually sit at the highest or lowest value [6], and faults are modeled as uniformly distributed over the array [6]. Measured ReRAM arrays have shown 9.8-13.5% corrupt devices, which IBM AIHWKit models as stuck at a random value within a band of the range [12]. The sampling is i.i.d. per cell, so the realized fault counts follow a binomial distribution.

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
@article{Ahmed2025PyRRAMSim,
  title   = {PyRRAM-Sim: A Toolkit for RRAM Non-Ideality Simulation},
  author  = {Ahmed, Soyed Tuhin},
  year    = {2025},
}
```


## Citations

[1] W. Dong, K. Zhou, Z. Kong, Q. Cheng, J. Huang, Z. Yang, M. Hashimoto, and L. Lin, "Efficient Calibration for RRAM-based In-Memory Computing using DoRA," *arXiv preprint arXiv:2504.03763*, 2025.

[2] P. Yao, H. Wu, B. Gao, J. Tang, Q. Zhang, W. Zhang, J. J. Yang, and H. Qian, "Fully hardware-implemented memristor convolutional neural network," *Nature*, vol. 577, pp. 641–646, 2020.

[3] A. Baroni, A. Glukhov, E. Pérez, C. Wenger, D. Ielmini, P. Olivo, and C. Zambelli, "Low Conductance State Drift Characterization and Mitigation in Resistive Switching Memories (RRAM) for Artificial Neural Networks," *IEEE Transactions on Device and Materials Reliability*, vol. 22, no. 3, pp. 340-348, 2022.

[4] Y. Liu, B. Gao, J. Tang, H. Wu, and H. Qian, "Architecture-circuit-technology co-optimization for resistive random access memory-based computation-in-memory chips," *Science China Information Sciences*, vol. 66, no. 10, 2023.

[5] C.-Y. Chen et al., "RRAM Defect Modeling and Failure Analysis Based on March Test and a Novel Squeeze-Search Scheme," *IEEE Transactions on Computers*, vol. 64, no. 1, 2015, doi: 10.1109/TC.2014.12.

[6] L. Xia, M. Liu, X. Ning, K. Chakrabarty, and Y. Wang, "Fault-Tolerant Training Enabled by On-Line Fault Detection for RRAM-Based Neural Computing Systems," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2018, doi: 10.1109/TCAD.2018.2855145.

[7] M. Fieback, G. Cardoso Medeiros, L. Wu, H. Aziza, R. Bishnoi, M. Taouil, and S. Hamdioui, "Defects, Fault Modeling, and Test Development Framework for RRAMs," *ACM Journal on Emerging Technologies in Computing Systems*, vol. 18, no. 3, Art. 52, 2022.

[8] S. Kannan, N. Karimi, R. Karri, and O. Sinanoglu, "Modeling, Detection, and Diagnosis of Faults in Multilevel Memristor Memories," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2015.

[9] G. Yuan et al., "Improving DNN Fault Tolerance using Weight Pruning and Differential Crossbar Mapping for ReRAM-based Edge AI," *ISQED*, 2021, arXiv:2106.09166.

[10] K. E. Jeon, S. Yeon, J. Kim, H. Bang, J. Rhe, and J. H. Ko, "Row-Column Hybrid Grouping for Fault-Resilient Multi-Bit Weight Representation on IMC Arrays," *ICCAD*, 2025, arXiv:2508.15685.

[11] A. Dorostkar, H. Farbeh, and H. R. Zarandi, "ReSAFT: An Efficient Stuck-at Fault-Tolerant Scheme for ReRAM-based Process-in-Memory Accelerators," *arXiv preprint arXiv:2610.09999*, 2026.

[12] N. Gong, M. J. Rasch, et al., *IEEE International Electron Devices Meeting (IEDM)*, 2022 (IEEE Xplore 10019569); measured corrupt-device rates as used in the IBM AIHWKit ReRAM array presets.

[13] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.

[14] A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Technical Report, University of Toronto, 2009.

[15] edadaltocg, "resnet18_cifar10," Hugging Face model repository, MIT License, https://huggingface.co/edadaltocg/resnet18_cifar10 (reported test accuracy 94.98%).
