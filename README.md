# PyRRAM-Sim: A Toolkit for RRAM Non-Ideality Simulation

PyRRAM-Sim is a lightweight, PyTorch-based simulation tool designed for researchers and engineers working on Deep Neural Networks (DNNs) with RRAM-based Compute-in-Memory (CIM) architectures. It provides an efficient, modular framework to simulate key hardware non-idealities, allowing for the evaluation of model robustness before hardware deployment.

This toolkit simulates the sequential, cumulative effects of permanent stuck-at faults, programming errors, short-term conductance relaxation, long-term, state-proportional conductance drift, per-inference read noise and temperature. The models are based on phenomena described in leading academic research to ensure hardware-accurate simulation.

It also evaluates and hardens instruction-tuned LLMs: the preference-inversion (flip) rate of aligned models under RRAM non-idealities and hardware-aware DPO (HARD-DPO) [16], and mixtures of fault-tolerant LoRA experts, merged at design time (MoF-LoRA) [17] or kept as separate adapters with telemetry-guided selection (Sparse-MoF) [18].

---

## What's New

* **Alignment robustness of LLMs** (`src/alignment.py`, `llm_main.py eval`): preference accuracy and preference flip rate of instruction-tuned LLMs on UltraFeedback [21, 22] under RRAM non-idealities, with Monte-Carlo chips and 95% confidence intervals [16]. Only the static projections (q/k/v/o, gate/up/down) are mapped to RRAM; tied embeddings and `lm_head` stay digital.
* **HARD-DPO** (`llm_main.py train-hard-dpo`): hardware-aware DPO [19] with LoRA [20], log-normal programming-noise injection, clean-noisy KL consistency and 3-sigma clipping [16]. Existing PEFT checkpoints of HARD-DPO load directly.
* **Fault-tolerant LoRA experts** (`src/lora.py`, `src/mof.py`): programming-error, drift and stuck-at experts [17], plus read-noise and temperature experts [18], for `Linear` and `Conv2d` layers. Experts stay separate: any subset can run digitally next to the crossbar (hybrid), be merged into the programmed weights, or be programmed into their own cells.
* **Sparse-MoF expert selection** (`mof.SparseMoF`): an online test of the simulated chip (stuck-cell rate, drift, read noise, temperature sensor) selects the experts; when the set changes, the experts are re-merged and the chip is reprogrammed with its stuck-at map intact [18].
* **Read noise and temperature**: per-inference, state-proportional read noise, and Arrhenius scaling of programming error and read noise with a user-supplied activation energy.
* **Simulator additions**: `target_modules` / `exclude_modules`, deployment clipping (`clip_sigma`), `reprogram()` on the same faulty chip, `fault_map()` (March-test equivalent), `telemetry()`, and `low_memory=True`, which replays every operation layer by layer instead of storing cell states, for multi-billion-parameter models.

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
* **LLM Alignment Evaluation**: Preference flip rate of instruction-tuned LLMs on simulated RRAM chips, and HARD-DPO training.
* **Fault-Tolerant Experts**: MoF-LoRA and Sparse-MoF experts for programming error, drift, stuck-at faults, read noise and temperature, with telemetry-guided selection.
* **PyTorch Integration**: Seamlessly integrates with existing PyTorch models (including Hugging Face Transformers) and validation pipelines.

---

## Installation

The toolkit is contained within the `PyRRAMSimulator` class in `src/`. Simply add the `src` folder to your project. It requires Python 3.9+ and PyTorch; `torchvision` is only needed for the command-line script `main.py`:

```bash
pip install torch torchvision
pip install transformers datasets   # only for the LLM features (alignment.py, llm_main.py)
```

| File | Contents |
|---|---|
| `src/pytorchrram.py` | `PyRRAMSimulator`: weight-to-conductance mapping, simulation pipeline, Monte-Carlo runs, read noise, temperature, telemetry |
| `src/rram_noise_model.py` | Conductance noise, stuck-at fault and Arrhenius models |
| `src/lora.py` | LoRA expert banks for `Linear` and `Conv2d` layers: activation, merging, checkpoints, PEFT import |
| `src/train_noise.py` | Training-time noise presets of HARD-DPO, MoF-LoRA and HAT-FI, and the simulator-exact preset |
| `src/mof.py` | Expert training stages (prog, drift, saf, read, temp) and the Sparse-MoF controller |
| `src/alignment.py` | Preference scoring, flip-rate evaluation and HARD-DPO training for LLMs |
| `src/helper.py` | `validate` (loss, accuracy) and `set_seed` |
| `main.py` | Command-line evaluation of a CIFAR-10-trained ResNet18 |
| `llm_main.py` | Command line for LLMs: `eval`, `train-hard-dpo`, `train-experts` |
| `mof_main.py` | Fault-tolerant experts on CIFAR-10 ResNet18: unprotected vs. MoF-LoRA vs. Sparse-MoF |

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

### LLM Alignment Robustness and HARD-DPO

`llm_main.py` works with any Hugging Face causal LM that has a chat template (Qwen2.5, Llama-3.1, Gemma-2, ...). The model and UltraFeedback (`HuggingFaceH4/ultrafeedback_binarized`) [21, 22] are downloaded on first use. By default only the projections `q/k/v/o_proj` and `gate/up/down_proj` are mapped to RRAM, on 8-bit (256-level) two-sided differential cells with `G` = 1-100 µS; programming error is state-independent, `sigma_G = alpha * G_max` [16].

```bash
# Flip rate of the instruction-tuned model: 500 held-out pairs, 10 chips per alpha
python llm_main.py eval --model Qwen/Qwen2.5-0.5B-Instruct --pairs 500 \
    --alphas 0 0.002 0.004 0.006 0.008 --chips 10 --out base.json

# HARD-DPO: 1000 steps on train_prefs[:10000] (paper defaults), saves the LoRA expert
python llm_main.py train-hard-dpo --model Qwen/Qwen2.5-0.5B-Instruct --steps 1000 --out hard_dpo.pt

# Deploy it merged into the crossbar and clipped at 3 sigma, then evaluate
python llm_main.py eval --model Qwen/Qwen2.5-0.5B-Instruct --experts hard_dpo.pt \
    --deploy merged --clip-sigma 3 --out hard_dpo.json

# A HARD-DPO checkpoint from the original PEFT scripts (lora_alpha 16), adapters kept digital
python llm_main.py eval --peft-adapter noise_robust_dpo_qwen2.5_0.5b.pt --lora-alpha 16 --deploy hybrid

# Released-script protocol (train_prefs[:500], 50 pairs per chip, G = 50-350 uS unquantized,
# every Linear including lm_head) to reproduce previously published numbers
python llm_main.py eval --legacy-eval --peft-adapter noise_robust_dpo_qwen2.5_0.5b.pt

# Fault-tolerant experts trained with the DPO objective; the SAF expert targets chip --fault-seed 7
python llm_main.py train-experts --train prog drift saf read --recipe hat-fi-llm --objective dpo \
    --steps 500 --p-on 0.01 --p-off 0.01 --fault-seed 7 --out experts.pt

# Sparse-MoF on that chip: telemetry selects the experts per chip
python llm_main.py eval --experts experts.pt --select-experts --p-on 0.01 --p-off 0.01 --fault-seed 7 \
    --drift 0.05 --read-alpha 0.005 --alphas 0.004
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model id or local path |
| `--split`, `--pairs` | `test_prefs`, 500 | Evaluation pairs (`train-*`: `train_prefs`, `--max-prefs` 10000) |
| `--alphas`, `--chips` | 0 ... 0.008, 10 | Programming-error sweep and Monte-Carlo chips per point |
| `--fault-model` | `state_independent` | Or `state_proportional`, `both` |
| `--n-levels`, `--cells-per-weight`, `--diff-style` | 256, 1, `two_sided` | Cells, as in `main.py` |
| `--target-modules` | the 7 projections | Layers mapped to RRAM and adapted by LoRA |
| `--clip-sigma` | none | Clip each weight tensor to `+-k std` before mapping |
| `--experts`, `--peft-adapter` | none | Expert checkpoint of this tool, or a PEFT LoRA state dict (`--lora-alpha`, `--peft-name`) |
| `--active` | all loaded | Experts to switch on |
| `--deploy` | `merged` | `merged` (experts in the crossbar), `hybrid` (experts digital), `analog-lora` (factors on own cells) |
| `--p-on`, `--p-off`, `--fault-seed`, `--compensate` | 0, 0, none, off | Stuck-at faults of the chip |
| `--relax-alpha`, `--drift`, `--read-alpha` | 0 | Relaxation, relative drift, per-inference read noise |
| `--temperature`, `--ea` | 25, none | Operating temperature (C) and activation energy (eV, required above 25 C) |
| `--select-experts` | off | Sparse-MoF selection from telemetry on every chip |
| `--low-memory` | off | Replay operations layer by layer (no per-cell state; for large models) |
| `--legacy-eval` | off | Protocol of the released HARD-DPO scripts |
| `--sigma`, `--noise`, `--noise-target` | 0.008, `lognormal`, `merged` | HARD-DPO training noise; `--noise simulator` uses the simulator-exact preset, `--noise-target base` perturbs only `W0` like the released script |
| `--beta`, `--lambda-kd`, `--kd-tau`, `--rank`, `--lora-alpha`, `--lr`, `--train-batch` | 0.1, 0.5, 1, 8, 16, 5e-5, 4 | HARD-DPO hyperparameters |
| `--train`, `--recipe`, `--objective` | `prog drift saf`, `hat-fi-llm`, `dpo` | Experts to train, coefficient recipe, task loss (`dpo` or `sft`) |

### Fault-Tolerant Experts on CIFAR-10

`mof_main.py` trains the five experts on CIFAR-10 ResNet18 (weight-space LoRA on all `Conv2d` layers [18]) for a target chip and compares the three deployments under cumulative non-idealities:

```bash
python mof_main.py --dataset-path ./data --steps 300 --save-experts resnet18_experts.pt \
    --temperature 85 --ea 0.1 --train-temperature 85
# re-evaluate saved experts, e.g. on a different fault rate
python mof_main.py --dataset-path ./data --load-experts resnet18_experts.pt --p-saf 0.02
```

`--temperature`, `--ea` and `--train-temperature` are needed only for the temperature expert and condition; without them those are skipped. The chip defaults are `--prog-alpha 0.03`, `--drift 0.05`, `--p-saf 0.04` (half stuck-on, half stuck-off, `--fault-seed 1`) and `--read-alpha 0.02`.

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

Preference flip rate of an instruction-tuned LLM, and HARD-DPO:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from alignment import PreferenceScorer, load_ultrafeedback, evaluate_flip_rate, train_hard_dpo, HardDPOConfig
from lora import DEFAULT_TARGET_MODULES, set_merge

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
llm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
scorer = PreferenceScorer(tok, batch_size=8)
pairs = load_ultrafeedback("test_prefs", n=500)

# Only the projections go to RRAM; copy_model=False avoids a third copy of the LLM
sim = PyRRAMSimulator(llm, G_min=1e-6, G_max=100e-6, n_levels=256, device=device,
                      target_modules=DEFAULT_TARGET_MODULES, copy_model=False)
res = evaluate_flip_rate(sim, scorer, pairs, alphas=(0.0, 0.004, 0.008), n_chips=10)
print(res["per_alpha"]["0.008"]["flip_rate"])   # {'mean': ..., 'std': ..., 'ci95': ...}

# HARD-DPO trains a LoRA expert in place; deploy it merged into the crossbar, clipped at 3 sigma
train_hard_dpo(llm, scorer, load_ultrafeedback("train_prefs", n=10000), HardDPOConfig(steps=1000))
set_merge(llm, True)
sim = PyRRAMSimulator(llm, 1e-6, 100e-6, n_levels=256, device=device,
                      target_modules=DEFAULT_TARGET_MODULES, clip_sigma=3.0, copy_model=False)
res = evaluate_flip_rate(sim, scorer, pairs, alphas=(0.0, 0.004, 0.008), n_chips=10)
```

Fault-tolerant experts for a target chip, then MoF-LoRA (static merge) and Sparse-MoF (telemetry-guided selection):

```python
import lora, mof

lora.add_lora(my_model, target_modules=None)        # every Linear / Conv2d except lm_head
chip_cfg = dict(stuck_at_params={'p_on': 0.02, 'p_off': 0.02, 'fault_seed': 7},
                programming_error_params={'alpha_ind': 0.03},
                relaxation_params={'alpha_ind': 0.0},
                drift_params={'relative_drift': 0.05})
chip = PyRRAMSimulator(my_model, 1e-6, 100e-6, n_levels=256, device=device, copy_model=False)
chip.apply_stuck_at_faults(**chip_cfg['stuck_at_params'])   # the SAF expert learns this chip

loss_fn = lambda m, batch: criterion(m(batch[0].to(device)), batch[1].to(device))
mof.train_experts(my_model, loss_fn, train_loader, steps=300,
                  experts=("prog", "drift", "saf", "read"), recipe="hat-fi-cnn", sim=chip)

# MoF-LoRA: all chain experts merged into the crossbar at design time
lora.set_active(my_model, ["prog", "drift", "saf"])
lora.set_merge(my_model, True)
acc_mof = chip.simulate(dataloader, criterion, **chip_cfg)

# Sparse-MoF: start with the always-on expert, measure the chip, select, re-merge and reprogram
ctl = mof.SparseMoF(chip, reprogram_params={k: v for k, v in chip_cfg.items() if k != 'stuck_at_params'})
ctl.reset()
chip.program(**chip_cfg)
print(ctl.step())   # {'telemetry': {...}, 'active': ['prog', 'drift', 'saf'], 'changed': True}
_, acc_sparse = validate(chip.analog_model, dataloader, criterion, device)
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

### Preference Inversion on Qwen2.5-0.5B-Instruct

**Setup**: `llm_main.py eval` on Qwen2.5-0.5B-Instruct [23] with the first 100 pairs of UltraFeedback `test_prefs` [21, 22] (98 pairs after `--max-length 1024` drops two long prompts), 3 chips per `alpha`, the 168 projection matrices on 8-bit two-sided cells (`G` = 1-100 µS), `--low-memory`, float32 on an Apple M5 GPU (MPS), PyTorch 2.8.0, Transformers 4.57.6. Flip rates are means ± 95% confidence intervals over the chips; every row uses the same chips.

| Configuration | Clean preference accuracy | Flip rate, `alpha` = 0 | `alpha` = 0.004 | `alpha` = 0.008 | RRAM preference accuracy, `alpha` = 0.008 |
|---|---|---|---|---|---|
| Instruction-tuned | 54.1% | 1.9% | 6.3 ± 1.2% | 16.4 ± 6.5% | 50.3% |
| + 3-sigma clipping only | 40.8% | 0.0% | 5.8 ± 3.3% | 10.0 ± 2.8% | 43.2% |
| HARD-DPO checkpoint trained with the original scripts of [16] (rank 8, `lora_alpha` 128; not distributed here), merged + 3-sigma clipping | 55.1% | 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 55.1% |
| HARD-DPO, 100 steps on a laptop, merged + 3-sigma clipping | 54.1% | 0.0% | 0.0% | 1.3 ± 1.2% | 53.7% |

* Programming error inverts the preference ordering of the instruction-tuned model: the flip rate grows from 1.9% (8-bit quantization alone) to 16% at `alpha` = 0.8%, as predicted by Theorem 1 of [16].
* Clipping the weights to 3 sigma before mapping removes outliers that set the mapping scale, but without a trained adapter it costs 13 points of clean preference accuracy. The HARD-DPO update learns to compensate for the clipping [16]: the merged and clipped HARD-DPO checkpoint keeps (and slightly raises) the clean accuracy and shows no flip on any chip up to `alpha` = 0.8%, and its preference accuracy on the chip equals the digital one. The checkpoint is the PEFT LoRA state dict of the original HARD-DPO scripts, imported with `--peft-adapter --lora-alpha 128`.
* The laptop run (100 steps, batch 2, `--max-length 512`, `--gradient-checkpointing`, 13 minutes) is a functional check of the trainer, not a reproduction of the paper's 1000-step training; all rows use only 98 pairs and 3 chips.

**Released-script protocol** (`--legacy-eval`): `train_prefs[:500]` with a new subset of 50 pairs per chip, `G` = 50-350 µS without quantization, every `Linear` including the tied `lm_head`, and the Qwen system prompt; 5 chips at `alpha` = 0.008 (`--batch-size 1 --max-length 2048` to fit the laptop's memory).

| Configuration | Clean preference accuracy | Flip rate, `alpha` = 0.008 | Reported in [16] |
|---|---|---|---|
| Instruction-tuned | 56.0% | 23.7 ± 13.3% | about 30-33% |
| HARD-DPO checkpoint, adapter digital, base weights clipped at 3 sigma | 62.4% | 2.0 ± 2.7% | near 2% |

The tool reproduces the published flip rates under the released protocol. The checkpoint's clean preference accuracy is 62.4% on this pool but 55.1% on the held-out `test_prefs`: the legacy pool lies inside the `train_prefs[:10000]` that HARD-DPO trains on, which is why the default evaluation uses `test_prefs`.

Reproduce with `llm_main.py eval --pairs 100 --chips 3 --alphas 0 0.004 0.008 --max-length 1024 --low-memory`, adding `--clip-sigma 3` and `--experts` / `--peft-adapter ... --lora-alpha 128 --deploy merged` for the HARD-DPO rows; the laptop adapter comes from `llm_main.py train-hard-dpo --steps 100 --train-batch 2 --max-length 512 --max-prefs 2000 --gradient-checkpointing`. The released-script rows use `llm_main.py eval --legacy-eval --alphas 0.008 --chips 5 --batch-size 1 --max-length 2048 --low-memory`, with `--clip-sigma 3 --peft-adapter ... --lora-alpha 128 --deploy hybrid` for the checkpoint. One evaluation of 98 pairs takes about 1.5 minutes per chip on the M5.

### Fault-Tolerant Experts on ResNet18

**Setup**: `mof_main.py --steps 200 --chips 2` on the CIFAR-10 ResNet18 [13-15] above: programming-error, drift, stuck-at and read-noise experts (recipe `hat-fi-cnn`, rank 4 on all `Conv2d` layers, 200 steps each, batch 128), the stuck-at expert trained on the target chip (`--fault-seed 1`). Evaluation on the full test set with cumulative conditions: programming error `alpha_ind` = 0.03, relative drift 0.05, 4% stuck cells (2% at `G_max`, 2% at `G_min`) and read noise 0.02, 256-level cells, 2 chips per cell (mean accuracy; the chip-to-chip spread is below 1 point). The temperature expert and condition are skipped because no activation energy is assumed.

| Condition | Unprotected | MoF-LoRA (prog + drift + saf merged) | Sparse-MoF | Experts selected by Sparse-MoF |
|---|---|---|---|---|
| Clean (ideal 8-bit chip) | 94.99% | 88.67% | 94.76% | prog |
| + programming error | 93.64% | 88.31% | 93.53% | prog |
| + drift | 91.71% | 87.69% | 91.47% | prog, drift |
| + stuck-at faults | 10.01% | 89.94% | 89.57% | prog, drift, saf |
| + read noise | 10.06% | 89.85% | 89.45% | prog, drift, saf, read |

* The stuck-at faults drive the unprotected network to chance level; the stuck-at expert, trained on the chip's fault map, recovers it to about 90%, the "SAF cliff" and recovery reported in [18].
* The static MoF-LoRA merge costs 6 points on a chip without stuck-at faults, because the drift clipping and the stuck-at correction are applied even where they are not needed. Telemetry-guided selection avoids this: the experts are switched on only when the online test detects their non-ideality, so Sparse-MoF tracks the unprotected model on healthy chips and MoF-LoRA on faulty ones [18].
* At these mild soft-fault levels and 200 training steps, the drift and read-noise experts do not improve on the unprotected model (within 0.5 points), consistent with the moderate soft-fault gains in [18].
* The online test reads back what was injected (with these chip settings it measures a stuck-cell rate of 4.0% and a relative drift of 0.050), and the selection matched the injected non-idealities on every chip.

Reproduce with `python mof_main.py --dataset-path ./data --steps 200 --chips 2 --workers 2` (about 15 minutes on an Apple M5).

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

### Read Noise

* **Explanation**: Electron-conduction fluctuations make every read of a cell slightly different, so unlike programming error, read noise is drawn anew for every inference and does not change the stored state [18, 24].
* **Model** (`set_read_noise(alpha_read)`): on every forward pass each cell is read as
    ```
    G_read = G + N(0, (a * G)^2),    a = alpha_read * thermal_factor(T)
    ```
    The simulator turns the per-cell noise into the exact per-weight standard deviation of the read weight (over polarities and bit-slices) when the chip is programmed, and samples it in a forward hook; stored conductances are restored after each pass.
* **Basis**: State-proportional per-inference read noise as in HAT-FI [18], following the read-noise characteristics of ReRAM devices [24].

### Temperature

* **Explanation**: Thermally activated filament fluctuations increase conductance variability at higher temperature [18].
* **Model** (`set_temperature(T, Ea_eV, T0=25)`): programming error applied afterwards and read noise are scaled by the Arrhenius factor
    ```
    alpha(T) = alpha(T0) * exp(Ea / kB * (1/T0 - 1/T))      (T, T0 in kelvin)
    ```
    The activation energy `Ea` has no default and is required for any `T != T0`, since it depends on the device stack and HAT-FI does not report a value [18]. Relaxation and drift are not scaled.
* **Basis**: Arrhenius relation of HAT-FI [18, Eq. 3]; programming error that is reprogrammed at the operating temperature experiences the scaled noise.

---

## Alignment Robustness and Fault-Tolerant Experts

### Preference Flip Rate

* **Explanation**: Programming error perturbs the mapped projection weights and can erase the likelihood margin between a preferred response `y+` and a rejected response `y-`, so the deployed accelerator ranks `y-` above `y+` although the digital model does not: a preference inversion [16].
* **Model**: with the masked per-token average log-likelihood of the response tokens (prompt and template tokens excluded),
    ```
    l(y | x)  = 1/N(x, y) * sum_t M_t(x, y) * log pi(y_t | x, y_<t)
    m         = l(y+ | x) - l(y- | x)                      (digital model)
    m_hw      = same on the simulated chip
    flip rate = #{m > 0 and m_hw <= 0} / #{m > 0}           = P[m_hw <= 0 | m > 0]
    CI95      = 1.96 * sigma_MC / sqrt(N_chips)
    ```
    The digital reference is the model as deployed (merged experts and deployment clipping, no quantization or noise). Preference accuracy is the fraction of pairs with a positive margin. Every chip uses the same seed at every `alpha` and for every model, so models are compared on the same chips. Pairs come from the held-out `test_prefs` split of UltraFeedback [21, 22]; the chat template's default system prompt is used. Theorem 1 of [16] predicts `P[flip] = Phi(-m / (alpha * C))`, so flips concentrate on pairs with small clean margins and grow with `alpha`.
* **Mapped layers**: `q/k/v/o_proj` and `gate/up/down_proj`; attention scores, the KV cache, embeddings and `lm_head` stay in the digital path [16]. `--legacy-eval` reproduces the released evaluation scripts instead (see [How to Run](#llm-alignment-robustness-and-hard-dpo)).

### HARD-DPO

* **Explanation**: DPO [19] increases the clean margin but does not control its sensitivity to hardware perturbations. HARD-DPO optimizes the DPO objective under the noisy forward pass and adds a clean-to-noisy consistency term, with parameter-efficient LoRA [20] updates that are merged into the weights before mapping [16].
* **Model**: for every adapted projection `W = W0 + (alpha_LoRA / r) B A`,
    ```
    W~     = clip(W * exp(eps), -3 std(W), +3 std(W)),   eps ~ N(0, sigma^2)        (sign-preserving)
    Delta  = [l~(y+) - l~(y-)] - [l_ref(y+) - l_ref(y-)]                             (noisy policy, reference)
    L      = -log sigmoid(beta * Delta)  +  lambda_KD * tau^2 * mean_t KL(p_clean,t || p_noisy,t)   (on y+)
    ```
    Defaults (paper and released script): `sigma` = 0.008, `beta` = 0.1, `lambda_KD` = 0.5, `tau` = 1, rank 8, `lora_alpha` 16, AdamW with lr 5e-5, batch 4, 1000 steps on `train_prefs[:10000]`. Chosen and rejected responses share one noise sample per step. The reference policy is the same model with its experts switched off, so no second copy of the LLM is held. `--noise-target base` perturbs only `W0` and keeps the LoRA update clean, as the released script does; `--noise simulator` injects the simulator-exact programming error instead of the log-normal surrogate.
* **Deployment**: merged into the crossbar and clipped at 3 sigma (`--deploy merged --clip-sigma 3`, the paper's best setting), or kept digital (`--deploy hybrid`), optionally magnitude-pruned (`--prune`).

### MoF-LoRA and Sparse-MoF Experts

* **Explanation**: Different non-idealities need conflicting corrections, so one LoRA expert is trained per non-ideality. MoF-LoRA merges the programming-error, drift and stuck-at experts into the base weights at design time [17]. Sparse-MoF keeps all experts, including read-noise and temperature experts, as separate adapters and activates the relevant subset on the deployed chip from hardware telemetry, so the correction follows the device as it ages [18].
* **Expert training** (`mof.train_experts`, AdamW lr 3e-4, weight decay 1e-4, cosine annealing [17]); each expert trains only its own factors:

    | Expert | Builds on | Training perturbation |
    |---|---|---|
    | `prog` | `W0` | truncated Gaussian `TN(0, (alpha std(X))^2, +-3 sigma)`, on `W0 + dW` [18] or on `W0`, `A`, `B` separately [17] |
    | `drift` | `clip(W0 + dW_prog, +-3 std)` | state-proportional `sigma = alpha * |W|` [18] (none in [17], Algorithm 1) and `R_drift = alpha_R * sum max(0, |W2| - 3 std(W2))^2` |
    | `saf` | chain `W2`, masked | stuck-at map of the target chip from `fault_map()` (March-test equivalent); faulty weights frozen to the values they take on that chip, so the expert works around them |
    | `read` | `W0` | `sigma = alpha * |W|`, resampled every step |
    | `temp` | `W0` | truncated Gaussian with `alpha(T)` from the Arrhenius factor |

    Recipes hold the published coefficients: `hat-fi-llm` (rank 64, `lora_alpha` 128; `alpha` = 0.06 / 0.03 / 0.03 / 0.06 for prog / drift / read / temp, `p_SAF` = 2%) and `hat-fi-cnn` (rank 4; 0.5 / 0.3 / 0.3 / 0.5, `p_SAF` = 15%) [18], and `mof-lora` (rank 32, scale 1/r, 5% programming noise) [17]. The strength `alpha_R` of `R_drift` is not reported in the papers (default 1.0). `saf_exact=True` recomputes the frozen values from the current weights on every step, so a differential pair with one stuck cell follows its healthy partner exactly as on the chip.
* **Deployment modes**: `set_merge(model, True)` programs `W0` plus the active experts into the crossbar (MoF-LoRA, zero runtime overhead); `False` keeps `W0` on the crossbar and runs the active experts digitally (hybrid); `map_adapters=True` programs the expert factors into their own cells, where they see the same non-idealities (fully analog LoRA).
* **Sparse-MoF selection** (`mof.SparseMoF`): `telemetry()` is the online test of the chip: the stuck-cell rate (March test), the relative drift since the last write-verify, the relative read noise from repeated reads of a fixed sample of cells, and the temperature sensor. The thresholds are those of HAT-FI [18]:

    | Expert | Activated when |
    |---|---|
    | `prog` | always |
    | `drift` | measured relative drift >= 0.010 |
    | `saf` | measured stuck-cell rate >= 2% |
    | `read` | measured relative read noise >= 0.003 |
    | `temp` | temperature >= 35 C |

    When the active set changes, the experts are re-merged and the chip is reprogrammed: the stuck-at map is kept and programming error is drawn anew (`reprogram_params`). Drift and stuck-at detections are latched, because reprogramming resets the measured drift but not its cause. An optional accuracy probe on a calibration set is the secondary trigger: if accuracy falls more than `max_drop` below the baseline, each optional expert is toggled and kept if the probe improves.

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

If you use the alignment evaluation or HARD-DPO, please also cite [16]; for MoF-LoRA experts, [17]; for Sparse-MoF expert selection, read-noise and temperature experts, [18]:

```bibtex
@article{Ahmed2026HARDDPO,
  title   = {{HARD-DPO}: Hardware-Aware Robust Direct Preference Optimization for {RRAM}-based {LLM} Accelerators},
  author  = {Ahmed, Soyed Tuhin and Chakrabarty, Krishnendu},
  journal = {IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year    = {2026},
}

@article{Ahmed2026MoFLoRA,
  title   = {{MoF-LoRA}: Mixture of Low-Rank Fault-Tolerant Experts for {RRAM}-Based In-Memory Computing},
  author  = {Ahmed, Soyed Tuhin and Ortega, Eduardo and Xiao, Patrick and Feinberg, Ben and Bennett, Christopher H. and Marinella, Matthew J. and Chakrabarty, Krishnendu},
  journal = {IEEE Journal on Emerging and Selected Topics in Circuits and Systems},
  volume  = {16},
  number  = {2},
  pages   = {563--576},
  year    = {2026},
  doi     = {10.1109/JETCAS.2026.3655242},
}

@inproceedings{Sharma2026HATFI,
  title     = {{HAT-FI}: Hardware-Aware Training for Fault-Tolerant {LLM} Inference on {RRAM}-Based Compute-in-Memory},
  author    = {Sharma, Pragya and Ahmed, Soyed Tuhin and Firouzi, Farshad and Chakrabarty, Krishnendu},
  booktitle = {IEEE VLSI Test Symposium (VTS)},
  year      = {2026},
  doi       = {10.1109/VTS69484.2026.11563304},
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

[16] S. T. Ahmed and K. Chakrabarty, "HARD-DPO: Hardware-Aware Robust Direct Preference Optimization for RRAM-based LLM Accelerators," *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, 2026. Code: https://github.com/SoyedTuhinAhmed/HARD_DPO.

[17] S. T. Ahmed, E. Ortega, P. Xiao, B. Feinberg, C. H. Bennett, M. J. Marinella, and K. Chakrabarty, "MoF-LoRA: Mixture of Low-Rank Fault-Tolerant Experts for RRAM-Based In-Memory Computing," *IEEE Journal on Emerging and Selected Topics in Circuits and Systems*, vol. 16, no. 2, pp. 563-576, 2026, doi: 10.1109/JETCAS.2026.3655242.

[18] P. Sharma, S. T. Ahmed, F. Firouzi, and K. Chakrabarty, "HAT-FI: Hardware-Aware Training for Fault-Tolerant LLM Inference on RRAM-Based Compute-in-Memory," *IEEE VLSI Test Symposium (VTS)*, 2026, doi: 10.1109/VTS69484.2026.11563304.

[19] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

[20] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," *International Conference on Learning Representations (ICLR)*, 2022.

[21] G. Cui, L. Yuan, N. Ding, et al., "UltraFeedback: Boosting Language Models with Scaled AI Feedback," *International Conference on Machine Learning (ICML)*, 2024, arXiv:2310.01377.

[22] L. Tunstall, E. Beeching, N. Lambert, et al., "Zephyr: Direct Distillation of LM Alignment," *arXiv preprint arXiv:2310.16944*, 2023 (binarized UltraFeedback, https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized).

[23] Qwen Team, "Qwen2.5 Technical Report," *arXiv preprint arXiv:2412.15115*, 2024.

[24] K. Schnieders, C. Funck, et al., "Effect of Electron Conduction on the Read Noise Characteristics in ReRAM Devices," *APL Materials*, vol. 10, no. 10, 2022.
