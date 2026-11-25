# PyRRAM-Sim: A Toolkit for RRAM Non-Ideality Simulation

PyRRAM-Sim is a lightweight, PyTorch-based simulation tool designed for researchers and engineers working on Deep Neural Networks (DNNs) with RRAM-based Compute-in-Memory (CIM) architectures. It provides an efficient, modular framework to simulate key hardware non-idealities, allowing for the evaluation of model robustness before hardware deployment.

This toolkit simulates the sequential, cumulative effects of permanent stuck-at faults, programming errors, short-term conductance relaxation, long-term, state-proportional conductance drift, per-inference read noise and temperature. The models are based on phenomena described in leading academic research to ensure hardware-accurate simulation.

It reproduces analog compute-in-memory over a twelve-month orbital mission for space data centers: time-dependent PCM, ReRAM and SONOS device models, single-event upsets, total ionizing dose and temperature, an aihwkit-style tile periphery, and the mitigations of the COINS'26 case study [25].

It also evaluates and hardens instruction-tuned LLMs: the preference-inversion (flip) rate of aligned models under RRAM non-idealities and hardware-aware DPO (HARD-DPO) [16], and mixtures of fault-tolerant LoRA experts, merged at design time (MoF-LoRA) [17] or kept as separate adapters with telemetry-guided selection (Sparse-MoF) [18].

---

## What's New

* **Space data centers: analog CiM in orbit** (`src/spacecim/`, `space_main.py`, `space_reproduce.py`): the compute-in-memory case study of the COINS'26 Space Data Centers paper [25], natively in PyRRAM-Sim. A Presto remote-sensing transformer [32] on EuroSAT [33] runs on PCM, ReRAM (single device or differential pair) or SONOS crossbars through 1 s ... 12 months of drift, read noise, single-event upsets, total ionizing dose and temperature, with noise-aware LoRA, global drift compensation and on-orbit recalibration. The device models and the tile periphery are checked against IBM aihwkit 1.1.0, the simulator the paper used (see [Validation Results](#space-data-center-case-study-coins26)).
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
* **Space Missions**: Time-dependent PCM / ReRAM / SONOS crossbars with upsets, dose and temperature, drift compensation and recalibration over an orbital mission.
* **PyTorch Integration**: Seamlessly integrates with existing PyTorch models (including Hugging Face Transformers) and validation pipelines.

---

## Installation

The toolkit is contained within the `PyRRAMSimulator` class in `src/`. Simply add the `src` folder to your project. It requires Python 3.9+ and PyTorch; `torchvision` is only needed for the command-line script `main.py`:

```bash
pip install torch torchvision
pip install transformers datasets   # only for the LLM features (alignment.py, llm_main.py)
pip install einops                  # only for the space case study (Presto); rasterio pyproj to build the EuroSAT cache
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
| `src/spacecim/devices.py` | Time-dependent PCM, ReRAM (CMO/HfOx) and SONOS device models and conductance mappings |
| `src/spacecim/environment.py` | Single-event upsets, total ionizing dose, temperature |
| `src/spacecim/tile.py` | Tiled analog layers with DAC/ADC periphery, weight and output noise, global drift compensation |
| `src/spacecim/mission.py` | Mission timeline, recalibration, and the arms of the COINS'26 case study |
| `src/presto/` | Presto model [32] (MIT license), EuroSAT loader [33], fine-tuning and noise-aware LoRA |
| `space_main.py` | Command line for the space case study: `build-cache`, `finetune`, `lora`, `mission`, `summarize`, `plot` |
| `space_reproduce.py` | Runs the case-study arms over seeds on several GPUs |
| `scripts/sdc_coins26/` | Step-by-step shell scripts and tutorial reproducing the COINS'26 case study (download, cache, training, missions, comparison with the paper) |
| `src/spacecim/paper_coins26.json` | The paper's 10-seed results for every point, used by `space_main.py summarize` |

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

### Space Data Centers: Analog CiM Over an Orbital Mission

The case study of [25]: Presto [32] fine-tuned on EuroSAT multispectral [33] (official splits [34]), mapped onto analog crossbars and flown for twelve months. Data and weights: the EuroSAT_MS archive (`https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip` or Zenodo record 7711810), the split files `eurosat-{train,val,test}.txt` from the same Hugging Face dataset, and Presto's `default_model.pt` from `github.com/nasaharvest/presto` (`data/`). The shell scripts in [`scripts/sdc_coins26/`](scripts/sdc_coins26/README.md) do all of this step by step (tutorial included); the commands underneath are:

```bash
# 1. EuroSAT GeoTIFFs -> array cache (once; needs rasterio, pyproj)
python space_main.py build-cache --eurosat-root ./EuroSAT_MS --split-dir ./splits --cache-dir ./cache

# 2. Design time, per seed: FP32 fine-tune (E0), then noise-aware LoRA merged into the weights (E2)
python space_main.py finetune --cache-dir ./cache --presto-weights default_model.pt --seed 0 --out e0_seed0.pt
python space_main.py lora --cache-dir ./cache --checkpoint e0_seed0.pt --seed 0 --out e2_seed0.pt

# 3. Missions: any arm of the paper ({seed} is substituted in the checkpoint paths)
python space_main.py mission --cache-dir ./cache --arms E1a E3_noLoRA E4_noLoRA --seeds 0 \
    --e0 'e0_seed{seed}.pt' --e2 'e2_seed{seed}.pt' --compile --out results.jsonl
python space_main.py summarize results.jsonl

# All figures on 8 GPUs (one job = one arm x seed); fewer seeds for the sweeps
python space_reproduce.py --cache-dir ./cache --e0 'e0_seed{seed}.pt' --e2 'e2_seed{seed}.pt' \
    --figures fig2 fig3 fig5 fig6 fig7 fig8 --seeds 0 1 2 3 4 5 6 7 8 9 --figure-seeds fig6=0,1 fig7=0,1 fig8=0,1 \
    --gpus 0 1 2 3 4 5 6 7 --compile --out-dir results/
python space_main.py summarize results/*.jsonl
python space_main.py plot results/*.jsonl --out-dir figures/   # needs matplotlib
```

| Arms | Paper figure | Content |
|---|---|---|
| `E1a`, `E1b`, `E2`, `E3`, `E4` | Fig. 2 | Mitigation ladder on PCM: naive (without / with upsets at 1, 10, 100x), + noise-aware LoRA, + drift compensation, + recalibration |
| `E3_noLoRA`, `E4_noLoRA`, `E4_head` | Fig. 3, Sec. V-B | Leave-one-out of the adapter; head-only (deployable) recalibration |
| `E1b`, `E4` (10x, 100x endpoints) | Fig. 4 | Accuracy at 12 months against the upset rate |
| `P1Y`...`P4Y`, `R*Y`, `RP*Y`, `S*Y` | Fig. 5 | The ladder on PCM, ReRAM single device, ReRAM differential pair, SONOS |
| `D1`, `D4`, `RPD*`, `SD*`, `SX*` | Fig. 6 | Total ionizing dose per technology; `SX`: SONOS with the charge-loss channel |
| `D1`...`D4` | Fig. 7 | Dose sweep 0.1-3000 krad(Si), freshly programmed and after 12 months |
| `T1`...`T4` | Fig. 8 | Operating temperature -100 to 125 C, freshly programmed and after 12 months |

`--compile` fuses the tile MVM with `torch.compile` (about 2x faster on a GPU); if `/tmp` is mounted `noexec`, point `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a writable, executable folder. `--test-limit` evaluates on a class-stratified subset for quick checks. Checkpoints of the original case-study code (`best.pt`, `merged.pt`) load directly.

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

A twelve-month mission of any model on analog crossbars (every `nn.Linear` except the head is mapped; the device is PCM by default):

```python
from spacecim import mission as M

cfg = M.MissionConfig(device="reram_cmo_diff", device_kwargs={"g_max": 25.0, "g_min": 9.0},
                      seu_lambda=1e-6,                      # upsets / device / day
                      tid={"enabled": True, "dose_rate_krad_per_year": 1.0, "k_tid": 0.05},
                      thermal={"enabled": True, "T_op_C": 70.0},
                      drift_compensation=True, recalibrate=("layernorm", "head"))
evaluate = lambda m: {"accuracy": validate(m, dataloader, criterion, device)[1] / 100}
records = M.run_mission(my_model, cfg, seed=0, evaluate=evaluate,
                        calib_batches=lambda gen: calib_loader,          # labelled calibration batches
                        loss_on_batch=lambda m, b: criterion(m(b[0].to(device)), b[1].to(device)))
for r in records:
    print(r["t_label"], r["accuracy"], r["seu_hits"])       # 1s ... 12mo
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

### Space Data Center Case Study (COINS'26)

**Equivalence with aihwkit.** The case study of [25] was run with IBM aihwkit 1.1.0 [27]. The PyRRAM-Sim port was compared against that pipeline, built from the original case-study code and configuration, on the same weights (256 x 256 tiles, 4 programming draws per point):

| Check | PyRRAM-Sim | aihwkit 1.1.0 |
|---|---|---|
| Tile forward, all noise off (DAC, ADC, ABS_MAX, weight scaling) | relative error 8e-8 | reference |
| 600 x 300 layer split into 6 tiles | relative error 1e-7 | reference |
| PCM weight gain at 1 h / 5 d / 12 months | 0.770 / 0.611 / 0.496 | 0.771 / 0.611 / 0.497 |
| PCM residual weight error at 12 months (std) | 0.0267 | 0.0268 |
| ReRAM single device, residual std at 12 months | 0.144 | 0.144 |
| ReRAM differential pair, gain / residual std at 12 months | 0.830 / 0.076 | 0.830 / 0.076 |
| SONOS (log-law projection), gain at 12 months | 0.887 | 0.887 |
| Drift-compensation gain alpha at 12 months, PCM / ReRAM pair | 2.028 / 1.571 | 2.028 / 1.576 |
| Thermal activation (-40, 85, 125 C), TID charge loss (3 modes) | identical | reference |
| Upsets in 12 months at 1e-4 / device / day | 4,865 | Poisson mean 4,784 |

Every point agrees within the sampling spread; the deterministic parts agree to floating-point precision. Runs use different random streams, so accuracies match in distribution, not draw for draw.

**Reproduction.** All arms of Figs. 2-8 were rerun with PyRRAM-Sim on the case study's own fine-tuned (E0) and LoRA-merged (E2) Presto checkpoints. These are the same trained models, so the digital references are identical: 0.9571 FP32 and 0.9049 LoRA-merged, mean over 10 seeds. The run covered 188 mission jobs (arm x seed) on 8 RTX A5500 GPUs and took 2 h 10 min with `--compile`. Seeds: 10 for Figs. 2-4 (as in the paper), 5 for Fig. 5, and 2 for the dose and temperature sweeps of Figs. 6-8. Each cell reads **PyRRAM-Sim / paper**, the paper value being the mean of the paper's runs for the same seeds; accuracy is top-1 on the 5,400 EuroSAT test images.

*Fig. 2 and Fig. 3: mitigation ladder on PCM, 10 seeds*

| Configuration | 1 s | 1 h | 1 week | 1 month | 12 months |
|---|---|---|---|---|---|
| Baseline, no upsets | 0.865 / 0.868 | 0.273 / 0.236 | 0.145 / 0.127 | 0.134 / 0.122 | 0.123 / 0.118 |
| Baseline | 0.865 / 0.868 | 0.273 / 0.236 | 0.145 / 0.129 | 0.133 / 0.124 | 0.117 / 0.108 |
| + noise-aware LoRA | 0.864 / 0.868 | 0.298 / 0.282 | 0.141 / 0.136 | 0.133 / 0.129 | 0.121 / 0.125 |
| + drift compensation | 0.869 / 0.869 | 0.866 / 0.853 | 0.822 / 0.838 | 0.829 / 0.820 | 0.781 / 0.776 |
| + recalibration (fully mitigated) | 0.918 / 0.917 | 0.913 / 0.912 | 0.904 / 0.903 | 0.900 / 0.901 | 0.890 / 0.887 |
| Drift compensation only | 0.875 / 0.873 | 0.841 / 0.834 | 0.774 / 0.793 | 0.734 / 0.754 | 0.661 / 0.669 |
| Drift compensation + recalibration (best) | 0.941 / 0.941 | 0.935 / 0.935 | 0.921 / 0.922 | 0.915 / 0.914 | 0.900 / 0.900 |
| Fully mitigated, head-only recalibration | 0.900 / 0.899 | 0.898 / 0.892 | 0.887 / 0.886 | 0.884 / 0.883 | 0.868 / 0.867 |

*Fig. 4: accuracy at 12 months against the upset rate (lambda = 1e-6 per device per day), 10 seeds*

| 12 months, upset rate | 1 lambda | 10 lambda | 100 lambda |
|---|---|---|---|
| Baseline | 0.117 / 0.108 | 0.127 / 0.116 | 0.099 / 0.102 |
| LoRA + drift compensation | 0.781 / 0.776 | 0.592 / 0.519 | 0.128 / 0.111 |
| Drift compensation only | 0.661 / 0.669 | 0.406 / 0.391 | 0.126 / 0.104 |
| Fully mitigated | 0.890 / 0.887 | 0.839 / 0.831 | 0.462 / 0.441 |
| Drift compensation + recalibration | 0.900 / 0.900 | 0.827 / 0.816 | 0.409 / 0.397 |
| Head-only recalibration | 0.868 / 0.867 | 0.793 / 0.777 | 0.409 / 0.390 |

*Fig. 5: the ladder on each technology at 12 months, 5 seeds*

| 12 months | Naive | + LoRA | + drift compensation | + recalibration |
|---|---|---|---|---|
| PCM | 0.108 / 0.119 | 0.126 / 0.129 | 0.749 / 0.780 | 0.889 / 0.888 |
| ReRAM, single device | 0.125 / 0.133 | 0.165 / 0.132 | 0.193 / 0.148 | 0.562 / 0.602 |
| ReRAM, differential pair | 0.125 / 0.114 | 0.151 / 0.141 | 0.397 / 0.399 | 0.822 / 0.830 |
| SONOS (log-law projection) | 0.637 / 0.648 | 0.789 / 0.805 | 0.821 / 0.827 | 0.910 / 0.909 |

*Fig. 6: total ionizing dose per technology, freshly programmed, 2 seeds*

| Freshly programmed (1 s), dose krad(Si) | 1 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|
| PCM, naive | 0.887 / 0.881 | 0.814 / 0.865 | 0.723 / 0.732 | 0.440 / 0.489 | 0.205 / 0.223 |
| PCM, fully mitigated | 0.910 / 0.915 | 0.912 / 0.906 | 0.902 / 0.901 | 0.853 / 0.853 | 0.732 / 0.721 |
| ReRAM pair, naive | 0.884 / 0.874 | 0.886 / 0.882 | 0.862 / 0.807 | 0.696 / 0.794 | 0.362 / 0.497 |
| ReRAM pair, fully mitigated | 0.913 / 0.919 | 0.918 / 0.917 | 0.913 / 0.916 | 0.904 / 0.895 | 0.799 / 0.803 |
| SONOS, naive | 0.913 / 0.910 | 0.927 / 0.930 | 0.911 / 0.925 | 0.835 / 0.863 | 0.675 / 0.630 |
| SONOS, fully mitigated | 0.921 / 0.918 | 0.923 / 0.924 | 0.919 / 0.920 | 0.915 / 0.914 | 0.865 / 0.890 |
| SONOS + charge loss, naive | 0.917 / 0.905 | 0.109 / 0.110 | 0.109 / 0.109 | 0.103 / 0.103 | 0.103 / 0.103 |
| SONOS + charge loss, fully mitigated | 0.923 / 0.922 | 0.806 / 0.804 | 0.238 / 0.256 | 0.104 / 0.105 | 0.104 / 0.104 |

*Fig. 7: PCM dose sweep, freshly programmed and after 12 months, 2 seeds*

| PCM, dose krad(Si) | 1 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|
| Naive, fresh | 0.887 / 0.881 | 0.814 / 0.865 | 0.723 / 0.732 | 0.440 / 0.489 | 0.205 / 0.223 |
| Naive, 12 months | 0.109 / 0.109 | 0.109 / 0.109 | 0.159 / 0.109 | 0.109 / 0.160 | 0.104 / 0.202 |
| + LoRA, fresh | 0.876 / 0.865 | 0.841 / 0.846 | 0.780 / 0.795 | 0.587 / 0.624 | 0.240 / 0.249 |
| + LoRA, 12 months | 0.109 / 0.109 | 0.109 / 0.109 | 0.109 / 0.109 | 0.109 / 0.109 | 0.171 / 0.158 |
| + drift compensation, fresh | 0.867 / 0.866 | 0.838 / 0.845 | 0.825 / 0.786 | 0.684 / 0.614 | 0.146 / 0.230 |
| + drift compensation, 12 months | 0.715 / 0.758 | 0.742 / 0.695 | 0.600 / 0.718 | 0.423 / 0.414 | 0.325 / 0.272 |
| Fully mitigated, fresh | 0.910 / 0.915 | 0.912 / 0.906 | 0.902 / 0.901 | 0.853 / 0.853 | 0.732 / 0.721 |
| Fully mitigated, 12 months | 0.878 / 0.882 | 0.879 / 0.870 | 0.860 / 0.844 | 0.797 / 0.790 | 0.661 / 0.676 |

*Fig. 8: PCM operating temperature, freshly programmed and after 12 months, 2 seeds*

| PCM, operating temperature | -40 C | 0 C | 25 C | 55 C | 85 C | 125 C |
|---|---|---|---|---|---|---|
| Naive, fresh | 0.109 / 0.109 | 0.109 / 0.109 | 0.812 / 0.852 | 0.185 / 0.190 | 0.163 / 0.170 | 0.111 / 0.130 |
| Naive, 12 months | 0.109 / 0.109 | 0.109 / 0.109 | 0.109 / 0.109 | 0.462 / 0.370 | 0.160 / 0.156 | 0.143 / 0.146 |
| + LoRA, fresh | 0.109 / 0.109 | 0.109 / 0.109 | 0.842 / 0.821 | 0.280 / 0.274 | 0.156 / 0.153 | 0.111 / 0.114 |
| + LoRA, 12 months | 0.109 / 0.109 | 0.109 / 0.109 | 0.109 / 0.109 | 0.536 / 0.476 | 0.280 / 0.263 | 0.193 / 0.173 |
| + drift compensation, fresh | 0.112 / 0.106 | 0.491 / 0.641 | 0.843 / 0.821 | 0.635 / 0.621 | 0.308 / 0.334 | 0.191 / 0.197 |
| + drift compensation, 12 months | 0.096 / 0.092 | 0.289 / 0.302 | 0.816 / 0.718 | 0.618 / 0.593 | 0.503 / 0.467 | 0.382 / 0.401 |
| Fully mitigated, fresh | 0.343 / 0.333 | 0.866 / 0.866 | 0.917 / 0.912 | 0.868 / 0.870 | 0.707 / 0.715 | 0.630 / 0.639 |
| Fully mitigated, 12 months | 0.275 / 0.248 | 0.785 / 0.781 | 0.881 / 0.887 | 0.871 / 0.875 | 0.770 / 0.811 | 0.734 / 0.731 |

Over all 532 points (every arm, checkpoint, upset rate, dose and temperature), the mean difference from the paper is +0.001 (no bias) and the median absolute difference is 0.007. 97% of the points lie within two standard errors of the paper's seed spread and 99% within three, as expected for two samples of the same distribution. The largest differences are at points where every chip sits at chance level (about 0.11) and the paper's spread is close to zero, for example the naive chip at 100x upsets after 3 months (0.134 vs 0.106), or in the 2-seed sweeps, where each value rests on two chips. No group of sweep points is biased: the mean difference is between -0.009 and +0.005 for each of the PCM, ReRAM, SONOS dose and temperature groups. The one sweep point beyond three standard errors with a well-defined spread is SONOS, fully mitigated, at 300 krad(Si) (0.865 vs 0.890). The chip-to-chip spreads match as well: over the 77 ten-seed points, our standard deviation is a median 0.99 times the paper's. The exception is drift compensation alone (`E3_noLoRA`) at 12 months, where the mean matches but the spread is wider (0.117 vs 0.042).

Training reproduces too. Run through `scripts/sdc_coins26/03_train.sh`, the E0 fine-tuning recreated the paper's checkpoints bit for bit for seeds 0 and 1 (test accuracy 0.9624 and 0.9500). The reimplemented noise-aware LoRA lands within the paper's spread (0.911 and 0.898 vs 0.916 and 0.880). Flying Figs. 2-3 on these models gives a mean difference of -0.007 from the paper over 92 points, with 93% within two standard errors.

The findings of [25] hold in the reproduction:

* Drift alone takes the naive chip from 0.865 to 0.27 within an hour.
* Global drift compensation is the decisive mitigation (0.12 -> 0.66 at 12 months), and recalibration adds 11 to 24 points on top.
* Noise-aware LoRA helps before recalibration (0.78 vs 0.66) but not after it (0.890 vs 0.900).
* At 12 months, fully mitigated: SONOS 0.910 > PCM 0.889 > ReRAM pair 0.822 > ReRAM single device 0.562.
* Fully mitigated PCM keeps 0.85 up to 100 krad(Si) when freshly programmed (0.80 after 12 months). SONOS with the charge-loss channel collapses between 10 and 30 krad(Si) (0.81 -> 0.24).
* Fully mitigated PCM is best near the 25 C reference temperature (0.917 fresh) and loses accuracy toward both ends of the -100 to 125 C range.

`python space_main.py plot` draws the curves of Figs. 2 and 5-8 from the result files.

**Reproducing these results.** The scripts in [`scripts/sdc_coins26/`](scripts/sdc_coins26/README.md) run the whole study from an empty folder; their README is a step-by-step tutorial covering time per step, reusing existing checkpoints, reading the comparison, and troubleshooting:

```bash
bash scripts/sdc_coins26/00_setup.sh        # Python environment
bash scripts/sdc_coins26/01_download.sh     # EuroSAT_MS, official splits, Presto weights
bash scripts/sdc_coins26/02_build_cache.sh  # GeoTIFFs -> array cache
bash scripts/sdc_coins26/smoke_test.sh      # 3-minute check of the whole pipeline
bash scripts/sdc_coins26/03_train.sh        # E0 and E2 for seeds 0-9
bash scripts/sdc_coins26/04_missions.sh     # Figs. 2-8 (add --full for 10 seeds everywhere)
bash scripts/sdc_coins26/05_report.sh       # every point next to the paper's value, and the figures
```

To fly the paper's own checkpoints, as above, set `E0_PATTERN` and `E2_PATTERN` to them and skip `03_train.sh`. `space_main.py summarize` compares every point with the paper's values, which ship in `src/spacecim/paper_coins26.json`, and reports z-scores.

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

## Space Environment and Mission Models

These models (`src/spacecim/`) reproduce the compute-in-memory case study of [25]. A mission programs the crossbars once, on the ground, and reads them at checkpoints from 1 s to 12 months; each read starts from the programmed state at time `t`, while upsets accumulate from checkpoint to checkpoint.

### Devices and Cell Mapping

Each tile normalizes its weights so that `max|w| = 1` maps to `G_max`. Following aihwkit [27], the conductances are re-derived from the programmed weights at every read, so the idle leg of a differential pair reads back at its floor.

| Device | Mapping | Programming error | Time dependence | Basis |
|---|---|---|---|---|
| PCM | pair, `G` in [0, 25] µS | `sigma = 0.263 + 1.965 g - 1.173 g^2` µS, `g = G / G_max` | drift `G (t / t0)^(-nu)`, `nu ~ N(mu(g), sigma(g))` per device, `mu` in [0.049, 0.1]; 1/f read noise `Q_s(g) sqrt(ln((t + t_r) / 2 t_r))` | measured PCM [26], aihwkit `PCMLikeNoiseModel` [27] |
| ReRAM (CMO/HfOx) | single device over [9, 25] µS (offset-binary), or differential pair with the 9 µS floor on both legs | `0.0008 + 0.0011 G` µS (0.2 % program-and-verify) | absolute decay `-0.089 ln t` µS, spread `0.042 ln t + 0.41` µS, read noise | aihwkit `ReRamCMONoiseModel` [27] |
| SONOS | pair, [0, 16] µS at `V_read = 0.1` V | `20 (1 - exp(-I / 176 nA))` nA | measured retention polynomial and spread for 0-5 days; projected beyond under a log (default), power or linear law | 40 nm SONOS, 128K devices [30, 31] |

The mapping decides how a decay presents: an absolute conductance loss is an offset under the single-device mapping but a gain under the differential pair, which drift compensation can invert [25].

### Tile Periphery

256 x 256 tiles (balanced splits of larger layers), a 6-bit DAC with per-row ABS_MAX input scaling, additive weight read noise of 2 % of the tile's `w_max` per read, output noise 0.02, an 8-bit ADC with bound 12, digital bias, and the classifier head in digital logic. This is the configuration of [25] on the aihwkit torch tile [27].

### Space Environment

* **Single-event upsets**: per tile, `N_hits ~ Poisson(lambda N_devices dt)` between reads, `lambda = 1e-6` per device per day nominal. A hit redraws the conductance uniformly in `[G_min, G_max]` with probability 0.8, else leaves it stuck at a bound. For the charge-trap SONOS cell, a hit removes a random fraction of the stored charge instead. Upset cells keep their value for the rest of the mission [25, 37].
* **Total ionizing dose**: `dose(t) = offset + rate t`, 1 krad(Si) per year nominal. The read-noise sigma is broadened by `1 + 0.05 dose`. An optional charge-loss channel shifts conductances by a fraction, or by a fixed quantum `k_shift dose G_max` clamped at `G_min` (`absolute`), which shrinks the weights of a differential pair instead of scaling them [25, 36].
* **Temperature**: drift exponent `nu * T_op / T_ref`, read noise `(T_op / T_ref)^0.5`, and thermally activated conduction `G exp(-E_a / k_B (1/T_op - 1/T_ref))`. Here `E_a` is interpolated against `G / G_max` from 0.30 eV (amorphous) to 0.02 eV (crystalline), a state-dependent distortion of the weight map [25, 35].

### Mitigations

* **Noise-aware LoRA** (design time): `W' = W + (16 / 8) B A` on the attention and MLP projections, Gaussian noise of 0.1 `max|W'|` on the effective weight during training, merged before mapping [25, 20, 28].
* **Global drift compensation** (runtime, label-free): the identity matrix is applied as a probe after programming and at every read, and the tile outputs are scaled by `alpha = mean|ref| / mean|cur|` [29].
* **Recalibration** (runtime, labelled): at every checkpoint the digital LayerNorm affine terms and the head (3,850 parameters) are refit on 250 class-stratified validation images for 5 epochs (AdamW, lr 1e-3, batch 16), starting from the launch parameters. Gradients pass through the analog tiles as the ideal product with their present weights. `E4_head` fits the head only, which needs forward passes alone [25].

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

If you use the alignment evaluation or HARD-DPO, please also cite [16]; for MoF-LoRA experts, [17]; for Sparse-MoF expert selection, read-noise and temperature experts, [18]; for the space-environment mission models, [25]:

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

@inproceedings{Ahmed2026SDC,
  title     = {Space Data Centers ({SDCs}): Rethinking Computing Infrastructure Beyond Earth},
  author    = {Ahmed, Soyed Tuhin and Farahani, Bahar and Chakrabarty, Krishnendu and Firouzi, Farshad},
  booktitle = {IEEE International Conference on Omni Layer Intelligent Systems (COINS)},
  year      = {2026},
  doi       = {10.1109/COINS69995.2026.11715566},
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

[25] S. T. Ahmed, B. Farahani, K. Chakrabarty, and F. Firouzi, "Space Data Centers (SDCs): Rethinking Computing Infrastructure Beyond Earth," *IEEE International Conference on Omni Layer Intelligent Systems (COINS)*, 2026, doi: 10.1109/COINS69995.2026.11715566.

[26] S. R. Nandakumar, M. Le Gallo, I. Boybat, et al., "A phase-change memory model for neuromorphic computing," *Journal of Applied Physics*, vol. 124, no. 15, 2018.

[27] M. J. Rasch, D. Moreda, T. Gokmen, et al., "A flexible and fast PyTorch toolkit for simulating training and inference on analog crossbar arrays," *IEEE International Conference on Artificial Intelligence Circuits and Systems (AICAS)*, 2021 (IBM aihwkit, https://github.com/IBM/aihwkit, MIT license; version 1.1.0 used for validation).

[28] M. J. Rasch, C. Mackin, M. Le Gallo, et al., "Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators," *Nature Communications*, vol. 14, p. 5282, 2023.

[29] V. Joshi, M. Le Gallo, S. Haefeli, et al., "Accurate deep neural network inference using computational phase-change memory," *Nature Communications*, vol. 11, p. 2473, 2020.

[30] V. Agrawal, T. P. Xiao, C. H. Bennett, et al., "Subthreshold operation of SONOS analog memory to enable accurate low-power neural network inference," *IEEE International Electron Devices Meeting (IEDM)*, 2022, pp. 21.7.1-21.7.4.

[31] Sandia National Laboratories, "CrossSim: Accuracy simulation of analog in-memory computing," https://github.com/sandialabs/cross-sim.

[32] G. Tseng, R. Cartuyvels, I. Zvonkov, et al., "Lightweight, pre-trained transformers for remote sensing timeseries," *arXiv preprint arXiv:2304.14065*, 2023 (Presto, https://github.com/nasaharvest/presto, MIT license).

[33] P. Helber, B. Bischke, A. Dengel, et al., "EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification," *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217-2226, 2019.

[34] M. Neumann, A. S. Pinto, X. Zhai, et al., "Training general representations for remote sensing using in-domain knowledge," *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 2020, pp. 6730-6733.

[35] M. Le Gallo and A. Sebastian, "An overview of phase-change memory device physics," *Journal of Physics D: Applied Physics*, vol. 53, no. 21, p. 213002, 2020.

[36] X. Zhang, Q. Guo, D. Zhou, et al., "Total ionizing dose (TID) of phase change random access memory," *4th International Conference on Radiation Effects of Electronic Devices (ICREED)*, 2021, pp. 1-6.

[37] Z. Yao, L. Yue, F. Zhang, et al., "Single-event effects induced by heavy ions in 40 nm resistive random access memory," *IEEE 6th Advanced Information Technology, Electronic and Automation Control Conference (IAEAC)*, 2022, pp. 1437-1441.
