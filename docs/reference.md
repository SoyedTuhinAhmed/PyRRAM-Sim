# Reference

The complete description of PyRRAM-Sim: files, command-line options, the Python API, and the physical models behind every simulated effect. For a gentle start, read the [tutorials](../README.md#tutorials) first. Numbers in brackets, such as [16], refer to the [list of references](references.md).

## Contents

* [Features](#features)
* [Files](#files)
* [Command-line options](#command-line-options): [`main.py`](#mainpy-a-resnet18-on-a-simulated-chip), [`mof_main.py`](#mof_mainpy-fault-tolerant-experts-on-cifar-10), [`llm_main.py`](#llm_mainpy-llm-alignment-and-experts), [`space_main.py`](#space_mainpy-space-data-centers)
* [Python API](#python-api)
* [Supported non-ideality models](#supported-non-ideality-models)
* [Alignment robustness and fault-tolerant experts](#alignment-robustness-and-fault-tolerant-experts)
* [Space environment and mission models](#space-environment-and-mission-models)

## Features

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

## Files

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
| `src/spacecim/mission.py` | Mission timeline, recalibration, and the configurations of the COINS'26 case study (E1a ... SX4) |
| `src/presto/` | Presto model [32] (MIT license), EuroSAT loader [33], fine-tuning and noise-aware LoRA |
| `space_main.py` | Command line for the space case study: `build-cache`, `finetune`, `lora`, `mission`, `summarize`, `plot` |
| `space_reproduce.py` | Runs the case-study configurations over seeds on several GPUs |
| `scripts/sdc_coins26/` | Step-by-step shell scripts and tutorial reproducing the COINS'26 case study (download, cache, training, missions, comparison with the paper) |
| `src/spacecim/paper_coins26.json` | The paper's 10-seed results for every point, used by `space_main.py summarize` |

## Command-line options

### `main.py`: a ResNet18 on a simulated chip

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

### `mof_main.py`: fault-tolerant experts on CIFAR-10

`mof_main.py` trains the five experts on CIFAR-10 ResNet18 (weight-space LoRA on all `Conv2d` layers [18]) for a target chip and compares the three deployments under cumulative non-idealities:

```bash
python mof_main.py --dataset-path ./data --steps 300 --save-experts resnet18_experts.pt \
    --temperature 85 --ea 0.1 --train-temperature 85
# re-evaluate saved experts, e.g. on a different fault rate
python mof_main.py --dataset-path ./data --load-experts resnet18_experts.pt --p-saf 0.02
```

`--temperature`, `--ea` and `--train-temperature` are needed only for the temperature expert and condition; without them those are skipped. The chip defaults are `--prog-alpha 0.03`, `--drift 0.05`, `--p-saf 0.04` (half stuck-on, half stuck-off, `--fault-seed 1`) and `--read-alpha 0.02`.

### `llm_main.py`: LLM alignment and experts

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

# A HARD-DPO checkpoint from the original PEFT scripts, adapters kept digital; --lora-alpha must match
# the checkpoint's training (the scripts' default is 16; the paper's checkpoint used 128)
python llm_main.py eval --clip-sigma 3 --peft-adapter noise_robust_dpo_qwen2.5_0.5b.pt --lora-alpha 128 --deploy hybrid

# Released-script protocol (train_prefs[:500], 50 pairs per chip, G = 50-350 uS unquantized,
# every Linear including lm_head) to reproduce previously published numbers
python llm_main.py eval --legacy-eval --alphas 0.008 --chips 5 --batch-size 1 --max-length 2048 --low-memory \
    --clip-sigma 3 --peft-adapter noise_robust_dpo_qwen2.5_0.5b.pt --lora-alpha 128 --deploy hybrid

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

### `space_main.py`: space data centers

The case study of [25]: Presto [32] fine-tuned on EuroSAT multispectral [33] (official splits [34]), mapped onto analog crossbars and flown for twelve months. Data and weights: the EuroSAT_MS archive (`https://hf.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip` or Zenodo record 7711810), the split files `eurosat-{train,val,test}.txt` from the same Hugging Face dataset, and Presto's `default_model.pt` from `github.com/nasaharvest/presto` (`data/`). The shell scripts in [`scripts/sdc_coins26/`](tutorials/5_space_ai.md) do all of this step by step (tutorial included); the commands underneath are:

```bash
# 1. EuroSAT GeoTIFFs -> array cache (once; needs rasterio, pyproj)
python space_main.py build-cache --eurosat-root ./EuroSAT_MS --split-dir ./splits --cache-dir ./cache

# 2. Design time, per seed: FP32 fine-tune (E0), then noise-aware LoRA merged into the weights (E2)
python space_main.py finetune --cache-dir ./cache --presto-weights default_model.pt --seed 0 --out e0_seed0.pt
python space_main.py lora --cache-dir ./cache --checkpoint e0_seed0.pt --seed 0 --out e2_seed0.pt

# 3. Missions: any configuration of the paper ({seed} is substituted in the checkpoint paths)
python space_main.py mission --cache-dir ./cache --configs E1a E3_noLoRA E4_noLoRA --seeds 0 \
    --e0 'e0_seed{seed}.pt' --e2 'e2_seed{seed}.pt' --compile --out results.jsonl
python space_main.py summarize results.jsonl

# All figures on 8 GPUs (one job = one configuration x one seed); fewer seeds for the sweeps
python space_reproduce.py --cache-dir ./cache --e0 'e0_seed{seed}.pt' --e2 'e2_seed{seed}.pt' \
    --figures fig2 fig3 fig5 fig6 fig7 fig8 --seeds 0 1 2 3 4 5 6 7 8 9 --figure-seeds fig6=0,1 fig7=0,1 fig8=0,1 \
    --gpus 0 1 2 3 4 5 6 7 --compile --out-dir results/
python space_main.py summarize results/*.jsonl
python space_main.py plot results/*.jsonl --out-dir figures/   # needs matplotlib
```

| Configurations | Paper figure | Content |
|---|---|---|
| `E1a`, `E1b`, `E2`, `E3`, `E4` | Fig. 2 | Mitigation ladder on PCM: naive (without / with upsets at 1, 10, 100x), + noise-aware LoRA, + drift compensation, + recalibration |
| `E3_noLoRA`, `E4_noLoRA`, `E4_head` | Fig. 3, Sec. V-B | Leave-one-out of the adapter; head-only (deployable) recalibration |
| `E1b`, `E4` (10x, 100x endpoints) | Fig. 4 | Accuracy at 12 months against the upset rate |
| `P1Y`...`P4Y`, `R*Y`, `RP*Y`, `S*Y` | Fig. 5 | The ladder on PCM, ReRAM single device, ReRAM differential pair, SONOS |
| `D1`, `D4`, `RPD*`, `SD*`, `SX*` | Fig. 6 | Total ionizing dose per technology; `SX`: SONOS with the charge-loss channel |
| `D1`...`D4` | Fig. 7 | Dose sweep 0.1-3000 krad(Si), freshly programmed and after 12 months |
| `T1`...`T4` | Fig. 8 | Operating temperature -100 to 125 C, freshly programmed and after 12 months |

`--compile` fuses the tile MVM with `torch.compile` (about 2x faster on a GPU). `--fused` is `--compile` plus a cheaper backward pass for on-orbit recalibration: each tile computes its analog output once and applies the ideal gradient only in the backward pass. Recalibration is about 15% faster (RTX A5500), evaluation is unchanged, and the gradients agree with the default ones to about 1e-5. If `/tmp` is mounted `noexec`, point `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to a writable, executable folder. `--test-limit` evaluates on a class-stratified subset for quick checks. Checkpoints of the original case-study code (`best.pt`, `merged.pt`) load directly.

## Python API

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

## Supported Non-ideality Models

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
* **Mapped layers**: `q/k/v/o_proj` and `gate/up/down_proj`; attention scores, the KV cache, embeddings and `lm_head` stay in the digital path [16]. `--legacy-eval` reproduces the released evaluation scripts instead (see [`llm_main.py`](#llm_mainpy-llm-alignment-and-experts)).

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
