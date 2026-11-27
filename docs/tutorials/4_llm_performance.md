# Tutorial 4: An LLM on a Faulty Chip, Rescued by Experts (HAT-FI)

**Goal:** measure how well a language model still works on a chip with broken cells, drift and read noise, and win the performance back with fault-tolerant experts that the chip itself selects.
**Time:** an NVIDIA GPU is recommended for training the experts. Evaluation alone also runs on a 24 GB laptop.
**Papers:** HAT-FI [18] and MoF-LoRA [17] (see [references](../references.md)).
**Before this:** [Tutorial 2](2_fault_tolerant_experts.md) (experts on an image network) and [Tutorial 3](3_llm_alignment.md) (how LLM results are measured).

## The idea

This tutorial combines the previous two. The model is the LLM of tutorial 3; the protection is the expert method of tutorial 2:

* four experts (`prog`, `drift`, `saf`, `read`), each a small LoRA add-on trained against one problem;
* **Sparse-MoF**: the chip is tested (stuck-cell rate, drift, read noise) and only the experts it needs are switched on;
* **MoF-LoRA**: all experts merged into the weights, for comparison.

Performance is measured as in tutorial 3: **RRAM acc** (how often the model on the chip prefers the better answer) next to **clean acc** (the same on a normal computer), and the **flip rate**.

## The chip

All steps use the same simulated chip, given by these options:

| Option | Meaning |
|---|---|
| `--p-on 0.015 --p-off 0.015` | 3% broken cells: 1.5% stuck at the highest, 1.5% at the lowest conductance. HAT-FI switches its stuck-cell expert on from 2%, so 3% makes sure the chip test sees them |
| `--fault-seed 7` | which cells are broken (the same chip in every step) |
| `--alphas 0.004` | 0.4% programming error |
| `--drift 0.05` | 5% drift |
| `--read-alpha 0.005` | 0.5% read noise at every inference |

## Step 1: The unprotected model on this chip

```bash
python llm_main.py eval --pairs 100 --chips 3 --max-length 1024 --low-memory \
    --alphas 0.004 --p-on 0.015 --p-off 0.015 --fault-seed 7 --drift 0.05 --read-alpha 0.005 \
    --out unprotected.json
```

Write down `RRAM acc` and `flip rate`. Compare them with tutorial 3, where the chip had no broken cells, no drift and no read noise.

## Step 2: Train the experts

```bash
python llm_main.py train-experts --train prog drift saf read --recipe hat-fi-llm --objective dpo \
    --steps 500 --p-on 0.015 --p-off 0.015 --fault-seed 7 --out experts.pt
```

* `--recipe hat-fi-llm` uses the expert sizes and noise levels of the HAT-FI paper.
* `--objective dpo` trains the experts on preference pairs, the same task as in tutorial 3.
* The `saf` expert needs to know *which* cells are broken. It reads them from the chip given by `--p-on --p-off --fault-seed`, so use the same values as in Step 1.

Each expert trains for `--steps` steps. On a laptop, add `--train-batch 2 --max-length 512 --max-prefs 2000 --gradient-checkpointing` and lower `--steps` (for example 100) to fit in memory and time.

## Step 3: Let the chip choose its experts (Sparse-MoF)

```bash
python llm_main.py eval --pairs 100 --chips 3 --max-length 1024 --low-memory \
    --alphas 0.004 --p-on 0.015 --p-off 0.015 --fault-seed 7 --drift 0.05 --read-alpha 0.005 \
    --experts experts.pt --select-experts --out sparse_mof.json
```

For every simulated chip, a line like this is printed:

```
alpha=0.004 chip 1/3: flip=...% clean=...% hw=...% experts=['prog', 'drift', 'saf', 'read']
```

`experts=[...]` is what the chip test switched on. HAT-FI's rules: `prog` is always on; `drift` when the measured drift is at least 1%; `saf` when at least 2% of the cells are stuck; `read` when the read noise is at least 0.3%.

## Step 4: All experts always on (MoF-LoRA)

```bash
python llm_main.py eval --pairs 100 --chips 3 --max-length 1024 --low-memory \
    --alphas 0.004 --p-on 0.015 --p-off 0.015 --fault-seed 7 --drift 0.05 --read-alpha 0.005 \
    --experts experts.pt --active prog drift saf --out mof_lora.json
```

## Step 5: Compare

Put `RRAM acc` and `flip rate` of Steps 1, 3 and 4 side by side. The expected pattern, from the papers and from [tutorial 2](2_fault_tolerant_experts.md), is:

* Step 1 should be the worst.
* Steps 3 and 4 recover most of the gap to `clean acc`.
* On a *healthy* chip, Sparse-MoF should beat MoF-LoRA, because it does not apply corrections the chip does not need. Check this by rerunning Steps 3 and 4 without `--p-on --p-off --drift --read-alpha`.

If your results differ, look at the training length first: more `--steps` give stronger experts.

## Things to try

1. A temperature expert: train all experts plus `temp` with `--train prog drift saf read temp --train-temperature 85`, then evaluate with `--temperature 85 --ea 0.1` (`--ea` is the activation energy in eV; it depends on the device, so you choose it).
2. Keep the experts outside the chip: `--deploy hybrid` runs them digitally next to the crossbar, so they are not affected by its errors.
3. A different broken chip: evaluate with `--fault-seed 8`, while the `saf` expert was trained for chip 7.

All options: `python llm_main.py train-experts --help` and `python llm_main.py eval --help`.

**Next:** [Tutorial 5](5_space_ai.md) takes an AI chip to space.
