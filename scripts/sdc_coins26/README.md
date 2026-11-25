# Reproducing the Space Data Centers (COINS'26) Case Study

This folder reproduces the compute-in-memory case study of

> S. T. Ahmed, M. Farahani, K. Chakrabarty, F. Firouzi, "Space Data Centers (SDCs): Rethinking Computing Infrastructure Beyond Earth," IEEE COINS 2026, doi:10.1109/COINS69995.2026.11715566

with PyRRAM-Sim. A Presto remote-sensing transformer, fine-tuned on EuroSAT, is mapped onto analog crossbars (PCM, ReRAM, SONOS) and flown through a 12-month orbital mission with conductance drift, read noise, single-event upsets, total ionizing dose and temperature. The study then measures how much accuracy noise-aware LoRA, global drift compensation and on-orbit recalibration recover. The scripts rebuild every number of Figs. 2-8 and compare it with the paper.

| Script | Step | Time on 8 RTX A5500 |
|---|---|---|
| `00_setup.sh` | Python environment (PyTorch, einops, rasterio, pyproj, matplotlib) | a few minutes |
| `01_download.sh` | EuroSAT_MS (2 GB), official splits, Presto weights | depends on the network |
| `02_build_cache.sh` | GeoTIFFs -> array cache (2.7 GB) | about a minute (CPU) |
| `smoke_test.sh` | The whole pipeline on 300 images, as a check | 3 minutes |
| `03_train.sh` | E0 (FP32 fine-tune) and E2 (noise-aware LoRA) for 10 seeds | about 1.5 h (35-50 min per seed) |
| `04_missions.sh` | The mission arms of Figs. 2-8 (188 jobs) | 2 h 10 min (17 GPU-hours) |
| `04_missions.sh --full` | 10 seeds for every figure (380 jobs) | about 5.5 h (42 GPU-hours) |
| `05_report.sh` | Comparison with the paper, figures | under a minute |
| `run_all.sh` | Steps 1-5 in a row | |

All scripts are bash and can be rerun safely: every step keeps what is already finished. A job interrupted by a reboot or a failure starts again from its beginning; nothing finished is recomputed.

## Quick start

```bash
git clone https://github.com/SoyedTuhinAhmed/PyRRAM-Sim.git && cd PyRRAM-Sim
bash scripts/sdc_coins26/00_setup.sh      # creates .venv (or PY=/path/to/python ... --no-venv)
bash scripts/sdc_coins26/run_all.sh       # download, cache, smoke test, train, missions, report
```

Everything is written to `sdc_work/` in the repository (about 8 GB; ignored by git). Set `SDC_WORK=/scratch/sdc` to put it elsewhere. Without a GPU, only the smoke test is practical; the full study needs one or more CUDA GPUs (24 GB was used; see [Troubleshooting](#troubleshooting) for smaller ones).

## Step by step

### 0. Environment

```bash
bash scripts/sdc_coins26/00_setup.sh
```

This creates `.venv/` and installs `torch numpy einops rasterio pyproj matplotlib`. For a specific CUDA build of PyTorch, install it first (https://pytorch.org/get-started/locally); the script keeps an existing torch. To use an environment you already have, install the same packages there and set `PY=/path/to/python` for every script (or run `00_setup.sh --no-venv` with `PY` set). The reproduction in the main README used Python 3.12, PyTorch 2.10 and CUDA 12.8.

### 1. Download the inputs

```bash
bash scripts/sdc_coins26/01_download.sh
```

| File | Source |
|---|---|
| `sdc_work/data/EuroSATallBands.zip` -> `eurosat_ms/` (27,000 GeoTIFFs, 13 Sentinel-2 bands, 64 x 64 px) | torchgeo's Hugging Face mirror, Zenodo record 7711810 as fallback |
| `sdc_work/data/splits/eurosat-{train,val,test}.txt` (16,200 / 5,400 / 5,400 images) | the official splits of Neumann et al., as used by Presto and torchgeo |
| `sdc_work/data/default_model.pt` | Presto's pretrained weights, `github.com/nasaharvest/presto` |

If you already have the archive, `EUROSAT_ZIP=/path/EuroSATallBands.zip bash .../01_download.sh` extracts it instead of downloading.

### 2. Build the cache

```bash
bash scripts/sdc_coins26/02_build_cache.sh
```

Reads every GeoTIFF once (raw reflectances, pixel coordinates, labels) into `sdc_work/cache/`. Presto's inputs (patch means, band layout, normalization, NDVI) are built from this cache on the GPU at run time. The cache is identical, byte for byte, to the one of the original case-study code.

### 3. Check the pipeline (3 minutes)

```bash
bash scripts/sdc_coins26/smoke_test.sh
```

Trains E0 and E2 for one epoch on 300 images, then flies the naive chip (`E1a`) and the fully mitigated chip (`E4`) on 300 test images and prints a summary. The accuracies are meaningless (the models barely trained); the test passes when it ends with `smoke test passed`. The naive chip falls to chance level (0.10) within a day, which is already the effect of Fig. 2.

### 4. Train the design-time models

```bash
bash scripts/sdc_coins26/03_train.sh                    # seeds 0-9, one seed per GPU at a time
bash scripts/sdc_coins26/03_train.sh --seeds 0,1 --gpus 0,1
```

For every seed `k` this writes

* `sdc_work/checkpoints/e0_seed<k>.pt`: Presto + linear head fine-tuned in FP32 (AdamW, lr 3e-4, weight decay 0.05, early stopping with patience 3). Paper: 0.957 +- 0.004 test accuracy.
* `sdc_work/checkpoints/e2_seed<k>.pt`: E0 trained further with noise-aware LoRA (rank 8, alpha 16 on qkv / proj / fc1 / fc2; Gaussian weight noise 0.1 max|W| in every forward pass), then merged into the weights. Paper: 0.905 +- 0.013.

The test accuracies are printed at the end (log: `sdc_work/logs/train_seed<k>.log`). E0 takes 11-18 epochs of about 2 minutes on an RTX A5500 before early stopping; E2 takes 8-20 minutes more. On the paper's setup (RTX A5500, PyTorch 2.10), the scripts reproduced the paper's E0 models exactly: test accuracy 0.9624 for seed 0 and 0.9500 for seed 1, the same as the original runs. The E2 models land within the paper's spread (0.911 and 0.898, against 0.916 and 0.880), since the LoRA training is a reimplementation. Other GPUs or library versions train slightly different models; the mission results then match the paper in distribution (see [What to expect](#what-to-expect)).

**Reusing existing checkpoints.** To fly the exact models of the paper (or any other), skip this step and point the missions at them. `{seed}` is replaced by the seed and globs are allowed:

```bash
export E0_PATTERN='/path/space-cim-case-study/results/e0_finetune/*_seed{seed}/checkpoints/best.pt'
export E2_PATTERN='/path/space-cim-case-study/results/e2_lora/*_seed{seed}/checkpoints/merged.pt'
```

### 5. Fly the missions

```bash
bash scripts/sdc_coins26/04_missions.sh                  # Figs. 2-8, default seed plan
bash scripts/sdc_coins26/04_missions.sh --full           # 10 seeds everywhere, as in the paper
bash scripts/sdc_coins26/04_missions.sh --figures "fig2 fig3" --seeds 0,1,2
```

Each job flies one arm with one seed: the chip is programmed at t = 0 and read at 1 s, 1 h, 1 day, 1 week, 1 month, 3, 6 and 12 months (or at 1 s and 12 months for the sweeps). At every checkpoint it is evaluated on the 5,400 test images. Jobs run one per GPU, the longest first. Results go to `sdc_work/results/<arm>_seed<k>.jsonl`, with a `.log` next to each and a `.done` marker once finished. Rerunning the script runs only the jobs without a `.done` marker. Follow the progress with `tail -f sdc_work/logs/missions_*.log`.

| Figure | Arms | What is varied | Default seeds | GPU-min per seed |
|---|---|---|---|---|
| Fig. 2 | `E1a`, `E1b`, `E2`, `E3`, `E4` | The mitigation ladder on PCM: naive without / with upsets, + noise-aware LoRA, + global drift compensation, + recalibration | 10 | 26 |
| Fig. 3 | `E3_noLoRA`, `E4_noLoRA`, `E4_head` | Without the adapter; recalibrating only the head | 10 | 16 |
| Fig. 4 | (in `E1b`, `E3`, `E3_noLoRA`, `E4`, `E4_noLoRA`, `E4_head`) | 12 months at 10x and 100x the nominal upset rate | 10 | - |
| Fig. 5 | `P1Y`-`P4Y`, `R1Y`-`R4Y`, `RP1Y`-`RP4Y`, `S1Y`-`S4Y` | The ladder on PCM, ReRAM single device, ReRAM differential pair, SONOS | 5 | 60 |
| Fig. 6 | `D1`, `D4`, `RPD1`, `RPD4`, `SD1`, `SD4`, `SX1`, `SX4` | Total ionizing dose 0.1-3000 krad(Si) per technology; `SX`: SONOS with charge loss | 2 | 67 |
| Fig. 7 | `D1`-`D4` | The PCM ladder against dose, fresh and after 12 months | 2 | 17 (+`D1`, `D4`) |
| Fig. 8 | `T1`-`T4` | Operating temperature -100 to 125 C, fresh and after 12 months | 2 | 64 |

Suffixes: `1` naive, `2` + noise-aware LoRA, `3` + drift compensation, `4` + recalibration (fully mitigated). The device and environment models behind each arm are described in the main README, section "Space Environment and Mission Models", and defined in `src/spacecim/mission.py` (`ARMS`).

### 6. Compare with the paper

```bash
bash scripts/sdc_coins26/05_report.sh
```

prints the headline points and writes

* `sdc_work/results/summary.txt` (and `summary.csv`): every point, as below;
* `sdc_work/figures/*.png`: accuracy against mission time (`timeline_ladder.png` for Fig. 2, `timeline_<technology>.png` for Fig. 5), dose (`dose_1s.png`, `dose_12mo.png`) and temperature (`temperature_1s.png`, `temperature_12mo.png`).

```
arm                    t     env   n    mean     std   paper   p.std      z
E0_digital       digital          10  0.9571  0.0041  0.9571  0.0041  -0.00
E4                  12mo          10  0.8899  0.0126  0.8873  0.0133  +0.43
E4@10x              12mo          10  0.8394  0.0363  0.8314  0.0275  +0.65
...
532 points compared with the paper: mean difference -0.0082, median |difference| 0.0098; |z| < 2 for 95%, |z| < 3 for 99%
```

`mean` and `std` are over your seeds and `paper` / `p.std` over the paper's 10 seeds (the reference values ship in `src/spacecim/paper_coins26.json`). `z` is the difference divided by its standard error, `p.std * sqrt(1/n + 1/10)`. If your chips follow the same distribution as the paper's, about 95% of the points have |z| < 2 and nearly all have |z| < 3. `env` is the dose in krad(Si) or the temperature in C for the sweep arms, and `@10x` marks the upset-rate multiplier.

You can also call the tools directly:

```bash
python space_main.py summarize sdc_work/results/*.jsonl --arms E4 E4_noLoRA   # selected arms
python space_main.py plot sdc_work/results/*.jsonl --out-dir figures
```

## What to expect

The reproduction in the main README (section "Space Data Center Case Study (COINS'26)") flew the paper's own checkpoints with the default seed plan. Every point is listed there. Headline results at 12 months:

| | PyRRAM-Sim | Paper (10 seeds) |
|---|---|---|
| FP32 / LoRA merged (digital) | 0.957 / 0.905 | 0.957 / 0.905 |
| Naive chip | 0.117 | 0.108 |
| Drift compensation | 0.661 | 0.669 |
| LoRA + drift compensation | 0.781 | 0.776 |
| Fully mitigated (LoRA + compensation + recalibration) | 0.890 | 0.887 |
| Fully mitigated at 10x / 100x upsets | 0.839 / 0.462 | 0.831 / 0.441 |
| Drift compensation + recalibration, no LoRA | 0.900 | 0.900 |
| Fully mitigated: SONOS / PCM / ReRAM pair / ReRAM single device | 0.910 / 0.889 / 0.822 / 0.562 | 0.911 / 0.888 / 0.829 / 0.592 |

* **Same checkpoints, same seeds** (as there): mean difference +0.001, 97% of the 532 points within two standard errors.
* **Your own trained models**: a full test of these scripts trained seeds 0 and 1 from scratch and flew Figs. 2-3. The E0 weights came out bit-identical to the paper's, and E2 within its spread. Digital accuracy was 0.956 and 0.905 (paper 0.957 and 0.905). Over the 92 mission points, the mean difference from the paper was -0.007, with 93% of points within two standard errors and 98% within three. Each trained model and each simulated chip is one draw, so with few seeds a single unusual draw moves a whole figure; compare with `z`, not by eye.
* **One known difference**: with drift compensation alone (`E3_noLoRA`) at 12 months, the mean matches the paper (0.661 vs 0.669 over 10 seeds), but the chips spread more (std 0.117 vs 0.042). The spreads agree at 1 month and in every other arm (median ratio 0.99 over the 77 ten-seed points). Seeds 0 and 1 happen to be the two lowest chips of this arm (0.465 and 0.638), so a 2-seed run reads low there.
* **Fewer seeds than the paper**: a 2-seed sweep point carries a standard error of about 0.8 times the paper's spread, so differences of 0.05-0.10 at single dose or temperature points are expected. Look for trends across neighbouring points instead.
* **Points at chance level** (about 0.10-0.11, for example every naive chip after a day) have almost no spread in the paper. There a difference of 0.02 already gives a large |z|; it is not a discrepancy.

## Running a single arm

The scripts are thin wrappers around two commands, which you can use directly:

```bash
# one arm, chosen seeds, into one file (appended)
python space_main.py mission --cache-dir sdc_work/cache --arms E4 E4_noLoRA --seeds 0 1 \
    --e0 'sdc_work/checkpoints/e0_seed{seed}.pt' --e2 'sdc_work/checkpoints/e2_seed{seed}.pt' \
    --compile --out my_results.jsonl

# a custom sweep: the PCM ladder at three doses only
python space_main.py mission ... --arms D4 --sweep-values 10 100 1000 --out dose.jsonl

# quick look: a class-stratified subset of the test set
python space_main.py mission ... --arms E1a --test-limit 540 --out quick.jsonl
```

Each line of a result file is one evaluation:

```json
{"t_label": "12mo", "t_seconds": 31536000.0, "accuracy": 0.8957, "macro_f1": 0.8916, "n_test": 5400,
 "seu_hits": 274, "n_devices": 791552, "recal_n_params": 3850, "gdc_alpha_mean": 2.0307, "arm": "E4",
 "seed": 0, "seu_multiplier": 1.0, "env_value": null, "endpoint_only": false, "mission_seed": 0, ...}
```

`seu_hits` counts the upsets accumulated since programming (out of `n_devices` conductances), `recal_n_params` is the number of parameters recalibration fits (LayerNorm + head), `gdc_alpha_mean` is the mean drift-compensation gain over the tiles, and `env_value` is the dose or temperature of the sweep arms. Lines with `"t_label": "digital"` hold the digital model's accuracy (arms `E0_digital`, `E2_digital`).

## Settings

All settings are environment variables, with defaults in `config.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `SDC_WORK` | `<repo>/sdc_work` | Root of everything written |
| `DATA_DIR`, `CACHE_DIR`, `CKPT_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `LOG_DIR` | under `SDC_WORK` | Individual folders |
| `PY` | `.venv/bin/python`, else `python3` | Interpreter |
| `GPUS` | all GPUs (`nvidia-smi`), or `CUDA_VISIBLE_DEVICES` | e.g. `GPUS="0 1 2 3"`; the scripts also take `--gpus 0,1` |
| `SEEDS` | `0 1 ... 9` | Seeds of Figs. 2-4 and of training |
| `FIGURE_SEEDS` | `fig5=0,1,2,3,4 fig6=0,1 fig7=0,1 fig8=0,1` | Fewer seeds for the expensive figures (`--full` clears it) |
| `FIGURES` | `fig2 fig3 fig5 fig6 fig7 fig8` | Figures to run |
| `E0_PATTERN`, `E2_PATTERN` | `$CKPT_DIR/e{0,2}_seed{seed}.pt` | Checkpoints to fly; globs allowed |
| `PRESTO_WEIGHTS`, `EUROSAT_ZIP` | under `DATA_DIR` | Use files you already have |
| `EVAL_BATCH` | 39 images | Lower it on GPUs with less memory |
| `TRAIN_ARGS` | none | Extra arguments for training, e.g. `--max-epochs 30` |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | under `SDC_WORK/.cache` | Where `torch.compile` writes its kernels |

## Troubleshooting

* **`torch.compile` fails with an ImportError or "Permission denied"** on a `.so` file: the kernel cache sits on a file system mounted `noexec` (often `/tmp`). The scripts already put it under `sdc_work/.cache`; if that is `noexec` too, set `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` to another folder, or run `04_missions.sh --no-compile` (about 2x slower).
* **CUDA out of memory** in the missions: `EVAL_BATCH=16 bash scripts/sdc_coins26/04_missions.sh`. Accuracy does not depend on the batch size.
* **A job failed**: its log is `sdc_work/results/<arm>_seed<k>.log`. Fix the cause and rerun `04_missions.sh`; only unfinished jobs run again.
* **No checkpoint for seed k**: `04_missions.sh` checks that both checkpoints exist for every seed it will use. Train them with `03_train.sh --seeds k`, or limit the seeds with `--seeds`.
* **The download fails**: fetch `EuroSATallBands.zip` (or `EuroSAT_MS.zip` from Zenodo) by hand and pass it as `EUROSAT_ZIP`. The cache builder accepts either archive's folder layout.
* **The login shell is csh / tcsh**: run the scripts with `bash script.sh` as shown; they do not depend on your login shell.
* **Two jobs per GPU** are slower than one (the evaluation saturates the GPU), so the scripts start one job per GPU.
