# Scripts: Space Data Centers (COINS'26) Case Study

These scripts reproduce the compute-in-memory case study of

> S. T. Ahmed, B. Farahani, K. Chakrabarty, F. Firouzi, "Space Data Centers (SDCs): Rethinking Computing Infrastructure Beyond Earth," IEEE COINS 2026, doi:10.1109/COINS69995.2026.11715566

**The step-by-step tutorial is [docs/tutorials/5_space_ai.md](../../docs/tutorials/5_space_ai.md).**

| Script | What it does | Time on 8 RTX A5500 GPUs |
|---|---|---|
| `00_setup.sh` | Python environment | a few minutes |
| `01_download.sh` | EuroSAT images, official splits, Presto weights (2 GB) | depends on the network |
| `02_build_cache.sh` | Reads the images once into a fast cache | about a minute |
| `smoke_test.sh` | Whole pipeline on 300 images, as a check | 3 minutes |
| `03_train.sh` | Trains the E0 and E2 models for 10 seeds | about 1.5 hours |
| `04_missions.sh` | Flies every configuration of Figs. 2-8 | about 2 hours (`--full`: 5.5 hours) |
| `05_report.sh` | Compares with the paper and draws figures | under a minute |
| `run_all.sh` | Steps 1-5 in a row | |
| `config.sh` | Shared settings (not run directly) | |

Every script can be rerun safely: finished work is kept and skipped.
