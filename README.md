# PyRRAM-Sim

**Find out how a neural network behaves on an analog memory chip, before the chip exists.**

New AI chips store a network's weights in tiny memory cells (RRAM, PCM or SONOS) and compute right inside the memory, which saves a lot of energy. But these cells are imperfect: values are written with errors, they drift over time, some cells are broken, every read is noisy, and in space radiation hits them. PyRRAM-Sim simulates these effects for any PyTorch model, so you can measure how much accuracy is lost and test methods that win it back.

New to the topic? Start with **[Concepts in 10 minutes](docs/concepts.md)**.

## Tutorials

| I want to... | Tutorial | Paper | Needs |
|---|---|---|---|
| Run a network on a simulated chip and break some cells | [1. Your first chip simulation](docs/tutorials/1_first_simulation.md) | | any laptop, 10 min |
| Protect a network with fault-tolerant experts | [2. Fault-tolerant experts](docs/tutorials/2_fault_tolerant_experts.md) | MoF-LoRA, HAT-FI | laptop, 15 min |
| Check whether an LLM keeps its judgement on a chip (alignment) | [3. LLM alignment and HARD-DPO](docs/tutorials/3_llm_alignment.md) | HARD-DPO | 24 GB laptop or GPU, 45 min |
| Measure an LLM's performance on a faulty chip and recover it | [4. LLM performance with experts](docs/tutorials/4_llm_performance.md) | HAT-FI | GPU |
| Fly an AI chip through a year in orbit | [5. AI in space](docs/tutorials/5_space_ai.md) | Space Data Centers | NVIDIA GPUs, hours |

## Install

```bash
git clone https://github.com/SoyedTuhinAhmed/PyRRAM-Sim.git
cd PyRRAM-Sim
pip install torch torchvision          # everything
pip install transformers datasets      # tutorials 3 and 4 (language models)
pip install einops rasterio pyproj     # tutorial 5 (space)
```

Python 3.9 or newer.

## Try it in 2 minutes

```bash
python main.py --dataset-path ./data --n-levels 256
```

This downloads CIFAR-10 and a trained ResNet18, then runs the network twice: normally, and on a simulated 8-bit chip. Both give 95% accuracy, because the simulated chip has no errors yet. Now break 10% of its memory cells, as measured on a real test chip:

```bash
python main.py --dataset-path ./data --n-levels 256 --apply-stuck-at --p-on 0.0904 --p-off 0.0175 --fault-seed 1
```

The accuracy falls to 10%, which is random guessing. [Tutorial 1](docs/tutorials/1_first_simulation.md) explains why, and [tutorial 2](docs/tutorials/2_fault_tolerant_experts.md) shows how to fix it.

## Does it reproduce the papers?

| Paper | Result |
|---|---|
| **HARD-DPO** (IEEE TCAD 2026) | HARD-DPO removes **92%** of an LLM's preference flips on a noisy chip (paper: 94%) |
| **MoF-LoRA** (IEEE JETCAS 2026) and **HAT-FI** (IEEE VTS 2026) | Fault-tolerant experts bring a network with broken cells back from **10% to 90%** accuracy |
| **Space Data Centers** (IEEE COINS 2026) | **95%** of the paper's 532 results are reproduced within seed-to-seed variation |

We believe the small differences come from randomness: every simulated chip draws its own random errors (its *seed*), so two sets of chips never give exactly the same average. 95% is what a perfect reproduction scores, since even two runs of the same simulator agree on only about 95% of the points by this test. Figures and details: **[Reproducing the papers](docs/reproducing_the_papers.md)**.

## Documentation

| Page | What is in it |
|---|---|
| [Concepts](docs/concepts.md) | In-memory computing, every effect and method explained in plain words, glossary |
| [Tutorials](#tutorials) | Five step-by-step guides |
| [Reproducing the papers](docs/reproducing_the_papers.md) | How close each reproduction is, with figures |
| [Reference](docs/reference.md) | All files, command-line options, the Python API, and the physical models |
| [References](docs/references.md) | The papers the models are based on |

## Citing

If you use PyRRAM-Sim in your research, we kindly ask that you cite the following paper:

```bibtex
@article{Ahmed2025PyRRAMSim,
  title   = {PyRRAM-Sim: A Toolkit for RRAM Non-Ideality Simulation},
  author  = {Ahmed, Soyed Tuhin},
  year    = {2025},
}
```

If you use the alignment evaluation or HARD-DPO, please also cite HARD-DPO; for MoF-LoRA experts, MoF-LoRA; for Sparse-MoF expert selection, read-noise and temperature experts, HAT-FI; for the space mission models, the Space Data Centers paper:

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

## License

Apache License 2.0, see [LICENSE](LICENSE). The bundled Presto model code (`src/presto/`) is MIT-licensed, see `src/presto/LICENSE_PRESTO`.
