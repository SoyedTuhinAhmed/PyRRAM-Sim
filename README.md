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
