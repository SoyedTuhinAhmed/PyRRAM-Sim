# Concepts in 10 Minutes

This page explains the ideas behind PyRRAM-Sim in plain language. You need basic knowledge of neural networks (weights, layers, accuracy) and nothing else. Each tutorial links back here when a term first appears.

## 1. Computing inside memory

A neural network spends most of its time multiplying inputs by weights. On a normal computer (CPU, GPU), the weights live in memory and travel to the processor for every multiplication, and that traffic costs most of the energy.

An **in-memory computing chip** (also called compute-in-memory, CiM or IMC) does the multiplication where the weights are stored. The weights sit in a grid of tiny memory cells called a **crossbar**:

```
            input voltages
              V1    V2    V3
              |     |     |
   column 1 --G11---G12---G13--> current I1 = G11*V1 + G12*V2 + G13*V3
   column 2 --G21---G22---G23--> current I2 = G21*V1 + G22*V2 + G23*V3
```

Each cell has a **conductance** `G` (how easily current flows through it; measured in microsiemens, µS). Applying a voltage `V` gives a current `G * V` (Ohm's law), and the currents of a column add up on the wire (Kirchhoff's law). So every column computes a weighted sum, which is exactly what a layer of a neural network does, in one step and with very little energy.

## 2. How a weight becomes a conductance

* **Conductance range.** A cell can only take conductances between a minimum `G_min` and a maximum `G_max` (for example 1 to 100 µS). Each weight is scaled into this range.
* **Levels.** A cell holds only a limited number of distinct values. "8-bit" means 256 levels.
* **Two cells per weight.** Conductance cannot be negative, but weights can. So each weight is usually stored in a **differential pair**: `weight ∝ G+ − G−`.
* **Bit-slicing.** A weight can also be split over several cells, for example 8 bits over eight cells that each hold 1 bit.

## 3. What goes wrong (non-idealities)

Real memory cells are not perfect. These imperfections are called **non-idealities**, and PyRRAM-Sim simulates each of them:

| Non-ideality | What happens | When |
|---|---|---|
| **Programming error** | Writing a value is imprecise: the cell ends up near the target, not on it | Once, when the chip is programmed |
| **Relaxation** | Right after writing, the cell settles to a slightly different value | Seconds to minutes after programming |
| **Drift** | The conductance keeps slowly changing over time (in PCM it shrinks steadily) | Hours to years |
| **Stuck-at faults** | Some cells are broken and stuck at `G_max` or `G_min` from the factory; writing does not change them | Always (manufacturing defects) |
| **Read noise** | Every read gives a slightly different value, like a noisy sensor | Every time the network runs |
| **Temperature** | Heat makes the errors and the noise larger | Depends on the environment |

In space, two more appear (tutorial 5):

| Non-ideality | What happens |
|---|---|
| **Single-event upsets (SEU)** | A charged particle hits a cell and randomly changes its value |
| **Total ionizing dose (TID)** | Radiation slowly accumulates in the chip and makes it noisier (krad(Si) is its unit) |

Each one changes the weights a little, and the network loses accuracy. Most research in this area asks two questions: *how much accuracy is lost?* and *how can it be won back?*

## 4. Memory technologies

| Technology | How it stores a value | Known weakness |
|---|---|---|
| **RRAM / ReRAM** (resistive RAM) | A thin conductive filament inside an oxide | Programming error, stuck cells |
| **PCM** (phase-change memory) | A material that switches between glassy and crystalline | Strong drift over time |
| **SONOS** (charge-trap flash) | Electric charge trapped in a layer | Slowly loses charge; radiation removes charge |

## 5. How to fight back (mitigations)

| Method | Idea | Tutorial |
|---|---|---|
| **Fault-aware compensation** | If one cell of a pair is stuck, reprogram its partner to compensate | 1 |
| **LoRA** | Instead of retraining a whole network, train a small add-on: two thin matrices `B·A` added to each weight matrix. Cheap to train and to store | 2, 3, 4 |
| **Fault-tolerant experts** | Train one small LoRA add-on per problem (one for programming error, one for drift, one for stuck cells, ...) | 2, 4 |
| **MoF-LoRA** | Merge all experts into the weights before programming the chip | 2, 4 |
| **Sparse-MoF** (HAT-FI) | Test the chip first (how many stuck cells? how much drift?) and switch on only the experts it needs | 2, 4 |
| **Noise-aware training** | Add simulated chip noise during training, so the network learns to tolerate it | 3, 5 |
| **HARD-DPO** | Noise-aware training for the *alignment* of a language model (see section 6) | 3 |
| **Weight clipping** | Cut off the few very large weights before mapping, so the others use the conductance range better | 3 |
| **Drift compensation** | Measure how much the chip's outputs have shrunk, and scale them back up | 5 |
| **Recalibration** | Retrain a tiny part of the network (here the last layer and the normalization layers) on a few labeled examples while the chip is in use | 5 |

## 6. Language models and alignment

An **instruction-tuned** language model (LLM) has been trained to follow instructions and to prefer good answers over bad ones. This preference is called its **alignment**.

* A **preference pair** is a question with two answers, one marked better by people (from the UltraFeedback dataset).
* The model **prefers** the answer it finds more likely.
* **Preference accuracy:** how often the model prefers the better answer.
* A **flip** happens when the model prefers the better answer on a normal computer, but the worse answer on the chip. The **flip rate** is the share of such pairs.

A high flip rate means the chip silently changes the model's judgement: the model on the chip is no longer the model you tested.

## 7. Randomness, seeds and "reproduced"

* Every simulated error is random. A **seed** is the number that fixes the random draws, so one seed is one simulated chip, and the same seed gives exactly the same chip again.
* Real chips differ from each other too. So results are averaged over several chips (also called **Monte-Carlo** runs), and the **spread** (standard deviation) tells how much chips differ.
* A **95% confidence interval** is the range in which the true average lies with 95% probability.
* **"Reproduced within seed-to-seed variation"** means: our average differs from the paper's by less than the difference you would expect between two sets of random chips (less than two standard errors). Even two runs of the *same* simulator with different seeds pass this test only about 95% of the time, so "95% reproduced" is what a perfect reproduction looks like.

## 8. Glossary

| Term | Meaning |
|---|---|
| Accuracy | Share of test inputs the network classifies correctly |
| ADC / DAC | Converters between analog currents/voltages and digital numbers at the edge of a crossbar |
| Chance level | Accuracy of random guessing (10% for 10 classes) |
| Checkpoint | A saved model file (`.pt`, `.pth`) |
| CIFAR-10 | Small image dataset (10 classes) used in tutorials 1 and 2 |
| Conductance `G` | How easily current flows through a cell; stores the weight |
| Configuration | One experiment setting in a paper, e.g. `E4` = PCM chip with all three mitigations |
| Crossbar | Grid of memory cells that computes weighted sums |
| DPO | Direct Preference Optimization, a way to train an LLM on preference pairs |
| EuroSAT | Satellite image dataset (10 land-use classes) used in tutorial 5 |
| Expert | A small LoRA add-on trained against one non-ideality |
| krad(Si) | Unit of radiation dose absorbed by silicon |
| Presto | A small transformer for satellite data, used in tutorial 5 |
| Qwen2.5-0.5B-Instruct | A small instruction-tuned LLM used in tutorials 3 and 4 |
| ResNet18 | A convolutional image network used in tutorials 1 and 2 |
| Telemetry | Measurements of the chip's health (stuck cells, drift, noise, temperature) |
