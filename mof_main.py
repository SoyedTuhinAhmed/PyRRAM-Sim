"""
Mixture of fault-tolerant LoRA experts on CIFAR-10 ResNet18 [MoF-LoRA, HAT-FI].

Trains the programming-error, drift, stuck-at, read-noise and temperature experts (or loads
them), then evaluates under cumulative RRAM non-idealities on the target chip:

    Unprotected    base model
    MoF-LoRA       prog + drift + saf experts merged into the crossbar at design time
    Sparse-MoF     experts selected per chip from telemetry, merged, chip reprogrammed
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import lora as L  # noqa: E402
import mof  # noqa: E402
from helper import validate  # noqa: E402
from main import CIFAR10_MEAN, CIFAR10_STD, cifar10_resnet18  # noqa: E402
from pytorchrram import PyRRAMSimulator  # noqa: E402

CONDITIONS = [
    ("clean", ()),
    ("+prog", ("prog",)),
    ("+prog+drift", ("prog", "drift")),
    ("+prog+drift+SAF", ("prog", "drift", "saf")),
    ("+read", ("prog", "drift", "saf", "read")),
    ("+temp (all 5)", ("prog", "drift", "saf", "read", "temp")),
]


def loaders(args):
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])
    train = torchvision.datasets.CIFAR10(args.dataset_path, train=True, download=True, transform=train_tf)
    test = torchvision.datasets.CIFAR10(args.dataset_path, train=False, download=True, transform=test_tf)
    if args.eval_images < len(test):
        test = torch.utils.data.Subset(test, range(args.eval_images))
    kw = dict(num_workers=args.workers, pin_memory=False)
    return (torch.utils.data.DataLoader(train, batch_size=args.train_batch, shuffle=True, drop_last=True, **kw),
            torch.utils.data.DataLoader(test, batch_size=500, shuffle=False, **kw))


class OnDevice:
    """Moves batches of a DataLoader to the training device."""

    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __iter__(self):
        for x, y in self.loader:
            yield x.to(self.device), y.to(self.device)


def chip_params(args, cond):
    stuck = ({"p_on": args.p_saf / 2, "p_off": args.p_saf / 2, "compensate": args.compensate,
              "fault_seed": args.fault_seed} if "saf" in cond and args.p_saf > 0 else None)
    prog = {"alpha_ind": args.prog_alpha} if "prog" in cond else None
    relax = {"alpha_ind": args.relax_alpha} if "drift" in cond else None
    drift = {"relative_drift": args.drift} if "drift" in cond else None
    return stuck, prog, relax, drift


def evaluate(sim, model, method, cond, args, testloader, device, trained):
    """Mean accuracy over --chips chips for one method and fault condition."""
    use_temp = "temp" in cond and args.temperature is not None
    sim.set_temperature(args.temperature if use_temp else 25.0, args.ea)
    sim.set_read_noise(args.read_alpha if "read" in cond else 0.0)
    stuck, prog, relax, drift = chip_params(args, cond)
    controller = None
    if method == "unprotected":
        L.set_active(model, [])
    elif method == "mof-lora":
        L.set_active(model, [e for e in mof.CHAIN if e in trained])
    else:
        controller = mof.SparseMoF(sim, deploy="merged")
        controller.reset()
    L.set_merge(model, True)
    accs, active = [], []
    for chip in range(args.chips):
        torch.manual_seed(args.seed + chip)
        if controller is not None:
            controller.reset()
        sim.program(stuck, prog, relax, drift)
        if controller is not None:
            controller.reprogram_params = {"programming_error_params": prog,
                                           "relaxation_params": relax, "drift_params": drift}
            active.append(controller.step()["active"])
        accs.append(validate(sim.analog_model, testloader, nn.CrossEntropyLoss(), device)[1])
    return float(np.mean(accs)), float(np.std(accs)), active


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--checkpoint", default=None, help="Own CIFAR-10 ResNet18 state dict (default: pretrained).")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    # experts
    p.add_argument("--load-experts", default=None, help="Expert checkpoint to evaluate instead of training.")
    p.add_argument("--save-experts", default=None)
    p.add_argument("--train", nargs="+", default=list(mof.EXPERT_ORDER), choices=list(mof.EXPERT_ORDER))
    p.add_argument("--recipe", default="hat-fi-cnn", choices=list(mof.RECIPES))
    p.add_argument("--steps", type=int, default=300, help="Training steps per expert.")
    p.add_argument("--train-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--train-temperature", type=float, default=None, help="Temperature for the temp expert (C).")
    p.add_argument("--reg-alpha", type=float, default=1.0)
    p.add_argument("--saf-exact", action="store_true")
    # chip and hardware
    p.add_argument("--g-min", type=float, default=1e-6)
    p.add_argument("--g-max", type=float, default=100e-6)
    p.add_argument("--n-levels", type=int, default=256)
    p.add_argument("--p-saf", type=float, default=0.04, help="Stuck-cell rate, half stuck-on and half stuck-off.")
    p.add_argument("--compensate", action="store_true")
    p.add_argument("--fault-seed", type=int, default=1, help="The target chip's fault map.")
    p.add_argument("--prog-alpha", type=float, default=0.03)
    p.add_argument("--relax-alpha", type=float, default=0.0)
    p.add_argument("--drift", type=float, default=0.05, help="Relative drift.")
    p.add_argument("--read-alpha", type=float, default=0.02)
    p.add_argument("--temperature", type=float, default=None, help="Hot operating temperature (C).")
    p.add_argument("--ea", type=float, default=None, help="Activation energy (eV); required with --temperature.")
    p.add_argument("--chips", type=int, default=3)
    p.add_argument("--eval-images", type=int, default=10000)
    p.add_argument("--methods", nargs="+", default=["unprotected", "mof-lora", "sparse-mof"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.temperature is not None and args.ea is None:
        raise SystemExit("--temperature needs --ea (activation energy in eV); HAT-FI reports no value.")
    if "temp" in args.train and (args.ea is None or args.train_temperature is None):
        print("Skipping the temperature expert: it needs --ea and --train-temperature.")
        args.train = [e for e in args.train if e != "temp"]
    torch.manual_seed(args.seed)
    trainloader, testloader = loaders(args)

    model = cifar10_resnet18(args.checkpoint).to(device).eval()
    L.add_lora(model, target_modules=None, exclude=("fc",))  # all Conv2d layers [HAT-FI]
    sim = PyRRAMSimulator(model, args.g_min, args.g_max, device=device, n_levels=args.n_levels, copy_model=False)

    if args.load_experts:
        L.load_expert_state_dict(model, torch.load(args.load_experts, map_location="cpu", weights_only=True)["experts"])
    else:
        if "saf" in args.train:
            sim.apply_stuck_at_faults(p_on=args.p_saf / 2, p_off=args.p_saf / 2, compensate=args.compensate,
                                      fault_seed=args.fault_seed)
        ce = lambda m, b: F.cross_entropy(m(b[0]), b[1])
        overrides = {"rank": args.rank} if args.rank else {}
        if args.rank:
            overrides["lora_alpha"] = args.rank
        mof.train_experts(model, ce, OnDevice(trainloader, device), args.steps, experts=args.train,
                          recipe=args.recipe, sim=sim, temperature_C=args.train_temperature, Ea_eV=args.ea,
                          saf_exact=args.saf_exact, reg_alpha=args.reg_alpha, lr=args.lr, **overrides)
        if args.save_experts:
            torch.save({"experts": L.expert_state_dict(model), "metadata": vars(args)}, args.save_experts)
            print(f"Saved experts to {args.save_experts}")
    trained = L.expert_names(model)

    conditions = [(n, c) for n, c in CONDITIONS if not ("temp" in c and args.temperature is None)]
    results = {}
    print(f"\n{'condition':<18}" + "".join(f"| {m:<22}" for m in args.methods))
    for name, cond in conditions:
        row = {}
        for method in args.methods:
            mean, std, active = evaluate(sim, model, method, cond, args, testloader, device, trained)
            row[method] = {"mean": mean, "std": std, "active": active}
        results[name] = row
        print(f"{name:<18}" + "".join(f"| {row[m]['mean']:6.2f} +- {row[m]['std']:4.2f}         " for m in args.methods))
        if "sparse-mof" in row:
            print(f"{'':<18}  Sparse-MoF experts per chip: {row['sparse-mof']['active']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results, "config": vars(args), "experts": trained}, f, indent=2)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
