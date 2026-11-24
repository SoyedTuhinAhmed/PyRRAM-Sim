"""
PyRRAM-Sim for instruction-tuned LLMs:

    eval            preference accuracy and flip (preference-inversion) rate under RRAM
                    non-idealities [HARD-DPO], optionally with LoRA experts and telemetry-guided
                    expert selection [HAT-FI]
    train-hard-dpo  hardware-aware DPO with clean-noisy consistency [HARD-DPO]
    train-experts   fault-tolerant LoRA experts (prog, drift, saf, read, temp) [MoF-LoRA, HAT-FI]
"""
import argparse
import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import lora as L  # noqa: E402
import mof  # noqa: E402
from alignment import (  # noqa: E402
    QWEN_SYSTEM_PROMPT, HardDPOConfig, PreferenceScorer, batches, dpo_expert_loss,
    evaluate_flip_rate, load_ultrafeedback, sft_loss, train_hard_dpo,
)
from pytorchrram import PyRRAMSimulator  # noqa: E402
from train_noise import SimulatorNoise  # noqa: E402


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(args, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def load_experts(model, args):
    """Adds the expert checkpoint (--experts) and/or a PEFT adapter (--peft-adapter)."""
    if not (args.experts or args.peft_adapter):
        return []
    L.add_lora(model, args.target_modules)
    if args.experts:
        ckpt = torch.load(args.experts, map_location="cpu", weights_only=True)
        L.load_expert_state_dict(model, ckpt["experts"])
    if args.peft_adapter:
        ckpt = torch.load(args.peft_adapter, map_location="cpu", weights_only=True)
        L.load_peft_adapter(model, ckpt, args.peft_name, args.lora_alpha)
    names = args.active or L.expert_names(model)
    L.set_active(model, names)
    L.set_merge(model, args.deploy == "merged")
    if args.prune is not None:
        print(f"Pruned {100 * L.prune_experts(model, args.prune):.2f}% of the expert factors (|x| < {args.prune:g}).")
    return names


def build_sim(model, args, device, legacy=False):
    kw = dict(G_min=args.g_min, G_max=args.g_max, device=device, n_levels=args.n_levels,
              cells_per_weight=args.cells_per_weight, differential=not args.no_differential,
              diff_style=args.diff_style, target_modules=args.target_modules, exclude_modules=("lm_head",),
              clip_sigma=args.clip_sigma, map_adapters=args.deploy == "analog-lora",
              low_memory=args.low_memory, copy_model=False)
    if legacy:
        # Released HARD-DPO evaluation scripts: G = 50-350 uS, unquantized, every nn.Linear incl. lm_head.
        kw.update(G_min=50e-6, G_max=350e-6, n_levels=2 ** 20 + 1, cells_per_weight=1, differential=True,
                  diff_style="two_sided", target_modules=None, exclude_modules=())
        if device.type == "mps":
            kw["n_levels"] = 2 ** 23 + 1
    return PyRRAMSimulator(model, **kw)


def add_common(p):
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model id or path.")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", help="auto (bfloat16 on CUDA, else float32), float32, bfloat16.")
    p.add_argument("--system-prompt", default=None, help="System message; default: the chat template's own.")
    p.add_argument("--max-length", type=int, default=None, help="Truncate pairs to this many tokens.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-modules", nargs="+", default=list(L.DEFAULT_TARGET_MODULES),
                   help="Projections mapped to RRAM and adapted by LoRA.")
    p.add_argument("--out", default=None, help="Output file (JSON results or expert checkpoint).")
    # hardware
    p.add_argument("--g-min", type=float, default=1e-6)
    p.add_argument("--g-max", type=float, default=100e-6)
    p.add_argument("--n-levels", type=int, default=256)
    p.add_argument("--cells-per-weight", type=int, default=1)
    p.add_argument("--diff-style", default="two_sided", choices=["two_sided", "one_sided"])
    p.add_argument("--no-differential", action="store_true")
    p.add_argument("--clip-sigma", type=float, default=None, help="Clip weights to +-k std before mapping.")
    p.add_argument("--low-memory", action="store_true", help="Replay layer by layer; no per-cell state.")
    # experts
    p.add_argument("--experts", default=None, help="Expert checkpoint written by this tool.")
    p.add_argument("--peft-adapter", default=None, help="PEFT LoRA checkpoint (e.g. a HARD-DPO .pt).")
    p.add_argument("--peft-name", default="dpo")
    p.add_argument("--lora-alpha", type=float, default=16, help="lora_alpha of the PEFT adapter.")
    p.add_argument("--active", nargs="+", default=None, help="Experts to activate (default: all loaded).")
    p.add_argument("--deploy", default="merged", choices=["merged", "hybrid", "analog-lora"],
                   help="merged: experts programmed into the crossbar; hybrid: experts digital; "
                        "analog-lora: expert factors on their own RRAM cells.")
    p.add_argument("--prune", type=float, default=None, help="Zero-shot magnitude pruning threshold for expert factors.")
    # chip
    p.add_argument("--p-on", type=float, default=0.0)
    p.add_argument("--p-off", type=float, default=0.0)
    p.add_argument("--p-int", type=float, default=0.0)
    p.add_argument("--compensate", action="store_true")
    p.add_argument("--fault-seed", type=int, default=None)
    p.add_argument("--temperature", type=float, default=25.0, help="Operating temperature in Celsius.")
    p.add_argument("--ea", type=float, default=None, help="Activation energy in eV (required if temperature != 25).")


def stuck_params(args):
    if args.p_on + args.p_off + args.p_int <= 0:
        return None
    return {"p_on": args.p_on, "p_off": args.p_off, "p_int": args.p_int,
            "compensate": args.compensate, "fault_seed": args.fault_seed}


def cmd_eval(args):
    device = pick_device(args.device)
    model, tok = load_model(args, device)
    names = load_experts(model, args)
    legacy = args.legacy_eval
    if legacy:
        split, pool, subset, system = "train_prefs", 500, 50, QWEN_SYSTEM_PROMPT
    else:
        split, pool, subset, system = args.split, args.pairs, None, args.system_prompt
    pairs = load_ultrafeedback(split, pool, args.shuffle_seed)
    scorer = PreferenceScorer(tok, system, args.max_length, args.batch_size)
    sim = build_sim(model, args, device, legacy)
    sim.set_temperature(args.temperature, args.ea)
    if args.read_alpha:
        sim.set_read_noise(args.read_alpha)
    controller = mof.SparseMoF(sim) if args.select_experts else None
    if controller is not None and not names:
        raise SystemExit("--select-experts needs --experts.")
    relax = {"alpha_ind": args.relax_alpha} if args.relax_alpha else None
    drift = {"relative_drift": args.drift} if args.drift else None
    if drift and not relax:
        relax = {"alpha_ind": 0.0}
    print(f"Evaluating {args.model} on {len(pairs)} pairs of {split}; experts={names or 'none'} "
          f"deploy={args.deploy}; {len(sim._slot_list)} tensors mapped to RRAM.")
    res = evaluate_flip_rate(sim, scorer, pairs, args.alphas, args.chips, args.fault_model,
                             stuck_params(args), relax, drift, subset, controller, args.seed)
    print("\n alpha   | flip rate (%)      | clean acc (%) | RRAM acc (%)")
    for a in args.alphas:
        r = res["per_alpha"][str(a)]
        print(f" {a:<7g} | {r['flip_rate']['mean']:6.2f} +- {r['flip_rate']['ci95']:5.2f} | "
              f"{r['clean_acc']['mean']:6.2f}        | {r['hw_acc']['mean']:6.2f}")
    if args.out:
        res["config"] = {k: v for k, v in vars(args).items() if k != "fn"}
        res["experts"] = names
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {args.out}")


def _save_experts(model, args, extra):
    out = args.out or f"experts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({"experts": L.expert_state_dict(model), "target_modules": args.target_modules,
                "metadata": {"model": args.model, **extra}}, out)
    print(f"Saved experts {L.expert_names(model)} to {out}")


def cmd_train_hard_dpo(args):
    device = pick_device(args.device)
    model, tok = load_model(args, device)
    pairs = load_ultrafeedback(args.split, args.max_prefs, args.shuffle_seed)
    scorer = PreferenceScorer(tok, args.system_prompt, args.max_length, args.batch_size)
    cfg = HardDPOConfig(sigma=args.sigma, clip_k=args.clip_k, noise_target=args.noise_target, beta=args.beta,
                        lambda_kd=args.lambda_kd, kd_tau=args.kd_tau, rank=args.rank, lora_alpha=args.lora_alpha,
                        lr=args.lr, steps=args.steps, batch_size=args.train_batch, seed=args.seed)
    noise = None
    if args.noise == "simulator":
        L.add_lora(model, args.target_modules)
        sim = build_sim(model, args, device)
        noise = SimulatorNoise(sim, alpha_ind=args.sigma)
    hist = train_hard_dpo(model, scorer, pairs, cfg, target_modules=args.target_modules,
                          gradient_checkpointing=args.gradient_checkpointing, noise=noise)
    _save_experts(model, args, {"stage": "hard_dpo", "config": vars(cfg), "noise": args.noise, "history": hist})


def cmd_train_experts(args):
    device = pick_device(args.device)
    model, tok = load_model(args, device)
    pairs = load_ultrafeedback(args.split, args.max_prefs, args.shuffle_seed)
    scorer = PreferenceScorer(tok, args.system_prompt, args.max_length, args.batch_size)
    L.add_lora(model, args.target_modules)
    if args.experts:
        ckpt = torch.load(args.experts, map_location="cpu", weights_only=True)
        L.load_expert_state_dict(model, ckpt["experts"])
    sim = None
    if "saf" in args.train:
        if stuck_params(args) is None:
            raise SystemExit("The SAF expert needs the target chip's faults: give --p-on/--p-off and --fault-seed.")
        sim = build_sim(model, args, device)
        sim.apply_stuck_at_faults(**stuck_params(args))
    cfg = HardDPOConfig(beta=args.beta, lambda_kd=args.lambda_kd, kd_tau=args.kd_tau)
    loss_fn = dpo_expert_loss(scorer, cfg) if args.objective == "dpo" else sft_loss(scorer)
    data = batches(scorer.encode(pairs), args.train_batch, args.seed)
    overrides = {k: v for k, v in (("rank", args.rank), ("lora_alpha", args.expert_alpha),
                                   ("prog", args.prog), ("drift", args.drift_alpha),
                                   ("read", args.read), ("temp", args.temp)) if v is not None}
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hist = mof.train_experts(model, loss_fn, data, args.steps, experts=args.train, recipe=args.recipe,
                             sim=sim, temperature_C=args.train_temperature, Ea_eV=args.ea,
                             saf_exact=args.saf_exact, reg_alpha=args.reg_alpha, lr=args.lr,
                             train_mode=args.gradient_checkpointing, **overrides)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    _save_experts(model, args, {"stage": "mof", "recipe": args.recipe, "objective": args.objective,
                                "stuck_at": stuck_params(args), "history": hist})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Flip rate under RRAM non-idealities.")
    add_common(e)
    e.add_argument("--split", default="test_prefs")
    e.add_argument("--pairs", type=int, default=500, help="Number of preference pairs.")
    e.add_argument("--shuffle-seed", type=int, default=None)
    e.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.002, 0.004, 0.006, 0.008])
    e.add_argument("--chips", type=int, default=10, help="Monte-Carlo chips per alpha.")
    e.add_argument("--fault-model", default="state_independent",
                   choices=["state_independent", "state_proportional", "both"])
    e.add_argument("--relax-alpha", type=float, default=0.0)
    e.add_argument("--drift", type=float, default=0.0, help="Relative drift.")
    e.add_argument("--read-alpha", type=float, default=0.0, help="Per-inference read noise.")
    e.add_argument("--select-experts", action="store_true", help="Sparse-MoF telemetry-guided selection per chip.")
    e.add_argument("--legacy-eval", action="store_true",
                   help="Reproduce the released HARD-DPO scripts: train_prefs[:500], 50 pairs per chip, "
                        "G 50-350 uS unquantized, every nn.Linear incl. lm_head, Qwen system prompt.")
    e.set_defaults(fn=cmd_eval)

    t = sub.add_parser("train-hard-dpo", help="HARD-DPO training.")
    add_common(t)
    t.add_argument("--split", default="train_prefs")
    t.add_argument("--max-prefs", type=int, default=10000)
    t.add_argument("--shuffle-seed", type=int, default=None)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--train-batch", type=int, default=4)
    t.add_argument("--sigma", type=float, default=0.008, help="Training noise (matched to alpha_prog).")
    t.add_argument("--noise", default="lognormal", choices=["lognormal", "simulator"])
    t.add_argument("--noise-target", default="merged", choices=["merged", "base"])
    t.add_argument("--clip-k", type=float, default=3.0)
    t.add_argument("--beta", type=float, default=0.1)
    t.add_argument("--lambda-kd", type=float, default=0.5)
    t.add_argument("--kd-tau", type=float, default=1.0)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--gradient-checkpointing", action="store_true")
    t.set_defaults(fn=cmd_train_hard_dpo)

    x = sub.add_parser("train-experts", help="MoF-LoRA / Sparse-MoF experts.")
    add_common(x)
    x.add_argument("--train", nargs="+", default=["prog", "drift", "saf"], choices=list(mof.EXPERT_ORDER))
    x.add_argument("--recipe", default="hat-fi-llm", choices=list(mof.RECIPES))
    x.add_argument("--objective", default="dpo", choices=["dpo", "sft"])
    x.add_argument("--split", default="train_prefs")
    x.add_argument("--max-prefs", type=int, default=10000)
    x.add_argument("--shuffle-seed", type=int, default=None)
    x.add_argument("--steps", type=int, default=500, help="Steps per expert.")
    x.add_argument("--train-batch", type=int, default=4)
    x.add_argument("--lr", type=float, default=3e-4)
    x.add_argument("--rank", type=int, default=None)
    x.add_argument("--expert-alpha", type=float, default=None, help="lora_alpha of the new experts.")
    x.add_argument("--prog", type=float, default=None)
    x.add_argument("--drift-alpha", type=float, default=None)
    x.add_argument("--read", type=float, default=None)
    x.add_argument("--temp", type=float, default=None)
    x.add_argument("--train-temperature", type=float, default=None, help="Temperature of the temp expert (C).")
    x.add_argument("--reg-alpha", type=float, default=1.0, help="R_drift strength.")
    x.add_argument("--saf-exact", action="store_true")
    x.add_argument("--gradient-checkpointing", action="store_true")
    x.add_argument("--beta", type=float, default=0.1)
    x.add_argument("--lambda-kd", type=float, default=0.5)
    x.add_argument("--kd-tau", type=float, default=1.0)
    x.set_defaults(fn=cmd_train_experts)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    args.fn(args)


if __name__ == "__main__":
    main()
