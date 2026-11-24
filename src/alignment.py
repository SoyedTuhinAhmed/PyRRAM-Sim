"""
Preference-alignment robustness of instruction-tuned LLMs on RRAM compute-in-memory.

For a preference pair (x, y+, y-) the clean margin is m = l(y+|x) - l(y-|x), with l the
masked per-token average log-likelihood of the response tokens [HARD-DPO, Eq. 23]. A pair
flips (preference inversion) when the clean model ranks it correctly, m > 0, but the RRAM
model does not, m_hw <= 0. The flip rate is P[m_hw <= 0 | m > 0] over the clean-correct
pairs [HARD-DPO, Eq. 14, Theorem 1], estimated over Monte-Carlo chips with
CI95 = 1.96 * sigma_MC / sqrt(N).

HARD-DPO [HARD-DPO] trains a LoRA expert with DPO under the noisy forward pass, plus a
clean-to-noisy KL consistency term on the preferred response:
L = L_DPO(noisy policy) + lambda_KD * KL(p_clean || p_noisy).
"""
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from lora import (
    DEFAULT_TARGET_MODULES, add_expert, add_lora, expert_names, experts_disabled,
    lora_modules, noise_context, set_active, set_trainable,
)
from train_noise import LogNormalNoise

QWEN_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


def load_ultrafeedback(split: str = "test_prefs", n: Optional[int] = None, shuffle_seed: Optional[int] = None,
                       dataset: str = "HuggingFaceH4/ultrafeedback_binarized") -> list:
    """UltraFeedback binarized preference pairs (the final assistant turn of chosen / rejected)."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    return [PreferencePair(str(r["prompt"]), str(r["chosen"][-1]["content"]), str(r["rejected"][-1]["content"]))
            for r in ds]


def empty_cache(device):
    """Returns cached allocator blocks (variable-length batches fragment the cache), so that the
    next chip's mapping of large matrices finds contiguous memory."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def _body_and_head(model):
    body = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    return body, model.get_output_embeddings()


def _logits(head, h, model):
    logits = head(h).float()
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:  # Gemma 2 caps its logits in the LM head
        logits = torch.tanh(logits / cap) * cap
    return logits


class PreferenceScorer:
    """
    Tokenizes preference pairs with the model's chat template and scores responses with the
    masked per-token average log-likelihood. system_prompt=None uses the template's default.
    """

    def __init__(self, tokenizer, system_prompt: Optional[str] = None, max_length: Optional[int] = None,
                 batch_size: int = 8, logit_chunk: int = 64, pad_multiple: int = 64):
        self.tok = tokenizer
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.system_prompt = system_prompt
        self.max_length = max_length
        self.batch_size = batch_size
        self.logit_chunk = logit_chunk
        self.pad_multiple = max(1, pad_multiple)

    def _encode_one(self, prompt: str, response: str):
        msgs = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else []) + \
               [{"role": "user", "content": prompt}]
        ctx = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(msgs + [{"role": "assistant", "content": response}],
                                            tokenize=False, add_generation_prompt=False)
        ctx_ids = self.tok(ctx, add_special_tokens=False).input_ids
        ids = self.tok(full, add_special_tokens=False).input_ids
        start = len(ctx_ids)
        if self.max_length is not None and len(ids) > self.max_length:
            ids = ids[:self.max_length]
        if start >= len(ids):
            return None
        return ids, start

    def encode(self, pairs: Sequence[PreferencePair]) -> list:
        """Returns [(chosen_ids, chosen_start, rejected_ids, rejected_start)]; pairs whose prompt
        fills max_length are dropped."""
        out = []
        for p in pairs:
            c, r = self._encode_one(p.prompt, p.chosen), self._encode_one(p.prompt, p.rejected)
            if c is not None and r is not None:
                out.append((c[0], c[1], r[0], r[1]))
        if len(out) < len(pairs):
            print(f"[PreferenceScorer] dropped {len(pairs) - len(out)} pairs whose prompt exceeds max_length")
        return out

    def _batch(self, seqs, device):
        """Right-padded ids, attention mask and response mask for [(ids, start)]. Lengths are
        padded to a multiple of pad_multiple, which bounds the number of distinct shapes (and
        compiled kernels / graphs); right padding does not change causal-LM scores."""
        T = max(len(ids) for ids, _ in seqs)
        T = -(-T // self.pad_multiple) * self.pad_multiple
        pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        resp = torch.zeros((len(seqs), T), dtype=torch.bool)
        for i, (s, start) in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
            resp[i, start:len(s)] = True
        return ids.to(device), attn.to(device), resp.to(device)

    def sequence_logps(self, model, ids, attn, resp, chunk=None):
        """Masked per-token average log-likelihood of the response tokens, one value per row."""
        body, head = _body_and_head(model)
        h = body(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state[:, :-1]
        labels, mask = ids[:, 1:], resp[:, 1:]
        chunk = chunk or self.logit_chunk
        total = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.float32)
        for i in range(0, h.shape[1], chunk):
            m = mask[:, i:i + chunk]
            if not bool(m.any()):
                continue
            logp = F.log_softmax(_logits(head, h[:, i:i + chunk], model), dim=-1)
            tok = logp.gather(2, labels[:, i:i + chunk].unsqueeze(2)).squeeze(2)
            total = total + (tok * m).sum(1)
        return total / mask.sum(1).clamp(min=1)

    @torch.inference_mode()
    def score(self, model, encoded: list, device=None):
        """(log-likelihood of chosen, of rejected) for every encoded pair."""
        device = device or next(model.parameters()).device
        model.eval()
        lc, lr = [], []
        for i in range(0, len(encoded), self.batch_size):
            chunk = encoded[i:i + self.batch_size]
            for side, store in ((0, lc), (2, lr)):
                ids, attn, resp = self._batch([(e[side], e[side + 1]) for e in chunk], device)
                store.append(self.sequence_logps(model, ids, attn, resp).float().cpu())
        empty_cache(device)
        return torch.cat(lc), torch.cat(lr)


def preference_metrics(clean_c, clean_r, hw_c, hw_r) -> dict:
    clean_ok, hw_ok = clean_c > clean_r, hw_c > hw_r
    flips = clean_ok & ~hw_ok
    n_ok = int(clean_ok.sum())
    return {
        "clean_acc": 100.0 * clean_ok.float().mean().item(),
        "hw_acc": 100.0 * hw_ok.float().mean().item(),
        "flip_rate": 100.0 * int(flips.sum()) / n_ok if n_ok else 0.0,
        "n_pairs": len(clean_ok), "n_clean_correct": n_ok, "n_flips": int(flips.sum()),
        "mean_margin_clean": (clean_c - clean_r).mean().item(),
        "mean_margin_hw": (hw_c - hw_r).mean().item(),
    }


def _summary(values: list) -> dict:
    a = np.asarray(values, dtype=np.float64)
    std = a.std(ddof=1) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "std": float(std), "ci95": float(1.96 * std / math.sqrt(len(a)))}


def _programming_error(alpha: float, fault_model: str) -> Optional[dict]:
    if alpha <= 0:
        return None
    a_ind = alpha if fault_model in ("state_independent", "both") else 0.0
    a_prop = alpha if fault_model in ("state_proportional", "both") else 0.0
    return {"alpha_ind": a_ind, "alpha_prop": a_prop, "fault_model": fault_model}


def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)


def evaluate_flip_rate(
    sim,
    scorer: PreferenceScorer,
    pairs: Sequence[PreferencePair],
    alphas: Sequence[float] = (0.0, 0.002, 0.004, 0.006, 0.008),
    n_chips: int = 10,
    fault_model: str = "state_independent",
    stuck_at_params: Optional[dict] = None,
    relaxation_params: Optional[dict] = None,
    drift_params: Optional[dict] = None,
    subset_size: Optional[int] = None,
    controller=None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Preference accuracy and flip rate under RRAM programming error, swept over alpha
    (sigma_G = alpha * G_max, state-independent by default [HARD-DPO, Eq. 7]).

    The clean reference is the digital model as deployed (sim.write_ideal_weights: merged
    experts and deployment clipping, no quantization or noise). Every chip is a fresh
    sim.program(...) with the given non-idealities; with stuck_at_params and no fault_seed each
    chip gets its own fault map. subset_size reproduces the released scripts, which draw a new
    random subset of the pool for every chip. A SparseMoF `controller` runs one telemetry-guided
    selection per chip before scoring; flips are then counted against the digital model with
    only the always-on experts.
    """
    encoded = scorer.encode(pairs)
    if controller is not None:
        controller.reset()  # flips are counted against the always-on deployment
        sim.reset_analog_model()
    sim.write_ideal_weights()
    clean_c, clean_r = scorer.score(sim.analog_model, encoded)
    results = {"alphas": list(alphas), "n_chips": n_chips, "per_alpha": {}}
    for alpha in alphas:
        runs = []
        for chip in range(n_chips):
            # Chip k uses the same seed at every alpha and for every model (common random numbers).
            _seed_all(seed + chip)
            idx = (np.random.choice(len(encoded), subset_size, replace=False)
                   if subset_size is not None else np.arange(len(encoded)))
            pe = _programming_error(alpha, fault_model)
            if controller is not None:
                controller.reset()
            sim.program(stuck_at_params, pe, relaxation_params, drift_params)
            info = {}
            if controller is not None:
                controller.reprogram_params = {"programming_error_params": pe,
                                               "relaxation_params": relaxation_params,
                                               "drift_params": drift_params}
                info = controller.step()
            sub = [encoded[i] for i in idx]
            hw_c, hw_r = scorer.score(sim.analog_model, sub)
            m = preference_metrics(clean_c[idx], clean_r[idx], hw_c, hw_r)
            if info:
                m["active_experts"] = info["active"]
            runs.append(m)
            if verbose:
                extra = f" experts={info['active']}" if info else ""
                print(f"alpha={alpha:g} chip {chip + 1}/{n_chips}: flip={m['flip_rate']:.2f}% "
                      f"clean={m['clean_acc']:.1f}% hw={m['hw_acc']:.1f}%{extra}")
        results["per_alpha"][str(alpha)] = {
            "flip_rate": _summary([r["flip_rate"] for r in runs]),
            "clean_acc": _summary([r["clean_acc"] for r in runs]),
            "hw_acc": _summary([r["hw_acc"] for r in runs]),
            "runs": runs,
        }
    return results


# ---------------------------------------------------------------------------- HARD-DPO

@dataclass
class HardDPOConfig:
    """Defaults of the released HARD-DPO training script and paper [HARD-DPO, Sec. V-A]."""
    sigma: float = 0.008            # training noise, matched to alpha_prog = 0.008
    clip_k: Optional[float] = 3.0   # clip the noisy weights to +-3 std [HARD-DPO, Eq. 26]
    noise_target: str = "merged"    # "merged" (paper text) or "base" (released script)
    beta: float = 0.1
    lambda_kd: float = 0.5
    kd_tau: float = 1.0
    rank: int = 8
    lora_alpha: float = 16
    lr: float = 5e-5
    steps: int = 1000
    batch_size: int = 4
    grad_clip: float = 1.0
    expert: str = "dpo"
    logit_chunk: int = 256
    log_every: int = 10
    seed: int = 42


def _masked_avg_and_kd(model, h_s, h_t, ids, resp, n_chosen, chunk, kd_tau):
    """
    Average log-likelihood of every row of the student batch, and the token-averaged
    KL(teacher || student) over the chosen rows' response tokens. Each chunk of vocabulary-size
    logits is recomputed in the backward pass (activation checkpointing), so only one chunk of
    logits is alive at a time.
    """
    _, head = _body_and_head(model)
    labels, mask = ids[:, 1:], resp[:, 1:]

    def chunk_terms(h, h_teacher, lab, m):
        logp = F.log_softmax(_logits(head, h, model), dim=-1)
        tok = (logp.gather(2, lab.unsqueeze(2)).squeeze(2) * m).sum(1)
        if h_teacher is None:
            return tok, torch.zeros((), device=h.device)
        with torch.no_grad():
            p_t = F.softmax(_logits(head, h_teacher, model) / kd_tau, dim=-1)
        log_s = logp[:n_chosen] if kd_tau == 1 else F.log_softmax(
            _logits(head, h[:n_chosen], model) / kd_tau, dim=-1)
        kl = F.kl_div(log_s, p_t, reduction="none").sum(-1) * kd_tau ** 2
        return tok, (kl * m[:n_chosen]).sum()

    total = torch.zeros(ids.shape[0], device=ids.device)
    kl_sum = torch.zeros((), device=ids.device)
    for i in range(0, h_s.shape[1], chunk):
        m = mask[:, i:i + chunk]
        if not bool(m.any()):
            continue
        args = (h_s[:, i:i + chunk], None if h_t is None else h_t[:, i:i + chunk], labels[:, i:i + chunk], m)
        if h_s.requires_grad:
            tok, kl = torch.utils.checkpoint.checkpoint(chunk_terms, *args, use_reentrant=False)
        else:
            tok, kl = chunk_terms(*args)
        total = total + tok
        kl_sum = kl_sum + kl
    avg = total / mask.sum(1).clamp(min=1)
    kd = kl_sum / mask[:n_chosen].sum().clamp(min=1)
    return avg, kd


@contextmanager
def _clean_layers(model):
    """Removes training noise and stuck-at overlays for the clean teacher / reference passes."""
    mods = lora_modules(model)
    saved = [(m.noise, m.fault) for _, m in mods]
    try:
        for _, m in mods:
            m.noise = m.fault = None
        yield
    finally:
        for (_, m), (n, f) in zip(mods, saved):
            m.noise, m.fault = n, f


def hard_dpo_loss(model, scorer: PreferenceScorer, batch: list, cfg: HardDPOConfig,
                  ref_model=None, device=None):
    """
    HARD-DPO loss on a batch of encoded pairs [HARD-DPO, Eqs. 28-32]. The student pass uses the
    noise preset and stuck-at overlays installed on the model (lora.noise_context /
    fault_context); keep them installed through loss.backward(), which recomputes checkpointed
    layers. Chosen and rejected responses share one noisy forward pass (one chip realization per
    step). The clean teacher and the reference (`ref_model`, or the same model with all experts
    off) run without them. Returns (loss, dpo, kd).
    """
    device = device or next(model.parameters()).device
    seqs = [(e[0], e[1]) for e in batch] + [(e[2], e[3]) for e in batch]
    ids, attn, resp = scorer._batch(seqs, device)
    B = len(batch)
    body, _ = _body_and_head(model)

    with torch.no_grad(), _clean_layers(model):
        if ref_model is None:
            with experts_disabled(model):
                ref_avg = scorer.sequence_logps(model, ids, attn, resp, cfg.logit_chunk)
        else:
            ref_avg = scorer.sequence_logps(ref_model, ids, attn, resp, cfg.logit_chunk)
        h_t = None
        if cfg.lambda_kd > 0:
            h_t = body(input_ids=ids[:B], attention_mask=attn[:B], use_cache=False).last_hidden_state[:, :-1]

    h_s = body(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state[:, :-1]
    avg, kd = _masked_avg_and_kd(model, h_s, h_t, ids, resp, B, cfg.logit_chunk, cfg.kd_tau)
    logits_diff = (avg[:B] - avg[B:]) - (ref_avg[:B] - ref_avg[B:])
    dpo = -F.logsigmoid(cfg.beta * logits_diff).mean()
    loss = dpo + cfg.lambda_kd * kd if cfg.lambda_kd > 0 else dpo
    return loss, dpo.detach(), kd.detach()


def train_hard_dpo(model, scorer: PreferenceScorer, pairs: Sequence[PreferencePair],
                   cfg: Optional[HardDPOConfig] = None, ref_model=None, target_modules=DEFAULT_TARGET_MODULES,
                   gradient_checkpointing: bool = False, noise=None) -> list:
    """
    Trains the HARD-DPO expert cfg.expert in place. The base model stays frozen and, unless
    ref_model is given, also serves as the DPO reference (experts off), so no second copy of
    the model is needed. noise=None uses the HARD-DPO log-normal preset; pass another preset
    (e.g. train_noise.SimulatorNoise) for the simulator-exact variant. Returns the loss history.
    """
    cfg = cfg or HardDPOConfig()
    if not lora_modules(model):
        add_lora(model, target_modules)
    if cfg.expert not in expert_names(model):
        add_expert(model, cfg.expert, cfg.rank, cfg.lora_alpha)
    set_active(model, [cfg.expert])
    params = set_trainable(model, [cfg.expert])
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    noise = noise or LogNormalNoise(cfg.sigma, cfg.clip_k, cfg.noise_target)
    if gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    else:
        model.eval()

    encoded = scorer.encode(pairs)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    history = []
    for step in range(cfg.steps):
        batch = [encoded[i] for i in rng.choice(len(encoded), size=cfg.batch_size, replace=False)]
        opt.zero_grad(set_to_none=True)
        with noise_context(model, noise):
            loss, dpo, kd = hard_dpo_loss(model, scorer, batch, cfg, ref_model)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        history.append({"step": step + 1, "loss": loss.item(), "dpo": dpo.item(), "kd": kd.item()})
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(f"[HARD-DPO] step {step + 1}/{cfg.steps} loss={loss.item():.4f} "
                  f"dpo={dpo.item():.4f} kd={kd.item():.4f}")
    if gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.eval()
    set_trainable(model, [])
    return history


def dpo_expert_loss(scorer: PreferenceScorer, cfg: Optional[HardDPOConfig] = None, ref_model=None):
    """
    loss_fn(model, batch) for mof.train_expert that trains an expert with the HARD-DPO objective
    on encoded pairs. The expert's own noise preset (installed by train_expert) is the noisy
    student; the clean teacher and the reference use the same model.
    """
    cfg = cfg or HardDPOConfig()

    def loss_fn(model, batch):
        # train_expert installed the expert's noise and stuck-at overlays: they make the noisy
        # student, while hard_dpo_loss runs the teacher and reference without them.
        return hard_dpo_loss(model, scorer, batch, cfg, ref_model)[0]

    return loss_fn


def sft_loss(scorer: PreferenceScorer):
    """loss_fn(model, batch) for mof.train_expert: negative log-likelihood of the preferred
    responses (supervised fine-tuning on y+)."""
    def loss_fn(model, batch):
        device = next(model.parameters()).device
        ids, attn, resp = scorer._batch([(e[0], e[1]) for e in batch], device)
        return -scorer.sequence_logps(model, ids, attn, resp).mean()
    return loss_fn


def batches(encoded: list, batch_size: int, seed: int = 0):
    """Infinite shuffled batches of encoded pairs, for mof.train_expert."""
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(len(encoded))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [encoded[j] for j in order[i:i + batch_size]]
