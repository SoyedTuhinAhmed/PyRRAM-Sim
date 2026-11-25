"""Presto encoder + mean/std patch aggregation + linear head for EuroSAT (Presto's
``PrestoFinetuningWithAggregates``), and the training stages of the case study."""
import copy
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from torch import nn

import lora as L
from train_noise import AbsMaxGaussianNoise

from .eurosat import PrestoInputBuilder, images_per_batch, iterate_batches
from .single_file_presto import Presto

LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")   # attn.qkv, attn.proj, mlp.fc1, mlp.fc2


class PrestoEuroSatClassifier(nn.Module):
    def __init__(self, encoder, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_size * 2, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, dynamic_world, latlons, mask, month):
        p = x.shape[1]
        enc = self.encoder(x=rearrange(x, "b p t d -> (b p) t d"),
                           dynamic_world=rearrange(dynamic_world, "b p t -> (b p) t"),
                           latlons=rearrange(latlons, "b p d -> (b p) d"),
                           mask=repeat(mask, "b t d -> (rep b) t d", rep=p),
                           month=rearrange(month, "b p -> (b p)"), eval_task=True)
        per = rearrange(enc, "(img p) d -> img p d", p=p)
        return self.head(torch.cat([per.mean(1), per.std(1, correction=int(p > 1))], dim=-1))


def build_model(presto_weights: Optional[Path], num_classes: int = 10) -> PrestoEuroSatClassifier:
    presto = Presto.construct()
    if presto_weights is not None:
        presto.load_state_dict(torch.load(presto_weights, map_location="cpu", weights_only=True), strict=True)
    enc = presto.encoder
    enc.pos_embed.requires_grad_(False)
    enc.month_embed.requires_grad_(False)
    return PrestoEuroSatClassifier(enc, num_classes).to("cpu")


def load_classifier(checkpoint: Path, presto_weights: Optional[Path] = None) -> PrestoEuroSatClassifier:
    """A fine-tuned (E0) or LoRA-merged (E2) checkpoint, e.g. the case study's best.pt / merged.pt."""
    model = build_model(None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state.get("state_dict", state))
    model.load_state_dict(state, strict=True)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, builder: PrestoInputBuilder, rows: np.ndarray, num_classes: int = 10,
             batch_images: Optional[int] = None) -> dict:
    """Accuracy and macro-F1. batch_images only changes throughput (default: Presto's 39)."""
    model.eval()
    preds, labels = [], []
    for batch in iterate_batches(builder, rows, batch_images or images_per_batch(builder.n_patches)):
        preds.append(model(*builder.model_args(batch)).argmax(-1).cpu())
        labels.append(batch["labels"].cpu())
    y_pred, y_true = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    f1 = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        d = 2 * tp + np.sum((y_pred == c) & (y_true != c)) + np.sum((y_pred != c) & (y_true == c))
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return {"accuracy": float(np.mean(y_pred == y_true)), "macro_f1": float(np.mean(f1)), "n_test": int(len(y_true))}


def train_model(model: nn.Module, builder: PrestoInputBuilder, train_rows, val_rows, lr=3e-4, weight_decay=0.05,
                max_epochs=100, patience=3, generator=None, noise=None, log: Callable = print) -> dict:
    """Presto's fine-tuning protocol: AdamW, early stopping on the validation loss."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    per = images_per_batch(builder.n_patches)
    best, best_state, stale, history = None, None, 0, []
    for epoch in range(max_epochs):
        started = time.time()
        model.train()
        for batch in iterate_batches(builder, train_rows, per, shuffle=True, generator=generator):
            opt.zero_grad(set_to_none=True)
            with L.noise_context(model, noise):
                loss = loss_fn(model(*builder.model_args(batch)), batch["labels"])
                loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tot, n, correct = 0.0, 0, 0
            for batch in iterate_batches(builder, val_rows, per):
                logits = model(*builder.model_args(batch))
                tot += float(loss_fn(logits, batch["labels"])) * len(batch["labels"])
                correct += int((logits.argmax(-1) == batch["labels"]).sum())
                n += len(batch["labels"])
        val_loss = tot / n
        history.append({"epoch": epoch, "val_loss": val_loss, "val_acc": correct / n})
        log(f"  epoch {epoch:3d} val_loss={val_loss:.4f} val_acc={correct / n:.4f} ({time.time() - started:.0f}s)")
        if best is None or val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return {"history": history, "best_val_loss": best}


def finetune(model, builder, train_rows, val_rows, seed: int, log: Callable = print) -> dict:
    """E0: FP32 fine-tuning of encoder and head."""
    return train_model(model, builder, train_rows, val_rows, generator=torch.Generator().manual_seed(seed), log=log)


def noise_aware_lora(model, builder, train_rows, val_rows, seed: int, rank: int = 8, alpha: float = 16,
                     noise_std: float = 0.1, max_epochs: int = 50, log: Callable = print) -> nn.Module:
    """
    E2: LoRA (r = 8, alpha = 16) on attn.qkv / attn.proj / mlp.fc1 / mlp.fc2 plus a trainable
    head, Gaussian noise of 0.1 max|W| on the effective weights during training, then merged
    into plain Linear layers for analog mapping [COINS'26, Sec. IV-D.1].
    """
    L.add_lora(model, LORA_TARGETS, exclude=("head",))
    L.add_expert(model, "lora", rank, alpha)
    L.set_active(model, ["lora"])
    L.set_trainable(model, ["lora"])
    for p in model.head.parameters():
        p.requires_grad_(True)
    train_model(model, builder, train_rows, val_rows, max_epochs=max_epochs, generator=torch.Generator().manual_seed(seed),
                noise=AbsMaxGaussianNoise(noise_std), log=log)
    L.merge_experts(model)
    for p in model.parameters():
        p.requires_grad_(True)
    model.encoder.pos_embed.requires_grad_(False)
    model.encoder.month_embed.requires_grad_(False)
    return model
