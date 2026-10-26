"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import math

from pathlib import Path

import numpy as np

import torch

from .model import ModelConfig, SharedDiffusion, identity_normalization

from .windows import WindowSampler, slice_batch, to_device

def learning_rate(update):
    if update < 10_000:
        return 1e-4 * (update + 1) / 10_000
    fraction = min(1.0, (update - 10_000) / 290_000)
    return 1e-5 + 0.5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * fraction))

def update_ema(ema, model, rate=0.999):
    with torch.no_grad():
        averaged, current = list(ema.parameters()), list(model.parameters())
        torch._foreach_mul_(averaged, rate)
        torch._foreach_add_(averaged, current, alpha=1 - rate)

def train_step(model, optimizer, batch, microbatch, ema=None):
    n = len(batch["role"])
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    model.train()
    for start in range(0, n, microbatch):
        end = min(n, start + microbatch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, values = model.loss(slice_batch(batch, start, end))
        (loss * ((end - start) / n)).backward()
        for key, value in {"loss": loss.detach(), **values}.items():
            metrics[key] = metrics.get(key, 0) + value * ((end - start) / n)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if ema is not None:
        update_ema(ema, model)
    metrics["gradient_norm"] = norm.detach()
    return metrics

@torch.no_grad()
def development_loss(model, sampler, batch_size=256, batches=8):
    model.eval()
    sampler.rng = np.random.default_rng(130_003)
    generator = torch.Generator(device="cuda").manual_seed(130_003)
    metrics = {}
    for _ in range(batches):
        batch = to_device(sampler.sample(batch_size, augment=False))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, values = model.loss(batch, generator)
        for key, value in {"loss": loss, **values}.items():
            metrics[key] = metrics.get(key, 0) + float(value) / batches
    return metrics

def load_model(checkpoint: Path, device="cuda", ema=True):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SharedDiffusion(ModelConfig(**state["model_config"]), state["normalization"]).to(device)
    model.load_state_dict(state["ema"] if ema else state["model"])
    model.eval()
    return model, state
