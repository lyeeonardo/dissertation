"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import math

from dataclasses import dataclass

import torch

from torch import nn

from torch.nn import functional as F

@dataclass(frozen=True)
class ModelConfig:
    observation_dim: int = 17
    action_dim: int = 6
    history: int = 8
    max_horizon: int = 16
    width: int = 512
    layers: int = 8
    heads: int = 8
    sampling_steps: int = 8
    condition_input: bool = True
    first_action_weight: float = 10.0

class CausalBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.projection = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 4 * width))

    def forward(self, x, condition):
        shift1, scale1, shift2, scale2 = self.modulation(condition).chunk(4, -1)
        first = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(first)
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        second = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.feedforward(second)

class SharedDiffusion(nn.Module):
    def __init__(self, config: ModelConfig, normalization: dict):
        super().__init__()
        self.config = config
        if config.width % config.heads or config.layers < 1 or config.sampling_steps < 1:
            raise ValueError("Invalid model architecture")
        d = config.width
        self.token_width = config.observation_dim + 1
        cdim = (
            config.history * config.observation_dim
            + (config.history - 1) * (config.action_dim + 1)
            + config.history
        )
        self.input = nn.Linear(3 * self.token_width, d)
        self.context = nn.Sequential(nn.Linear(cdim, d), nn.SiLU(), nn.Linear(d, d))
        self.condition = nn.Linear(3, d)
        self.quality = nn.Embedding(2, d)
        self.role = nn.Embedding(3, d)
        self.horizon = nn.Embedding(config.max_horizon + 1, d)
        self.time = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.randn(1, 2 * config.max_horizon, d) * 0.01)
        self.blocks = nn.ModuleList(CausalBlock(d, config.heads) for _ in range(config.layers))
        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, self.token_width)
        self.termination = nn.Linear(d, 1)
        for name in (
            "observation_mean",
            "observation_std",
            "delta_mean",
            "delta_std",
            "reward_mean",
            "reward_std",
        ):
            value = torch.as_tensor(normalization[name], dtype=torch.float32)
            if not torch.isfinite(value).all() or (name.endswith("std") and (value <= 0).any()):
                raise ValueError(f"Invalid normalization: {name}")
            self.register_buffer(name, value)

    def history_context(self, history):
        allowed = {"observations", "actions", "rewards", "mask"}
        if set(history) != allowed:
            raise ValueError(f"Public history fields must be exactly {allowed}")
        mask = history["mask"].float()
        transition_mask = mask[:, :-1] * mask[:, 1:]
        observations = (
            (history["observations"] - self.observation_mean) / self.observation_std
        ) * mask[..., None]
        actions = history["actions"] * transition_mask[..., None]
        rewards = ((history["rewards"] - self.reward_mean) / self.reward_std) * transition_mask
        return torch.cat((observations.flatten(1), actions.flatten(1), rewards, mask), -1)

    def encode(self, batch):
        """Training-only targets never enter context or globally summarized known values."""
        actions = batch["actions"]
        n, h, _ = actions.shape
        if h > self.config.max_horizon:
            raise ValueError("Unsupported horizon")
        current = batch["history"]["observations"][:, -1]
        x = actions.new_zeros(n, 2 * h, self.token_width)
        x[:, ::2, : self.config.action_dim] = actions
        x[:, 1::2, : self.config.observation_dim] = (
            batch["future_observations"] - current[:, None] - self.delta_mean
        ) / self.delta_std
        x[:, 1::2, -1] = (batch["rewards"] - self.reward_mean) / self.reward_std
        active_steps = torch.arange(h, device=x.device)[None] < batch["horizon"][:, None]
        semantic = torch.zeros_like(x, dtype=torch.bool)
        semantic[:, ::2, : self.config.action_dim] = active_steps[..., None]
        semantic[:, 1::2] = active_steps[..., None]
        known = torch.zeros_like(semantic)
        known[:, ::2, : self.config.action_dim] = (active_steps & (batch["role"] != 0)[:, None])[
            ..., None
        ]
        # True future validity affects loss only, never context or attention.
        valid = active_steps & batch["valid"].bool()
        return {
            "x": x,
            "semantic": semantic,
            "known": known,
            "valid": valid,
            "context": self.history_context(batch["history"]),
            "condition": batch["condition"],
            "horizon": batch["horizon"],
            "role": batch["role"],
            "quality": torch.where(batch["role"] == 0, batch["quality"], 0),
        }

    def forward(self, noisy, encoded, time):
        known = torch.where(encoded["known"], encoded["x"], 0)
        features = torch.stack(
            (time, torch.sin(math.pi * time), torch.cos(math.pi * time), time.square()), -1
        )
        labels = encoded["condition"]
        bits = torch.stack(((labels >> 2) & 1, (labels >> 1) & 1, labels & 1), -1).float()
        if not self.config.condition_input:
            bits = torch.zeros_like(bits)
        context = (
            self.context(encoded["context"])
            + self.condition(bits)
            + self.role(encoded["role"])
            + self.quality(encoded["quality"])
            + self.horizon(encoded["horizon"])
            + self.time(features)
        )
        x = (
            self.input(torch.cat((noisy, known, encoded["known"].float()), -1))
            + self.position[:, : noisy.shape[1]]
            + context[:, None]
        )
        for block in self.blocks:
            x = block(x, context)
        x = self.norm(x)
        return self.output(x), self.termination(x[:, 1::2]).squeeze(-1)

    def loss(self, batch, generator=None):
        e = self.encode(batch)
        x = e["x"]
        t = torch.rand(len(x), device=x.device, generator=generator) * 0.998 + 0.001
        noise = torch.randn(x.shape, device=x.device, generator=generator)
        alpha, sigma = torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
        noisy = alpha[:, None, None] * x + sigma[:, None, None] * noise
        noisy = torch.where(e["known"], x, noisy) * e["semantic"]
        predicted, termination = self(noisy, e, t)
        squared = (predicted.float() - x.float()).square()
        valid = e["valid"].float()
        action_mask = valid * batch["proposal_eligible"].float() * (batch["role"] == 0)[:, None]
        action_weights = action_mask.clone()
        action_weights[:, 0] *= self.config.first_action_weight
        action_error = squared[:, ::2, : self.config.action_dim].mean(-1)
        action_loss = (action_error * action_weights).sum(-1) / action_weights.sum(-1).clamp_min(1)
        state_error = squared[:, 1::2, : self.config.observation_dim].mean(-1)
        state_loss = (state_error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        reward_loss = (squared[:, 1::2, -1] * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        terminal_error = F.binary_cross_entropy_with_logits(
            termination.float(), batch["terminated"].float(), reduction="none"
        )
        terminal_loss = (terminal_error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        per_example = action_loss + state_loss + reward_loss + 0.1 * terminal_loss
        # The sampler provides exactly 50/25/25 roles; mean preserves that weighting
        # across arbitrary microbatch boundaries when scaled by microbatch size.
        metrics = {
            "action": action_loss.detach().mean(),
            "state": state_loss.detach().mean(),
            "reward": reward_loss.detach().mean(),
            "termination": terminal_loss.detach().mean(),
        }
        return per_example.mean(), metrics

    @torch.no_grad()
    def sample(
        self,
        history,
        condition,
        horizon=8,
        fixed_actions=None,
        noise=None,
        generator=None,
        quality=1,
        temperature=1.0,
        steps=None,
    ):
        """Low-level batched query. Imposed-action predictions always use NULL quality."""
        self.eval()
        device = next(self.parameters()).device
        h = horizon
        if not 1 <= h <= self.config.max_horizon:
            raise ValueError("Unsupported sampling horizon")
        n = len(condition)
        fixed = fixed_actions is not None
        if fixed:
            if fixed_actions.shape != (n, h, self.config.action_dim):
                raise ValueError("Invalid imposed-action shape")
            if not torch.isfinite(fixed_actions).all() or (fixed_actions.abs() > 1.000001).any():
                raise ValueError("Imposed actions must be finite and physically bounded")
        actions = (
            fixed_actions.float()
            if fixed
            else torch.zeros(n, h, self.config.action_dim, device=device)
        )
        zeros = torch.zeros(n, h, device=device)
        batch = {
            "history": history,
            "condition": condition.long(),
            "actions": actions,
            "future_observations": torch.zeros(n, h, self.config.observation_dim, device=device),
            "rewards": zeros,
            "valid": torch.ones_like(zeros, dtype=torch.bool),
            "horizon": torch.full((n,), h, dtype=torch.long, device=device),
            "role": torch.full(
                (n,), (1 if h == 1 else 2) if fixed else 0, dtype=torch.long, device=device
            ),
            "quality": torch.full((n,), 0 if fixed else quality, dtype=torch.long, device=device),
        }
        e = self.encode(batch)
        if noise is None:
            noise = torch.randn(e["x"].shape, device=device, generator=generator)
        if noise.shape != e["x"].shape:
            raise ValueError("Sampling noise must match the complete query shape")
        x = torch.where(e["known"], e["x"], noise * temperature) * e["semantic"]
        grid = torch.linspace(0.999, 0, (steps or self.config.sampling_steps) + 1, device=device)
        for time, next_time in zip(grid[:-1], grid[1:], strict=True):
            prediction, _ = self(x, e, time.expand(n))
            prediction = prediction.float()
            alpha, sigma = torch.cos(time * math.pi / 2), torch.sin(time * math.pi / 2)
            epsilon = (x - alpha * prediction) / sigma.clamp_min(1e-5)
            x = (
                torch.cos(next_time * math.pi / 2) * prediction
                + torch.sin(next_time * math.pi / 2) * epsilon
            )
            x = torch.where(e["known"], e["x"], x) * e["semantic"]
        _, terminal = self(x, e, torch.zeros(n, device=device))
        returned_actions = x[:, ::2, : self.config.action_dim].clamp(-1, 1)
        if fixed:
            returned_actions = fixed_actions.clone()
        observations = (
            x[:, 1::2, : self.config.observation_dim] * self.delta_std
            + self.delta_mean
            + history["observations"][:, -1, None]
        )
        rewards = x[:, 1::2, -1] * self.reward_std + self.reward_mean
        result = {
            "actions": returned_actions,
            "termination_probability": terminal.float().sigmoid(),
        }
        # Joint outcomes are explicitly named; the planner must physically re-predict
        # the bounded proposal actions across its entire common belief.
        prefix = "" if fixed else "joint_"
        result[prefix + "observations"] = observations
        result[prefix + "rewards"] = rewards
        return result

def identity_normalization(observation_dim=17):
    return {
        "observation_mean": [0.0] * observation_dim,
        "observation_std": [1.0] * observation_dim,
        "delta_mean": [0.0] * observation_dim,
        "delta_std": [1.0] * observation_dim,
        "reward_mean": 0.0,
        "reward_std": 1.0,
    }
