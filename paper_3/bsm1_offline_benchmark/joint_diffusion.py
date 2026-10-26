"""One frozen joint action/state/reward diffusion for oracle and belief queries.

The three query masks are trained explicitly. The public-only evidence energy
is a generalized denoising score, not a normalized observation likelihood.
No historical action-only weights or external value critic enter this model.
"""

from __future__ import annotations

import argparse

from dataclasses import asdict, dataclass

import json

import math

from pathlib import Path

import random

import time

import traceback

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from .diffusion import atomic_json

from .koopman import atomic_checkpoint

from .stock_shared_data import PUBLIC_INDICES, load_arrays

MODEL_CLASS = "BSM1JointDiffusion"

STATE_DIM, PUBLIC_DIM, ACTION_DIM, JOINT_DIM = 159, 55, 2, 162

ACTION_SLICE = slice(159, 161)

REWARD_INDEX = 161

HIDDEN_INDICES = np.setdiff1d(np.arange(STATE_DIM), PUBLIC_INDICES)

QUERY_NAMES = ("proposal", "forecast", "evidence")

HISTORY_LENGTHS = (1, 8, 24)

@dataclass
class JointConfig:
    horizon: int = 30
    history: int = 24
    width: int = 128
    normalization_floor: float = .01
    quality_dropout: float = .2
    maximum_time: float = .999
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    seed: int = 17
    prediction_type: str = "epsilon"

    def __post_init__(self):
        if self.horizon != 30 or self.history != 24 or self.width < 8:
            raise ValueError("This recipe fixes H30/max-history24 and requires width >=8")
        if not 0 < self.maximum_time < 1 or not 0 <= self.quality_dropout < 1:
            raise ValueError("Invalid cosine time or quality dropout")
        if self.prediction_type not in ("epsilon", "v"):
            raise ValueError("Choose epsilon or velocity prediction")
        if self.prediction_type == "v" and self.width < JOINT_DIM:
            raise ValueError("The repaired joint model must not bottleneck its independent noise channels")

def cosine_schedule(t):
    """Continuous VP schedule: alpha(0)=1, sigma(0)=0, alpha²+sigma²=1."""
    return torch.cos(t * (math.pi / 2)), torch.sin(t * (math.pi / 2))

class CausalBlock(nn.Module):
    """Per-lead normalization avoids future leakage through temporal GroupNorm."""
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.condition = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width)
        self.padding = 2 * dilation

    def forward(self, x, context):
        scale, shift = self.condition(context).chunk(2, -1)
        h = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        h = self.conv(F.pad(F.silu(h).transpose(1, 2), (self.padding, 0))).transpose(1, 2)
        return x + self.output(F.silu(h))

class JointDiffusion(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or JointConfig()
        w = self.config.width
        self.register_buffer("state_mean", torch.zeros(STATE_DIM))
        self.register_buffer("state_scale", torch.ones(STATE_DIM))
        self.register_buffer("cost_scale", torch.ones(()))
        self.register_buffer("public_indices", torch.as_tensor(PUBLIC_INDICES.copy()))
        self.register_buffer("hidden_indices", torch.as_tensor(HIDDEN_INDICES.copy()))
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 16)))
        # Public obs, previous executed action/reward, validity flags, relative time.
        self.history_input = nn.Sequential(nn.Linear(PUBLIC_DIM + 2 + 1 + 2 + 1, w), nn.SiLU(), nn.Linear(w, w))
        self.history_conv = nn.Conv1d(w, w, 3, padding=1)
        self.history_output = nn.Sequential(nn.Linear(2 * w, w), nn.SiLU(), nn.Linear(w, w))
        self.initializer = nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 2 * len(HIDDEN_INDICES)))
        self.condition = nn.Sequential(nn.Linear(STATE_DIM + w + 32 + 2 + 3, w), nn.SiLU(), nn.Linear(w, w))
        self.input = nn.Linear(4 * JOINT_DIM, w)
        self.blocks = nn.ModuleList(CausalBlock(w, d) for d in (1, 2, 4, 8))
        self.output = nn.Sequential(nn.LayerNorm(w), nn.SiLU(), nn.Linear(w, JOINT_DIM))

    def set_normalization(self, training):
        x = np.asarray(training["privileged"], dtype=np.float64)
        mean, scale = x.mean(0), x.std(0).clip(self.config.normalization_floor)
        cost = float(np.mean(training["costs"], dtype=np.float64))
        if not np.isfinite(x).all() or not np.isfinite(cost) or cost <= 0:
            raise ValueError("Finite TRAIN states and a positive TRAIN cost mean are required")
        self.state_mean.copy_(torch.as_tensor(mean, device=self.state_mean.device))
        self.state_scale.copy_(torch.as_tensor(scale, device=self.state_mean.device))
        self.cost_scale.fill_(cost)

    def history_context(self, history):
        obs, actions, rewards = (history[k] for k in ("observations", "previous_actions", "previous_rewards"))
        valid, reward_valid = history["valid"].bool(), history["reward_valid"].bool()
        if obs.ndim != 3 or obs.shape[-1] != PUBLIC_DIM or not 1 <= obs.shape[1] <= self.config.history:
            raise ValueError("History must contain1..24 public55 packets")
        if actions.shape != obs.shape[:2] + (2,) or rewards.shape != obs.shape[:2] + (1,):
            raise ValueError("Prior actions/rewards must align with public history")
        if valid.shape != obs.shape[:2] or reward_valid.shape != valid.shape:
            raise ValueError("History and reward validity masks must align")
        if obs.shape[1] < self.config.history:
            # Right-align every API representation on the trained24-slot clock.
            # In particular L1's current observation has position0, not -1.
            padding = self.config.history - obs.shape[1]
            obs = F.pad(obs, (0, 0, padding, 0))
            actions = F.pad(actions, (0, 0, padding, 0))
            rewards = F.pad(rewards, (0, 0, padding, 0))
            valid = F.pad(valid, (padding, 0)); reward_valid = F.pad(reward_valid, (padding, 0))
        # Calling code validates that the final/root slot is always present.
        normalized = (obs - self.state_mean[self.public_indices]) / self.state_scale[self.public_indices]
        position = torch.linspace(-1, 0, obs.shape[1], device=obs.device, dtype=obs.dtype)[None, :, None]
        values = torch.cat((normalized, 2 * actions - 1, rewards * reward_valid[..., None],
                            valid[..., None], reward_valid[..., None], position.expand(len(obs), -1, -1)), -1)
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        tokens = self.history_input(values) * valid[..., None]
        tokens = (tokens + F.silu(self.history_conv(tokens.transpose(1, 2)).transpose(1, 2))) * valid[..., None]
        pooled = tokens.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.history_output(torch.cat((tokens[:, -1], pooled), -1))

    def initialize(self, history, *, samples=0, generator=None):
        """Predict only104 missing states; all55 current measured coordinates stay exact."""
        context = self.history_context(history)
        mu, log_scale = self.initializer(context).chunk(2, -1)
        log_scale = log_scale.clamp(-5, 2)
        mean = self.state_mean.expand(len(mu), -1).clone()
        scale = torch.zeros_like(mean)
        mean[:, self.hidden_indices] = mu * self.state_scale[self.hidden_indices] + self.state_mean[self.hidden_indices]
        scale[:, self.hidden_indices] = log_scale.exp() * self.state_scale[self.hidden_indices]
        mean[:, self.public_indices] = history["observations"][:, -1]
        result = dict(mean=mean, scale=scale, hidden_mean_normalized=mu, hidden_log_scale=log_scale)
        if samples:
            noise = torch.randn((len(mu), int(samples), len(HIDDEN_INDICES)), device=mu.device, generator=generator)
            states = mean[:, None].expand(-1, int(samples), -1).clone()
            states[:, :, self.hidden_indices] = (mean[:, None, self.hidden_indices] +
                scale[:, None, self.hidden_indices] * noise).clamp_min(0)
            result["samples"] = states
        return result

    def pack(self, next_private, actions, rewards):
        return torch.cat(((next_private - self.state_mean) / self.state_scale,
                          2 * actions - 1, rewards[..., None]), -1)

    def unpack(self, clean, horizon=None):
        clean = clean[:, :horizon] if horizon is not None else clean
        states = clean[..., :STATE_DIM] * self.state_scale + self.state_mean
        return dict(next_private=states, public=states[..., self.public_indices],
                    actions=(clean[..., ACTION_SLICE] + 1) / 2,
                    rewards=clean[..., REWARD_INDEX], clean=clean)

    def query_masks(self, modes, lengths=None):
        b, h = len(modes), self.config.horizon
        modes = modes.long()
        lengths = torch.full_like(modes, h) if lengths is None else lengths.long()
        lengths = torch.where(modes == 2, torch.ones_like(lengths), lengths)
        active = torch.arange(h, device=modes.device)[None] < lengths[:, None]
        target = torch.zeros(b, h, JOINT_DIM, dtype=torch.bool, device=modes.device)
        known = torch.zeros_like(target)
        target[..., :STATE_DIM] = ((modes != 2)[:, None] & active)[..., None]
        target[..., ACTION_SLICE] = ((modes == 0)[:, None] & active)[..., None]
        target[..., REWARD_INDEX] = active
        target[:, 0, self.public_indices] = True
        known[..., ACTION_SLICE] = ((modes != 0)[:, None] & active)[..., None]
        return target, known

    def forward(self, noisy, times, current159, history, quality, quality_present, modes,
                target_mask, known_mask, *, known_values=None, history_context=None):
        if noisy.shape[1:] != (self.config.horizon, JOINT_DIM) or current159.shape != (len(noisy), STATE_DIM):
            raise ValueError("Joint H30×162 and current159 tensors required")
        context = self.history_context(history) if history_context is None else history_context
        phase = times[:, None] * self.frequencies[None]
        q_present = quality_present.to(noisy.dtype) * (modes == 0)
        condition = torch.cat(((current159 - self.state_mean) / self.state_scale, context,
            phase.sin(), phase.cos(), (quality * q_present)[:, None], q_present[:, None],
            F.one_hot(modes.long(), 3)), -1)
        condition = self.condition(condition)
        visible = target_mask | known_mask
        known_values = torch.zeros_like(noisy) if known_values is None else known_values
        # Actual commands are clean conditioning as well as correctly noised
        # inpainted channels. Hidden outcome targets can never enter this field.
        x = self.input(torch.cat((torch.where(visible, noisy, torch.zeros_like(noisy)),
            torch.where(known_mask, known_values, torch.zeros_like(known_values)), target_mask, known_mask), -1))
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x)

    def _project_clean(self, clean):
        # Physical concentrations/flow are nonnegative; the actuator box is original.
        states = torch.maximum(clean[..., :STATE_DIM], -self.state_mean / self.state_scale)
        return torch.cat((states, clean[..., ACTION_SLICE].clamp(-1, 1), clean[..., REWARD_INDEX:]), -1)

    def denoise_components(self, noisy, times, prediction):
        """Recover x0/epsilon; velocity prediction has no division by tiny alpha."""
        times = torch.as_tensor(times, device=noisy.device, dtype=noisy.dtype)
        while times.ndim < noisy.ndim:
            times = times.unsqueeze(-1)
        a, s = cosine_schedule(times)
        if self.config.prediction_type == "v":
            return a * noisy - s * prediction, s * noisy + a * prediction
        return (noisy - s * prediction) / a, prediction

    @torch.no_grad()
    def _sample(self, current159, history, *, mode, actions=None, quality=None, passes=20,
                horizon=30, generator=None, initial_noise=None, known_noise=None,
                initial_clean=None, start_index=0):
        if not 1 <= horizon <= self.config.horizon or passes < 1 or not 0 <= start_index < passes:
            raise ValueError("Invalid horizon/pass/refinement index")
        b, device = len(current159), current159.device
        modes = torch.full((b,), mode, device=device, dtype=torch.long)
        target, known = self.query_masks(modes, torch.full_like(modes, horizon))
        clean_known = torch.zeros(b, self.config.horizon, JOINT_DIM, device=device)
        if mode == 1:
            if actions is None or actions.shape != (b, horizon, 2):
                raise ValueError("Forecast needs one actual/imposed action per requested lead")
            if not torch.isfinite(actions).all() or bool(((actions < 0) | (actions > 1)).any()):
                raise ValueError("Imposed commands must be finite and in the original[0,1] box")
            clean_known[:, :horizon, ACTION_SLICE] = 2 * actions - 1
        q = torch.zeros(b, device=device) if quality is None else torch.as_tensor(quality, device=device).expand(b)
        present = torch.full((b,), quality is not None and mode == 0, device=device)
        shape = clean_known.shape
        noise = torch.randn(shape, device=device, generator=generator) if initial_noise is None else initial_noise
        fixed_noise = torch.randn(shape, device=device, generator=generator) if known_noise is None else known_noise
        if noise.shape != shape or fixed_noise.shape != shape:
            raise ValueError("Sampling noise must have shape[B,30,162]")
        schedule = torch.linspace(self.config.maximum_time, 0., passes + 1, device=device)
        alpha, sigma = cosine_schedule(schedule[start_index])
        x = noise if initial_clean is None else alpha * initial_clean + sigma * noise
        context = self.history_context(history)
        for index in range(start_index, passes):
            a, s = cosine_schedule(schedule[index])
            na, ns = cosine_schedule(schedule[index + 1])
            x = torch.where(known, a * clean_known + s * fixed_noise, x)
            x = torch.where(target | known, x, torch.zeros_like(x))
            prediction = self(x, schedule[index].expand(b), current159, history, q, present,
                       modes, target, known, known_values=clean_known, history_context=context)
            clean, eps = self.denoise_components(x, schedule[index], prediction)
            clean = self._project_clean(clean)
            clean = torch.where(known, clean_known, clean)
            x = na * clean + ns * eps
        x = torch.where(known, clean_known, x)
        if not torch.isfinite(x).all():
            raise FloatingPointError("Nonfinite joint diffusion sample")
        return self.unpack(x, horizon)

    def sample_proposals(self, current159, history, quality=1., *, passes=20, horizon=30,
                         generator=None, initial_noise=None):
        return self._sample(current159, history, mode=0, quality=quality, passes=passes, horizon=horizon,
                            generator=generator, initial_noise=initial_noise)

    def predict_actions(self, current159, history, actions, *, passes=20, generator=None,
                        initial_noise=None, known_noise=None):
        """Quality is always absent; no private future targets enter this query."""
        return self._sample(current159, history, mode=1, actions=actions, passes=passes, horizon=actions.shape[1],
                            generator=generator, initial_noise=initial_noise, known_noise=known_noise)

    def refine(self, current159, history, original, quality=1., *, passes=8, midpoint=4, generator=None):
        clean = original["clean"] if isinstance(original, dict) else original
        horizon = clean.shape[1]
        padded = torch.zeros(len(clean), self.config.horizon, JOINT_DIM, device=clean.device)
        padded[:, :horizon] = clean
        return self._sample(current159, history, mode=0, quality=quality, passes=passes, horizon=horizon,
                            initial_clean=padded, start_index=midpoint, generator=generator)

    @torch.no_grad()
    def evidence(self, current159, history, next_public, reward, actual_action, *,
                 times=(.2, .4, .6, .8), noise=None, generator=None):
        """Shared corruption across hypotheses; future hidden states never supplied.

        Returns alpha²/sigma² times normalized clean errors. The equivalent
        epsilon errors are reported as an implementation identity, not weighted again.
        noise shape[T,58] contains55 public, one reward and two action noises.
        """
        b, device = len(current159), current159.device
        ts = torch.as_tensor(times, dtype=current159.dtype, device=device)
        if ts.ndim != 1 or bool(((ts <= 0) | (ts >= 1)).any()):
            raise ValueError("Evidence times must lie strictly inside(0,1)")
        observed = torch.as_tensor(next_public, device=device, dtype=current159.dtype).expand(b, PUBLIC_DIM)
        rewards = torch.as_tensor(reward, device=device, dtype=current159.dtype).expand(b)
        action = torch.as_tensor(actual_action, device=device, dtype=current159.dtype).expand(b, 2)
        if noise is None:
            noise = torch.randn(len(ts), PUBLIC_DIM + 3, device=device, generator=generator)
        if noise.shape != (len(ts), PUBLIC_DIM + 3):
            raise ValueError("Evidence common noise must be[T,58]:55 public, reward, two action noises")
        clean = torch.zeros(b, self.config.horizon, JOINT_DIM, device=device)
        clean[:, 0, self.public_indices] = (observed - self.state_mean[self.public_indices]) / self.state_scale[self.public_indices]
        clean[:, 0, REWARD_INDEX] = rewards
        clean[:, 0, ACTION_SLICE] = 2 * action - 1
        modes = torch.full((b,), 2, device=device, dtype=torch.long)
        target, known = self.query_masks(modes)
        context = self.history_context(history)
        clean_errors, epsilon_errors, energies = [], [], []
        for index, t in enumerate(ts):
            eps = torch.zeros_like(clean)
            eps[:, 0, self.public_indices] = noise[index, :PUBLIC_DIM]
            eps[:, 0, REWARD_INDEX] = noise[index, PUBLIC_DIM]
            eps[:, 0, ACTION_SLICE] = noise[index, PUBLIC_DIM + 1:]
            a, s = cosine_schedule(t)
            noisy = torch.where(target | known, a * clean + s * eps, torch.zeros_like(clean))
            predicted = self(noisy, t.expand(b), current159, history, torch.zeros(b, device=device),
                             torch.zeros(b, device=device), modes, target, known,
                             known_values=clean, history_context=context)
            reconstructed, predicted_epsilon = self.denoise_components(noisy, t, predicted)
            err = (reconstructed - clean).square()
            e = err[:, 0, self.public_indices].mean(-1) + err[:, 0, REWARD_INDEX]
            epsilon = (predicted_epsilon - eps).square()
            equivalent = epsilon[:, 0, self.public_indices].mean(-1) + epsilon[:, 0, REWARD_INDEX]
            clean_errors.append(e); epsilon_errors.append(equivalent); energies.append((a / s).square() * e)
        return dict(energy=torch.stack(energies).mean(0), energy_by_time=torch.stack(energies, 1),
                    clean_error_by_time=torch.stack(clean_errors, 1),
                    epsilon_error_by_time=torch.stack(epsilon_errors, 1), times=ts, common_noise=noise)

def masked_losses(predicted, target_noise, target_mask):
    """Equal block means, rather than letting159 states overwhelm2 actions/reward."""
    errors = (predicted - target_noise).square()
    values, present = [], []
    for selected in (slice(0, STATE_DIM), ACTION_SLICE, slice(REWARD_INDEX, JOINT_DIM)):
        mask = target_mask[..., selected]
        denominator = mask.sum((1, 2))
        values.append((errors[..., selected] * mask).sum((1, 2)) / denominator.clamp_min(1))
        present.append(denominator > 0)
    values, present = torch.stack(values, -1), torch.stack(present, -1)
    return (values.sum(-1) / present.sum(-1).clamp_min(1)).mean(), values

class JointWindows:
    """Only saved causal histories and contiguous H30 starts, loaded once per split."""
    def __init__(self, arrays, device, cost_scale):
        required = ("observations", "privileged", "next_privileged", "actions", "costs", "previous_actions",
            "previous_costs", "previous_cost_valid", "history_indices_h24", "history_mask_h24",
            "future_starts_h30", "quality", "quality_valid")
        if any(k not in arrays for k in required):
            raise ValueError("Use the completed shared-stock dataset with causal masks and quality validity")
        self.n = len(arrays["costs"])
        self.data = {}
        for key in required:
            value = arrays[key]
            dtype = torch.long if key in ("history_indices_h24", "future_starts_h30") else (
                torch.bool if key in ("previous_cost_valid", "history_mask_h24", "quality_valid") else torch.float32)
            self.data[key] = torch.as_tensor(value, dtype=dtype, device=device)
        self.starts = self.data["future_starts_h30"]
        self.cost_scale = float(cost_scale)
        if not len(self.starts):
            raise ValueError("No complete H30 windows")

    def batch(self, indices, history_lengths):
        d = self.data
        raw_indices = d["history_indices_h24"][indices]
        mask = d["history_mask_h24"][indices] & (torch.arange(24, device=indices.device)[None] >= 24-history_lengths[:, None])
        safe = raw_indices.clamp_min(0)
        reward_valid = mask & d["previous_cost_valid"][safe]
        history = dict(observations=d["observations"][safe], previous_actions=d["previous_actions"][safe],
            previous_rewards=(-d["previous_costs"][safe] / self.cost_scale)[..., None],
            valid=mask, reward_valid=reward_valid)
        future = indices[:, None] + torch.arange(30, device=indices.device)[None]
        return dict(current=d["privileged"][indices], history=history, next_private=d["next_privileged"][future],
            actions=d["actions"][future], rewards=-d["costs"][future] / self.cost_scale,
            quality=d["quality"][indices], quality_valid=d["quality_valid"][indices])

def training_loss(model, batch, *, generator=None, modes=None, lengths=None, times=None, noise=None):
    b, device = len(batch["current"]), batch["current"].device
    if modes is None:
        modes = torch.randint(3, (b,), device=device, generator=generator)
    if lengths is None:
        options = torch.tensor([1, 4, 8, 30], device=device)
        lengths = options[torch.randint(4, (b,), device=device, generator=generator)]
        lengths = torch.where(modes == 0, torch.full_like(lengths, 30), lengths)
    if times is None:
        times = .001 + (model.config.maximum_time-.001) * torch.rand(b, device=device, generator=generator)
    clean = model.pack(batch["next_private"], batch["actions"], batch["rewards"])
    noise = torch.randn(clean.shape, device=device, generator=generator) if noise is None else noise
    target, known = model.query_masks(modes, lengths)
    a, s = cosine_schedule(times[:, None, None])
    noisy = torch.where(target | known, a * clean + s * noise, torch.zeros_like(clean))
    present = batch["quality_valid"] & (modes == 0)
    present &= torch.rand(b, device=device, generator=generator) >= model.config.quality_dropout
    context = model.history_context(batch["history"])
    prediction = model(noisy, times, batch["current"], batch["history"], batch["quality"], present,
                       modes, target, known, known_values=clean, history_context=context)
    native_target = a * noise - s * clean if model.config.prediction_type == "v" else noise
    denoising, blocks = masked_losses(prediction, native_target, target)
    mu, log_scale = model.initializer(context).chunk(2, -1)
    log_scale = log_scale.clamp(-5, 2)
    actual = ((batch["current"] - model.state_mean) / model.state_scale)[:, model.hidden_indices]
    initializer = (.5 * ((actual-mu) / log_scale.exp()).square() + log_scale).mean()
    loss = denoising + initializer
    return loss, dict(denoising=denoising, initializer_nll=initializer, hidden_mean_mse=(actual-mu).square().mean(),
                      block_losses=blocks, prediction=prediction, noise=noise,
                      native_target=native_target, target_mask=target)

def slice_history(history, indices):
    return {k: v[indices] for k, v in history.items()}

@torch.no_grad()
def validate(model, windows, indices, *, batch_size=128, sample_windows=64, sample_passes=10):
    """Fixed windows/noise, three histories, and separately labeled sample errors."""
    model.eval()
    device = model.state_mean.device
    histories, selection = {}, []
    for history_length in HISTORY_LENGTHS:
        generator = torch.Generator(device=device).manual_seed(model.config.seed + 773)
        accum = {q: [] for q in QUERY_NAMES}
        hidden, coverage = [], []
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            batch = windows.batch(idx, torch.full_like(idx, history_length))
            init = model.initialize(batch["history"])
            actual = ((batch["current"] - model.state_mean) / model.state_scale)[:, model.hidden_indices]
            errors = actual - init["hidden_mean_normalized"]
            hidden.append(errors.square().sum())
            coverage.append((errors.abs() <= 1.644853627 * init["hidden_log_scale"].exp()).sum())
            for mode, name in enumerate(QUERY_NAMES):
                modes = torch.full_like(idx, mode)
                _, metrics = training_loss(model, batch, generator=generator, modes=modes,
                                            lengths=torch.full_like(idx, 30))
                accum[name].append(metrics["denoising"] * len(idx))
        values = {name: float(torch.stack(accum[name]).sum().cpu()) / len(indices) for name in QUERY_NAMES}
        selection.extend(values.values())
        values["hidden_mean_mse"] = float(torch.stack(hidden).sum().cpu()) / (len(indices) * len(HIDDEN_INDICES))
        values["hidden_coverage90"] = float(torch.stack(coverage).sum().cpu()) / (len(indices) * len(HIDDEN_INDICES))
        if sample_windows:
            idx = indices[:min(sample_windows, len(indices))]
            batch = windows.batch(idx, torch.full_like(idx, history_length))
            generator = torch.Generator(device=device).manual_seed(model.config.seed + 1773)
            forecast = model.predict_actions(batch["current"], batch["history"], batch["actions"],
                                             passes=sample_passes, generator=generator)
            # Known quality only: missing originals use explicit unconditional generation.
            # Batch mixed-null conditioning is covered by fixed-noise proposal diagnostics;
            # sampled conditional-action error below restricts to genuine quality labels.
            valid = batch["quality_valid"]
            proposal_metrics = dict(windows=int(valid.sum().cpu()))
            if bool(valid.any()):
                proposal = model.sample_proposals(batch["current"][valid], slice_history(batch["history"], valid),
                    quality=batch["quality"][valid], passes=sample_passes, generator=generator)
                proposal_metrics.update(action_mse=float((proposal["actions"]-batch["actions"][valid]).square().mean().cpu()),
                    reward_mse=float((proposal["rewards"]-batch["rewards"][valid]).square().mean().cpu()))
            values["sampled_proposal"] = proposal_metrics
            values["imposed_action_forecast"] = {}
            for h in (1, 4, 8, 30):
                delta = (forecast["next_private"][:, :h]-batch["next_private"][:, :h]) / model.state_scale
                values["imposed_action_forecast"][str(h)] = dict(
                    state_normalized_mse=float(delta.square().mean().cpu()),
                    public_normalized_mse=float(delta[..., model.public_indices].square().mean().cpu()),
                    reward_mse=float((forecast["rewards"][:, :h]-batch["rewards"][:, :h]).square().mean().cpu()))
        histories[str(history_length)] = values
    return dict(selection_score=float(np.mean(selection)), by_history=histories,
                selector="Equal mean fixed-noise denoising loss over three query modes and histories1/8/24",
                sampled_errors="Single fixed-noise DDIM samples versus recorded outcomes/actions; not distribution likelihoods")

def reserve_decision(curve, initial_updates):
    """Report the fixed additional100k criterion; this function launches nothing."""
    before = [r for r in curve if r["updates"] <= initial_updates-20000]
    recent = [r for r in curve if r["updates"] == initial_updates]
    result = dict(eligible=False, additional_updates=100000,
                  rule="Final200k combined fixedVAL score improves by >0.5% relative to the best score at or before180k")
    if initial_updates < 40000 or not before or not recent:
        return result | dict(reason="Insufficient fixed-budget curve intervals")
    prior = min(r["validation"]["selection_score"] for r in before)
    final = recent[-1]["validation"]["selection_score"]
    relative = (prior-final) / max(abs(prior), 1e-12)
    result.update(best_score_through_prior20k=prior, final_score=final, relative_improvement=relative,
                  eligible=relative > .005)
    return result

def load_model(checkpoint, device="cuda"):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if saved.get("model_class") != MODEL_CLASS:
        raise ValueError("Joint diffusion requires a fresh joint checkpoint; action-only weights are incompatible")
    model = JointDiffusion(JointConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    model.recipe = saved["recipe"]
    model.checkpoint_updates = saved["updates"]
    return model.eval()

def train(args):
    device = torch.device(args.device)
    if device.type != "cuda" and (not args.test_device or args.updates > 10):
        raise ValueError("Production joint training requires CUDA; --test-device is only a <=10-update implementation test")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stop without CPU fallback or automatic retry")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    output, data_path = Path(args.output).resolve(), Path(args.data).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh model output directory required; preserve previous checkpoints")
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source = None
    if args.resume:
        # RNG byte tensors and non-capturable Adam step counters must remain
        # on CPU. load_state_dict moves model weights/optimizer moments to the
        # actual parameter device; mapping the entire checkpoint to CUDA makes
        # torch.cuda.set_rng_state_all reject its saved CPU ByteTensors.
        source = torch.load(args.resume, map_location="cpu", weights_only=False)
        if source.get("model_class") != MODEL_CLASS or Path(source["recipe"]["data"]).resolve() != data_path:
            raise ValueError("Continuation requires a joint checkpoint on this exact shared dataset")
        config = JointConfig(**source["config"])
    else:
        config = JointConfig(width=args.width, learning_rate=args.learning_rate, seed=args.seed,
                             prediction_type=getattr(args, "prediction_type", "epsilon"))
    training, validation = load_arrays(data_path / "train.npz"), load_arrays(data_path / "validation.npz")
    model = JointDiffusion(config).to(device)
    model.set_normalization(training)
    if source:
        model.load_state_dict(source["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_windows = JointWindows(training, device, model.cost_scale.item())
    val_windows = JointWindows(validation, device, model.cost_scale.item())
    rng = np.random.default_rng(config.seed + 4401)
    candidates = validation["future_starts_h30"]
    indices_numpy = rng.choice(candidates, size=min(args.validation_windows, len(candidates)), replace=False)
    # One fixed random subset/order; no replacement based on observed errors.
    np.save(output / "validation_windows.npy", indices_numpy)
    val_indices = torch.as_tensor(indices_numpy, device=device, dtype=torch.long)
    del training, validation
    recipe = dict(model_class=MODEL_CLASS, config=asdict(config), data=str(data_path),
        source_checkpoint=str(Path(args.resume).resolve()) if args.resume else None,
        fresh_initialization=not bool(source), seed=config.seed, updates=args.updates, batch_size=args.batch_size,
        training_windows=len(train_windows.starts), validation_windows=len(val_indices),
        train_transitions=train_windows.n, validation_transitions=val_windows.n,
        reward="negative interval daily economic cost divided by positive TRAIN mean",
        cost_scale=float(model.cost_scale.cpu()), public_indices=PUBLIC_INDICES.tolist(), hidden_dimensions=len(HIDDEN_INDICES),
        output="Per lead[next_private159, normalized_actions2, reward1], H30",
        histories=list(HISTORY_LENGTHS), history_training="Uniform requested1/8/24 with explicit startup and previous-reward masks",
        queries=dict(proposal="all targets; quality validity plus20% dropout; missing quality explicitly absent",
                     forecast="states/reward, imposed full-box actions; quality absent; active prefix1/4/8/30",
                     evidence="only H1 public55+reward targets, actual action known; hidden future absent"),
        query_sampling="Uniform three query modes; all TRAIN windows retained without quality filtering/weights",
        loss="Equal state/action/reward block means of native prediction error over present targets, plus weight1 normalized hidden Gaussian NLL",
        prediction_type=config.prediction_type,
        prediction_target="alpha*epsilon-sigma*x0" if config.prediction_type == "v" else "epsilon",
        initializer="Only104 hidden coordinates supervised; mean/logscale head; public55 clamped exactly",
        temporal_model="Causal dilated convolutions, per-lead LayerNorm; no future-action input to earlier leads",
        sampler="Deterministic DDIM conditional on initial noise; cosine alpha/sigma; actions re-noised and reimposed each update",
        sample_projection="Original[0,1] action box and nonnegative physical concentrations/flow; no upper state bound",
        evidence="Generalized energy alpha²/sigma² times x0 public-mean error plus reward error; four common corruptions",
        checkpoint_selector="Equal fixed-noise denoising mean across proposal/forecast/evidence and histories1/8/24",
        reserve="One additional100k block only if final200k combined fixedVAL improves >0.5% over best through180k; explicit continuation",
        device=str(device), precision="float32; no autocast or TF32 matmul", historical_checkpoints_modified=False)
    curve, best, best_update, start_update = [], float("inf"), 0, 0
    if source:
        optimizer.load_state_dict(source["optimizer"])
        curve = source["learning_curve"]
        best, best_update, start_update = source["best_validation_loss"], source["best_updates"], source["updates"]
        if args.updates <= start_update:
            raise ValueError("Continuation target must exceed saved updates")
        if not np.array_equal(source["validation_windows"], indices_numpy):
            raise ValueError("Continuation must reuse fixed validation windows/count")
        torch.set_rng_state(source["rng"]["torch"].cpu())
        np.random.set_state(source["rng"]["numpy"]); random.setstate(source["rng"]["python"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in source["rng"]["cuda"]])
        # Keep the best from the original allocation eligible during continuation.
        original_best = Path(args.resume).resolve().parent / "best.pt"
        best_payload = torch.load(original_best, map_location="cpu", weights_only=False)
        atomic_checkpoint(output / "best.pt", best_payload)
    atomic_json(output / "recipe.json", recipe)
    last_progress, accumulated = time.monotonic(), []

    def payload(update, score):
        state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate())
        if device.type == "cuda":
            state["cuda"] = torch.cuda.get_rng_state_all()
        return dict(model_class=MODEL_CLASS, config=asdict(config), model=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=state, recipe=recipe, updates=update,
            validation_loss=score, best_validation_loss=best, best_updates=best_update,
            validation_windows=indices_numpy, learning_curve=curve)

    def progress(status, update, **extra):
        atomic_json(output / "progress.json", dict(status=status, stage="joint_diffusion", updates=update,
            target_updates=args.updates, updated_at=time.time(), elapsed_seconds=time.time()-started, **extra))

    def validation_point(update, train_metrics):
        nonlocal best, best_update
        progress("running", update, phase="validation")
        metrics = validate(model, val_windows, val_indices, batch_size=args.batch_size,
                           sample_windows=args.sample_windows, sample_passes=args.sample_passes)
        score = metrics["selection_score"]
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite fixed validation score")
        curve.append(dict(updates=update, training=train_metrics, validation=metrics, elapsed_seconds=time.time()-started))
        if score < best:
            best, best_update = score, update
            atomic_checkpoint(output / "best.pt", payload(update, score))
        atomic_checkpoint(output / "last.pt", payload(update, score))
        atomic_json(output / "learning_curve.json", curve)
        progress("running", update, phase="training", validation_score=score, best_updates=best_update)
        print(json.dumps(dict(updates=update, validation_score=score, best_updates=best_update)), flush=True)

    update = start_update
    try:
        if not source:
            validation_point(0, None)
        for update in range(start_update + 1, args.updates + 1):
            model.train()
            starts = train_windows.starts[torch.randint(len(train_windows.starts), (args.batch_size,), device=device)]
            lengths = torch.tensor(HISTORY_LENGTHS, device=device)[torch.randint(3, (args.batch_size,), device=device)]
            batch = train_windows.batch(starts, lengths)
            loss, metrics = training_loss(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            accumulated.append(torch.stack((loss.detach(), metrics["denoising"].detach(), metrics["initializer_nll"].detach())))
            if time.monotonic()-last_progress >= 10 or update == 1:
                values = torch.stack(accumulated).mean(0).cpu().tolist()
                if not all(math.isfinite(v) for v in values):
                    raise FloatingPointError("Nonfinite joint training objective")
                progress("running", update, phase="training", loss=values[0], denoising_loss=values[1], initializer_nll=values[2])
                accumulated.clear(); last_progress = time.monotonic()
            if update % args.validate_every == 0 or update == args.updates:
                values = torch.stack(accumulated).mean(0).cpu().tolist() if accumulated else [float(loss.detach().cpu()),
                    float(metrics["denoising"].detach().cpu()), float(metrics["initializer_nll"].detach().cpu())]
                accumulated.clear()
                validation_point(update, dict(loss=values[0], denoising=values[1], initializer_nll=values[2]))
                last_progress = time.monotonic()
        receipt = dict(status="complete", model_class=MODEL_CLASS, updates=args.updates, best_updates=best_update,
            selected_checkpoint=str(output / "best.pt"), final_checkpoint=str(output / "last.pt"),
            best_validation_loss=best, train_transitions=train_windows.n, validation_transitions=val_windows.n,
            training_windows=len(train_windows.starts), fresh_initialization=not bool(source),
            cost_scale=float(model.cost_scale.cpu()), elapsed_seconds=time.time()-started,
            reserve_check=reserve_decision(curve, min(args.updates, 200000)), history_lengths=list(HISTORY_LENGTHS), horizon=30)
        if source:
            receipt["reserve_check"].update(eligible=False, continuation_already_used=True)
        atomic_json(output / "completion.json", receipt)
        progress("complete", args.updates, best_updates=best_update, best_validation_loss=best)
        return receipt
    except BaseException as error:
        # Never query a potentially failed CUDA context to manufacture a checkpoint.
        atomic_json(output / "failure.json", dict(status="failed", updates=update, error=repr(error),
            traceback=traceback.format_exc(), last_preserved_checkpoint=str(output / "last.pt"), automatic_retry=False))
        progress("failed", update, error=repr(error))
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train",))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--updates", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--prediction-type", choices=("epsilon", "v"), default="epsilon")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--validate-every", type=int, default=10000)
    parser.add_argument("--validation-windows", type=int, default=512)
    parser.add_argument("--sample-windows", type=int, default=64)
    parser.add_argument("--sample-passes", type=int, default=10)
    parser.add_argument("--resume", help="Explicit checkpoint continuation into a fresh output; never automatic retry")
    parser.add_argument("--test-device", action="store_true", help="Intentional <=10-update CPU implementation test only")
    train(parser.parse_args())

if __name__ == "__main__":
    main()
