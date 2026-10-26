"""Yang transition-filter/CVAE/PAL baseline on shared public55 data.

Published network families and full 100/2000/30 stage budgets are preserved.
The common task changes measurements, objective and decision interval. All
training is offline; complete recorded 30-step returns alone enter the filter,
while administrative dataset/time boundaries do not suppress Bellman bootstrap.
"""

from __future__ import annotations

import argparse

import copy

from dataclasses import asdict, dataclass

from datetime import datetime, timezone

import json

import math

import os

import time

from pathlib import Path

from typing import Any

import numpy as np

import torch

from torch import nn

PUBLIC_DIM = 55

MODEL_CLASS = "StockYangPublic55"

@dataclass
class YangConfig:
    seed: int = 17
    envelope_batch_size: int = 10
    cvae_batch_size: int = 100
    policy_batch_size: int = 100
    envelope_epochs: int = 100
    cvae_epochs: int = 2000
    policy_epochs: int = 30
    return_steps: int = 30
    discount: float = 0.9
    filter_threshold: float = 1.1
    envelope_penalty: float = 5.0
    envelope_lr: float = 1e-4
    cvae_lr: float = 1e-4
    critic_lr: float = 1e-4
    actor_lr: float = 1e-6
    envelope_l2: float = 1e-6
    cvae_l2: float = 1e-8
    candidates: int = 10
    residual_fraction: float = 0.5
    target_rate: float = 0.001
    pal_alpha: float = 0.5
    progress_seconds: float = 10.0

    def __post_init__(self):
        counts = (self.envelope_batch_size, self.cvae_batch_size, self.policy_batch_size,
                  self.envelope_epochs, self.cvae_epochs, self.policy_epochs,
                  self.return_steps, self.candidates)
        if min(counts) < 1 or not 0 < self.discount <= 1 or not 0 < self.pal_alpha <= 1:
            raise ValueError("Positive stage sizes and valid discount/PAL exponent are required")

ADAPTATIONS = {
    "identity": "Yang2025 transition-filter/CVAE/residual-actor/PAL, paper-derived adaptation",
    "source": "https://doi.org/10.1016/j.neucom.2025.129977",
    "observation": "55 shared public sensor/influent channels, TRAIN standardized",
    "objective": "shared EQ + 0.3 OCI; fixed TRAIN-derived additive shift if needed, then positive TRAIN mean scale",
    "interval": "15 minutes; gamma=0.9 per decision; 30-step return=7.5 hours",
    "action": "shared [Qa/92230, KLa5/240] in [0,1], not original [Qa/1e5,KLa5/500]",
    "minibatches": "envelope10/CVAE100/policy100; complete100/2000/30 passes; policy100 is a declared WWTP-unstated choice",
    "missing_settings": "PAL alpha=0.5 declared; WWTP actor/critic batch not specified",
    "critic_initialization": "standard torch Linear initialization; no critic biases; ReLU output",
    "cvae_output": "sigmoid normalized bounded action; Gaussian reconstruction factor 0.5",
    "cvae_kl": "twice conventional Gaussian KL, matching published equation 32",
    "validation": "envelope asymmetric MSE; CVAE mean-latent reconstruction; policy Bellman PAL",
    "filter": "Only full30-step returns; V <= 1.1 V_hat; no partial tail returns or filter relaxation",
    "checkpoint_selection": "fixed final endpoint of each stage; best-validation files diagnostic only",
    "bootstrapping": "physical_terminal/bootstrap_mask only; time limits and split boundaries bootstrap actual stored next state",
}

def complete_window_returns(costs, episode_id, step, terminated, horizon=30, discount=0.9):
    """Full recorded H-step returns only; NaN marks unavailable tail windows."""
    c = np.asarray(costs, dtype=np.float64)
    episodes, steps, ends = np.asarray(episode_id), np.asarray(step), np.asarray(terminated, dtype=bool)
    if c.ndim != 1 or any(x.shape != c.shape for x in (episodes, steps, ends)):
        raise ValueError("Aligned scalar transition columns are required")
    if not np.isfinite(c).all() or horizon < 1 or not 0 < discount <= 1:
        raise ValueError("Invalid costs or horizon/discount")
    result = np.full(len(c), np.nan, dtype=np.float32)
    weights = discount ** np.arange(horizon)
    for episode in np.unique(episodes):
        ids = np.flatnonzero(episodes == episode)
        ids = ids[np.argsort(steps[ids], kind="stable")]
        if len(np.unique(steps[ids])) != len(ids):
            raise ValueError("Duplicate episode/step record")
        cuts = np.flatnonzero((np.diff(steps[ids]) != 1) | ends[ids[:-1]]) + 1
        for segment in np.split(ids, cuts):
            if len(segment) >= horizon:
                windows = np.lib.stride_tricks.sliding_window_view(c[segment], horizon)
                result[segment[:len(windows)]] = windows @ weights
    return result, np.isfinite(result)

def pal_loss(delta: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Equations 26–27, with detached importance normalizer (gradient equivalence)."""
    if not 0 < alpha <= 1:
        raise ValueError("PAL alpha must be in (0,1]")
    magnitude = delta.abs()
    eta = magnitude.detach().pow(alpha).clamp_min(1).mean()
    loss = torch.where(magnitude <= 1, 0.5 * delta.square(), magnitude.pow(1 + alpha) / (1 + alpha))
    return loss.mean() / eta

def envelope_loss(prediction, returns, penalty=5.0):
    weights = torch.where(prediction >= returns, penalty, 1.0)
    return (weights * (prediction - returns).square()).mean()

def transition_filter(returns, predictions, threshold=1.1):
    """Exact published retention inequality; caller handles an empty set explicitly."""
    return np.asarray(returns) <= threshold * np.asarray(predictions)

def stage_indices(count: int, retained: np.ndarray) -> dict[str, np.ndarray]:
    retained = np.asarray(retained, dtype=bool)
    if retained.shape != (count,):
        raise ValueError("Filter mask shape mismatch")
    return {"envelope": np.arange(count), "cvae": np.arange(count), "policy": np.flatnonzero(retained)}

def _mlp(input_dim, hidden, output_dim, final=None, *, bias=True):
    layers: list[nn.Module] = []
    for size in hidden:
        layers.extend((nn.Linear(input_dim, size, bias=bias), nn.ReLU()))
        input_dim = size
    layers.append(nn.Linear(input_dim, output_dim, bias=bias))
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)

class LowerEnvelope(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = _mlp(PUBLIC_DIM, (128, 128), 1)
        nn.init.constant_(self.network[-1].bias, 1.0)

    def forward(self, x):
        return self.network(x).squeeze(-1)

class BehaviorCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _mlp(PUBLIC_DIM + 2, (256, 128), 8)
        self.decoder = _mlp(PUBLIC_DIM + 4, (256, 128), 2, nn.Sigmoid())

    def encode(self, x, action):
        mean, log_std = self.encoder(torch.cat((x, action), -1)).chunk(2, -1)
        return mean, log_std.clamp(-3, 5)

    def decode(self, x, z):
        return self.decoder(torch.cat((x, z), -1))

    def loss(self, x, action):
        mean, log_std = self.encode(x, action)
        z = mean + log_std.exp() * torch.randn_like(mean)
        reconstructed = self.decode(x, z)
        reconstruction = 0.5 * (reconstructed - action).square().sum(-1)
        # Eq. 32 omits the conventional KL factor 1/2.
        kl_twice = (-1 + mean.square() + (2 * log_std).exp() - 2 * log_std).sum(-1)
        return (reconstruction + kl_twice).mean()

    def candidates(self, x, count, generator=None):
        batch = x.shape[0]
        z = torch.randn(batch, count, 4, device=x.device, dtype=x.dtype, generator=generator).clamp(-0.5, 0.5)
        repeated = x[:, None, :].expand(-1, count, -1)
        return self.decode(repeated, z)

class CostCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = _mlp(PUBLIC_DIM + 2, (256, 128), 1, nn.ReLU(), bias=False)

    def forward(self, x, action):
        return self.network(torch.cat((x, action), -1)).squeeze(-1)

class ResidualActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = _mlp(PUBLIC_DIM + 2, (128, 64), 2, nn.Tanh())

    def forward(self, x, base, spread, fraction=0.5):
        residual = fraction * spread * self.network(torch.cat((x, base), -1))
        return (base + residual).clamp(0, 1)

def select_candidates(x, candidates, critic):
    values = critic(x[:, None, :].expand(-1, candidates.shape[1], -1), candidates)
    indices = values.argmin(-1)
    base = candidates[torch.arange(len(x), device=x.device), indices]
    spread = candidates.amax(1) - candidates.amin(1)
    return base, spread

def _atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def _atomic_save(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def _now():
    return datetime.now(timezone.utc).isoformat()

def checked_step(loss, optimizer, parameters):
    """Check numerical validity without changing the published gradients."""
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite stock Yang training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return norm

def _load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        keys = ("observations", "next_observations", "actions", "costs", "episode_id", "step", "terminated", "physical_terminal", "bootstrap_mask")
        result = {key: source[key] for key in keys}
    n = len(result["costs"])
    if n == 0 or result["observations"].shape != (n, PUBLIC_DIM):
        raise ValueError("Offline RL requires nonempty public observations [N,55]")
    if result["next_observations"].shape != (n, PUBLIC_DIM) or result["actions"].shape != (n, 2):
        raise ValueError("Next-observation/action shape mismatch")
    for key in ("observations", "next_observations", "actions", "costs"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if np.any((result["actions"] < 0) | (result["actions"] > 1)):
        raise ValueError("Expected normalized actions in [0,1]")
    mask = np.asarray(result["bootstrap_mask"])
    terminal = np.asarray(result["physical_terminal"], dtype=bool)
    if mask.shape != (n,) or terminal.shape != (n,) or not np.array_equal(mask, (~terminal).astype(mask.dtype)):
        raise ValueError("bootstrap_mask must equal one minus physical_terminal, independent of administrative ends")
    return result

class Trainer:
    def __init__(self, data: Path, output: Path, device: str, config: YangConfig):
        self.config, self.output = config, Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Use a fresh stock Yang output; no automatic restart")
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        raw_train = _load_npz(Path(data) / "train.npz")
        raw_validation = _load_npz(Path(data) / "validation.npz")
        self.mean = raw_train["observations"].mean(0, dtype=np.float64).astype(np.float32)
        self.std = raw_train["observations"].std(0, dtype=np.float64).clip(1e-5).astype(np.float32)
        # OCI's inventory derivative can make individual economic costs negative.
        # A single fixed offset preserves differences; do not clip negative costs.
        train_min = float(np.min(raw_train["costs"]))
        self.cost_offset = max(0.0, -train_min + 1e-6) if train_min < 0 else 0.0
        self.cost_scale = float(np.mean(raw_train["costs"], dtype=np.float64)) + self.cost_offset
        if not self.cost_scale > 0:
            raise ValueError("TRAIN mean cost must be positive")
        self.train = self._tensors(raw_train)
        self.validation = self._tensors(raw_validation)
        self.updates = 0
        self.stage_updates = dict(envelope=0, cvae=0, policy=0)
        self.active_stage = "envelope"
        self.started = time.time()
        self.last_progress = 0.0
        self.history: list[dict[str, Any]] = []
        self.envelope = LowerEnvelope().to(self.device)
        self.cvae = BehaviorCVAE().to(self.device)
        self.critic = CostCritic().to(self.device)
        self.actor = ResidualActor().to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        _atomic_json(self.output / "training_contract.json", {
            "config": asdict(config), "adaptations": ADAPTATIONS, "model_class": MODEL_CLASS,
            "train_rows": len(raw_train["costs"]), "validation_rows": len(raw_validation["costs"]),
            "cost_scale_shifted_train_mean": self.cost_scale, "cost_offset_train_derived": self.cost_offset,
            "minimum_unshifted_train_cost": train_min,
            "minimum_shifted_validation_cost": float(np.min(raw_validation["costs"])) + self.cost_offset,
            "train_input": str((Path(data) / "train.npz").resolve()),
            "validation_input": str((Path(data) / "validation.npz").resolve()),
            "evaluation_truth_access": False, "fresh_initialization": True,
            "train_complete_return_rows": int(self.train["return_valid"].sum().item()),
            "validation_complete_return_rows": int(self.validation["return_valid"].sum().item()),
            "expected_envelope_updates": config.envelope_epochs * math.ceil(int(self.train["return_valid"].sum().item()) / config.envelope_batch_size),
            "expected_cvae_updates": config.cvae_epochs * math.ceil(len(raw_train["costs"]) / config.cvae_batch_size),
        })

    def _tensors(self, raw):
        returns, valid = complete_window_returns(
            (raw["costs"] + self.cost_offset) / self.cost_scale, raw["episode_id"], raw["step"], raw["terminated"],
            self.config.return_steps, self.config.discount,
        )
        values = {
            "x": (raw["observations"] - self.mean) / self.std,
            "next_x": (raw["next_observations"] - self.mean) / self.std,
            "action": raw["actions"], "cost": (raw["costs"] + self.cost_offset) / self.cost_scale,
            "bootstrap_mask": raw["bootstrap_mask"], "returns": np.nan_to_num(returns, nan=0.), "return_valid": valid,
        }
        return {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in values.items()}

    def progress(self, stage, epoch, total_epochs, **extra):
        self.active_stage = stage
        self.last_progress = time.monotonic()
        timestamp = _now()
        _atomic_json(self.output / "progress.json", {
            "status": "running", "stage": stage, "epoch": epoch, "total_epochs": total_epochs,
            "updates": self.updates, "stage_updates": self.stage_updates, "elapsed_seconds": time.time()-self.started,
            "updated_at": timestamp, "last_progress_utc": timestamp, **extra,
        })

    def _batches(self, indices, shuffle=True):
        indices = np.asarray(indices)
        order = self.rng.permutation(indices) if shuffle else indices
        batch_size = getattr(self.config, self.active_stage + "_batch_size", self.config.policy_batch_size)
        for start in range(0, len(order), batch_size):
            yield torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=self.device)

    def _tick(self, stage, epoch, total_epochs):
        self.updates += 1
        self.stage_updates[stage] += 1
        if time.monotonic() - self.last_progress >= self.config.progress_seconds:
            self.progress(stage, epoch, total_epochs)

    def _payload(self, stage, epoch, optimizers, best_score):
        return {
            "format": "stock_yang_public55_v1", "model_class": MODEL_CLASS, "stage": stage, "epoch": epoch,
            "config": asdict(self.config), "adaptations": ADAPTATIONS,
            "observation_mean": self.mean, "observation_std": self.std, "cost_scale": self.cost_scale,
            "cost_offset": self.cost_offset,
            "envelope": self.envelope.state_dict(), "cvae": self.cvae.state_dict(),
            "critic": self.critic.state_dict(), "actor": self.actor.state_dict(),
            "target_critic": self.target_critic.state_dict(), "target_actor": self.target_actor.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in optimizers],
            "best_validation_score": best_score, "updates": self.updates, "stage_updates": dict(self.stage_updates),
            "torch_rng": torch.get_rng_state(), "numpy_rng": self.rng.bit_generator.state,
            "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
        }

    def restore_stage(self, stage):
        state = torch.load(self.output / f"{stage}_final.pt", map_location=self.device, weights_only=False)
        # Model normalization is always based solely on the identical TRAIN data.
        if (not np.array_equal(state["observation_mean"], self.mean)
                or state["cost_scale"] != self.cost_scale or state["cost_offset"] != self.cost_offset):
            raise ValueError("Checkpoint/data normalizers disagree")
        names = {"envelope": ("envelope",), "cvae": ("cvae",),
                 "policy": ("critic", "actor", "target_critic", "target_actor")}[stage]
        for name in names:
            getattr(self, name).load_state_dict(state[name])

    def _finish_epoch(self, stage, epoch, total, train_loss, validation_loss, best, optimizers):
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise FloatingPointError(f"Nonfinite {stage} loss")
        improved = validation_loss < best
        best = min(best, validation_loss)
        entry = {"stage": stage, "epoch": epoch, "updates": self.updates,
                 "training_loss": train_loss, "validation_loss": validation_loss}
        self.history.append(entry)
        with (self.output / "training_history.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        payload = self._payload(stage, epoch, optimizers, best)
        _atomic_save(self.output / f"{stage}_latest.pt", payload)
        if epoch == total:
            _atomic_save(self.output / f"{stage}_final.pt", payload)
        if improved:
            _atomic_save(self.output / f"{stage}_best.pt", payload)
        self.progress(stage, epoch, total, **{"training_loss": train_loss, "validation_loss": validation_loss})
        return best

    @torch.no_grad()
    def _validation_loss(self, stage):
        totals, count = 0.0, 0
        # Fixed generator makes validation comparable across epochs.
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed + 1051)
        indices = (torch.nonzero(self.validation["return_valid"], as_tuple=False).flatten().cpu().numpy()
                   if stage == "envelope" else np.arange(len(self.validation["cost"])))
        if not len(indices):
            raise ValueError("Validation has no complete filter-return windows")
        for ids in self._batches(indices, shuffle=False):
            d = {key: value[ids] for key, value in self.validation.items()}
            if stage == "envelope":
                loss = envelope_loss(self.envelope(d["x"]), d["returns"], self.config.envelope_penalty)
            elif stage == "cvae":
                mean, _ = self.cvae.encode(d["x"], d["action"])
                loss = 0.5 * (self.cvae.decode(d["x"], mean) - d["action"]).square().sum(-1).mean()
            else:
                target = self._critic_target(d, generator)
                loss = pal_loss(self.critic(d["x"], d["action"]) - target, self.config.pal_alpha)
            totals += float(loss) * len(ids)
            count += len(ids)
        return totals / count

    def train_envelope(self):
        config = self.config
        optimizer = torch.optim.Adam(self.envelope.parameters(), lr=config.envelope_lr, weight_decay=2 * config.envelope_l2)
        best = math.inf
        indices = torch.nonzero(self.train["return_valid"], as_tuple=False).flatten().cpu().numpy()
        if not len(indices):
            raise ValueError("TRAIN has no full30-step recorded returns")
        self.progress("envelope", 0, config.envelope_epochs)
        for epoch in range(1, config.envelope_epochs + 1):
            total = 0.0
            for ids in self._batches(indices):
                loss = envelope_loss(self.envelope(self.train["x"][ids]), self.train["returns"][ids], config.envelope_penalty)
                checked_step(loss, optimizer, self.envelope.parameters())
                total += float(loss.detach()) * len(ids)
                self._tick("envelope", epoch, config.envelope_epochs)
            best = self._finish_epoch("envelope", epoch, config.envelope_epochs, total / len(indices),
                                      self._validation_loss("envelope"), best, [optimizer])
        self.restore_stage("envelope")

    @torch.no_grad()
    def retained(self):
        prediction = torch.cat([self.envelope(self.train["x"][ids])
                                for ids in self._batches(np.arange(len(self.train["cost"])), False)]).cpu().numpy()
        returns = self.train["returns"].cpu().numpy()
        valid = self.train["return_valid"].bool().cpu().numpy()
        retained = valid & transition_filter(returns, prediction, self.config.filter_threshold)
        np.savez_compressed(self.output / "transition_filter.npz", returns=returns, prediction=prediction, complete_return=valid, retained=retained)
        _atomic_json(self.output / "filter_summary.json", {
            "rows": len(retained), "retained_rows": int(retained.sum()), "retention_fraction": float(retained.mean()),
            "threshold": self.config.filter_threshold, "cvae_rows": len(retained),
            "complete_return_rows": int(valid.sum()), "excluded_incomplete_return_rows": int((~valid).sum()),
            "inequality": "recorded_return <= threshold * predicted_lower_envelope",
        })
        if not retained.any():
            raise RuntimeError("Transition filter retained no data; preserve checkpoint and inspect development fit")
        return retained

    def train_cvae(self):
        config = self.config
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=config.cvae_lr, weight_decay=2 * config.cvae_l2)
        best = math.inf
        # Deliberately includes transitions excluded from actor/critic training.
        indices = stage_indices(len(self.train["cost"]), np.ones(len(self.train["cost"]), dtype=bool))["cvae"]
        self.progress("cvae", 0, config.cvae_epochs)
        for epoch in range(1, config.cvae_epochs + 1):
            total = 0.0
            for ids in self._batches(indices):
                loss = self.cvae.loss(self.train["x"][ids], self.train["action"][ids])
                checked_step(loss, optimizer, self.cvae.parameters())
                total += float(loss.detach()) * len(ids)
                self._tick("cvae", epoch, config.cvae_epochs)
            best = self._finish_epoch("cvae", epoch, config.cvae_epochs, total / len(indices),
                                      self._validation_loss("cvae"), best, [optimizer])
        self.restore_stage("cvae")

    @torch.no_grad()
    def _critic_target(self, d, generator=None):
        alternatives = self.cvae.candidates(d["next_x"], self.config.candidates, generator)
        base, spread = select_candidates(d["next_x"], alternatives, self.target_critic)
        action = self.target_actor(d["next_x"], base, spread, self.config.residual_fraction)
        return d["cost"] + self.config.discount * d["bootstrap_mask"] * self.target_critic(d["next_x"], action)

    def train_policy(self):
        config = self.config
        self.cvae.requires_grad_(False).eval()
        indices = stage_indices(len(self.train["cost"]), self.retained())["policy"]
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        best = math.inf
        self.progress("policy", 0, config.policy_epochs)
        for epoch in range(1, config.policy_epochs + 1):
            total = 0.0
            for ids in self._batches(indices):
                d = {key: value[ids] for key, value in self.train.items()}
                target = self._critic_target(d)
                critic_loss = pal_loss(self.critic(d["x"], d["action"]) - target, config.pal_alpha)
                checked_step(critic_loss, critic_optimizer, self.critic.parameters())
                with torch.no_grad():
                    alternatives = self.cvae.candidates(d["x"], config.candidates)
                    base, spread = select_candidates(d["x"], alternatives, self.target_critic)
                amended = self.actor(d["x"], base, spread, config.residual_fraction)
                actor_loss = pal_loss(self.target_critic(d["x"], amended), config.pal_alpha)
                checked_step(actor_loss, actor_optimizer, self.actor.parameters())
                with torch.no_grad():
                    for current, target_net in ((self.critic, self.target_critic), (self.actor, self.target_actor)):
                        for parameter, target_parameter in zip(current.parameters(), target_net.parameters()):
                            target_parameter.lerp_(parameter, config.target_rate)
                total += float(critic_loss.detach()) * len(ids)
                self._tick("policy", epoch, config.policy_epochs)
            best = self._finish_epoch("policy", epoch, config.policy_epochs, total / len(indices),
                                      self._validation_loss("policy"), best, [critic_optimizer, actor_optimizer])
        self.restore_stage("policy")
        payload = self._payload("complete", config.policy_epochs, [critic_optimizer, actor_optimizer], best)
        _atomic_save(self.output / "final.pt", payload)
        _atomic_json(self.output / "completion.json", {"status": "complete", "method": "stock_yang", "model_class": MODEL_CLASS,
            "completed_at": _now(), "updates": self.updates, "stage_updates": dict(self.stage_updates),
            "epochs": {"envelope": config.envelope_epochs, "cvae": config.cvae_epochs, "policy": config.policy_epochs},
            "selected_checkpoint": str((self.output / "final.pt").resolve()),
            "controller_checkpoint": str((self.output / "final.pt").resolve()),
            "checkpoint_selection": "fixed final endpoint for each stage; best validation diagnostic only",
            "train_transitions": len(self.train["cost"]), "validation_transitions": len(self.validation["cost"]),
            "train_complete_return_rows": int(self.train["return_valid"].sum().item()), "policy_retained_rows": len(indices),
            "fresh_initialization": True, "elapsed_seconds": time.time()-self.started})
        self.progress("complete", config.policy_epochs, config.policy_epochs, status="complete")

class YangController:
    """Frozen policy. Public observations only; no private-state argument exists."""
    def __init__(self, payload, device="cuda"):
        if payload.get("model_class") != MODEL_CLASS:
            raise ValueError("Expected stock Yang public55 checkpoint")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        self.config = YangConfig(**payload["config"])
        self.mean = torch.as_tensor(payload["observation_mean"], device=self.device)
        self.std = torch.as_tensor(payload["observation_std"], device=self.device)
        self.cvae = BehaviorCVAE().to(self.device).requires_grad_(False).eval()
        self.critic = CostCritic().to(self.device).requires_grad_(False).eval()
        self.actor = ResidualActor().to(self.device).requires_grad_(False).eval()
        for name in ("cvae", "critic", "actor"):
            getattr(self, name).load_state_dict(payload[name])
        self.generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

    @torch.inference_mode()
    def act(self, observation, previous_action=None):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (PUBLIC_DIM,) or not np.isfinite(observation).all():
            raise ValueError("Yang controller requires exactly 55 finite public channels")
        x = ((torch.as_tensor(observation, device=self.device) - self.mean) / self.std)[None]
        alternatives = self.cvae.candidates(x, self.config.candidates, self.generator)
        base, spread = select_candidates(x, alternatives, self.critic)
        action = self.actor(x, base, spread, self.config.residual_fraction)[0].cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError("Nonfinite offline RL action")
        return action

def load_controller(path, device="cuda"):
    path = Path(path)
    if path.is_dir():
        path = path / "final.pt"
    return YangController(torch.load(path, map_location=device, weights_only=False), device)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--device", default="cuda")
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--envelope-epochs", type=int, default=100)
    training.add_argument("--cvae-epochs", type=int, default=2000)
    training.add_argument("--policy-epochs", type=int, default=30)
    training.add_argument("--cpu-threads", type=int, default=1)
    training.add_argument("--implementation-test", action="store_true")
    args = parser.parse_args(argv)
    config = YangConfig(**{key: getattr(args, key) for key in ("seed", "envelope_epochs", "cvae_epochs", "policy_epochs")})
    if not args.implementation_test:
        if (config.seed, config.envelope_epochs, config.cvae_epochs, config.policy_epochs) != (17, 100, 2000, 30):
            parser.error("Production stock Yang budgets/seed are frozen")
        if torch.device(args.device).type != "cuda":
            parser.error("Production training requires CUDA; explicit implementation tests alone use CPU")
    elif torch.device(args.device).type != "cpu":
        parser.error("Implementation tests must use CPU")
    if min(config.envelope_epochs, config.cvae_epochs, config.policy_epochs) < 1:
        parser.error("Stage epochs must be positive")
    torch.set_num_threads(args.cpu_threads)
    trainer = None
    try:
        trainer = Trainer(args.data, args.output, args.device, config)
        trainer.train_envelope()
        trainer.train_cvae()
        trainer.train_policy()
    except BaseException as exc:
        progress = args.output / "progress.json"
        if trainer is not None:
            trainer.progress(trainer.active_stage, 0, 0, status="failed", failure_type=type(exc).__name__, failure=str(exc))
        elif progress.exists():
            value = json.loads(progress.read_text())
            value.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
            _atomic_json(progress, value)
        raise

if __name__ == "__main__":
    main()
