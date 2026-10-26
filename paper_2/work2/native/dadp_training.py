"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np

import torch

from .dadp_adapter import (
    REPRESENTATION_NUMERICS,
    DADPPolicy,
    make_planner,
    make_representation,
    representation_precision,
)

from .windows import WindowSampler, to_device

class DADPWindows:
    """Same corpus, explicit stationary representation pairs and supported action chunks."""

    def __init__(self, output, arm="R", split="train", seed=13, value_bounds=None):
        self.base = WindowSampler(output, arm, split, seed, history=17, horizon=4)
        self.rng = self.base.rng
        self.normalization = self.base.normalization
        self.mean = np.asarray(self.normalization["observation_mean"], np.float32)
        self.std = np.asarray(self.normalization["observation_std"], np.float32)
        data, anchors = self.base.arrays, self.base.anchors
        past = anchors[:, None] + np.arange(-16, 2)
        # Paired cross-prediction needs a constant physical domain over each
        # observed history and target. A counterfactual root cannot be assigned
        # the future domain before its effects have been observed.
        stationary = (data["condition"][past] == data["condition"][anchors, None]).all(1)
        representation = (
            data["observation_valid"][past].all(1)
            & data["valid"][anchors]
            & data["valid"][anchors + 1]
            & stationary
        )
        future = anchors[:, None] + np.arange(4)
        supported = data["valid"][future].all(1) & data["eligible"][future].all(1)
        self.representation_pools, self.policy_pools = [], []
        for condition in range(8):
            labels = data["condition"][anchors] == condition
            self.representation_pools.append(anchors[labels & representation])
            self.policy_pools.append(anchors[labels & supported])
        if any(not len(pool) for pool in self.representation_pools + self.policy_pools):
            raise ValueError(
                "DADP needs stationary representation and supported policy windows in every cell"
            )
        # Finite recorded return-to-go; collection endpoints do not pretend to be
        # native deaths. These weights are ranking weights, not value targets.
        self.values = np.zeros(len(data["rewards"]), np.float32)
        offset = 0
        for row in self.base.rows:
            begin, n = offset + 16, row["length"]
            value = 0.0
            for t in range(begin + n - 1, begin - 1, -1):
                value = float(data["rewards"][t]) + 0.997 * value
                self.values[t] = value
            offset += 16 + n + 1 + 4
        eligible_values = self.values[np.concatenate(self.policy_pools)]
        if value_bounds is None:
            if split != "train":
                raise ValueError("DADP reward-weight scaling must be fit on TRAIN only")
            value_bounds = [float(eligible_values.min()), float(eligible_values.max())]
        self.value_bounds = list(value_bounds)
        if value_bounds[1] <= value_bounds[0]:
            raise ValueError("Degenerate DADP TRAIN return range")

    def draw(self, pools, batch):
        if batch % 8:
            raise ValueError("DADP minibatches must balance the eight physical regimes")
        labels = np.tile(np.arange(8), batch // 8)
        anchors = np.empty(batch, np.int64)
        for condition in range(8):
            mask = labels == condition
            anchors[mask] = self.rng.choice(pools[condition], mask.sum())
        return anchors, labels

    def representation(self, batch=128):
        first, labels = self.draw(self.representation_pools, batch)
        second, second_labels = self.draw(self.representation_pools, batch)
        if not np.array_equal(labels, second_labels):
            raise ValueError("Cross-prediction pairs must share the physical domain")
        anchors = np.stack((first, second), 1)
        indices = anchors[..., None] + np.arange(-16, 3)
        data = self.base.arrays
        states = (data["observations"][indices] - self.mean) / self.std
        factors = ((labels[:, None] & np.array([4, 2, 1])) > 0).astype(np.float32)
        return {
            "obs": {"state": states},
            "act": data["actions"][indices],
            "task_list": np.repeat(factors[:, None], 2, 1),
        }

    def policy(self, batch=256):
        anchors, _ = self.draw(self.policy_pools, batch)
        indices = anchors[:, None] + np.arange(-16, 4)
        data = self.base.arrays
        observations = data["observations"][indices].copy()
        observed = data["observation_valid"][indices]
        first = observed.argmax(1)
        initial = observations[np.arange(batch), first]
        observations = np.where(observed[..., None], observations, initial[:, None])
        trajectory = np.concatenate(
            ((observations - self.mean) / self.std, data["actions"][indices]), -1
        )
        low, high = self.value_bounds
        normalized = 2 * (self.values[anchors] - low) / (high - low) - 1
        weights = np.exp((normalized - 1) * 2).astype(np.float32)
        return {"trajectory": trajectory, "weight": weights[:, None]}

def load_policy(path, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=False)
    if state["stage"] != "policy":
        raise ValueError("Expected a DADP policy checkpoint")
    representation, planner = make_representation(device), make_planner(device)
    representation.load_state_dict(state["representation"])
    planner.model.load_state_dict(state["model"])
    planner.model_ema.load_state_dict(state["ema"])
    return DADPPolicy(representation, planner, state["normalization"]), state

def stage_loss(stage, model, representation, batch):
    device_type = next(representation.parameters()).device.type
    if stage == "representation":
        with representation_precision(device_type):
            result = representation.train_step(batch)
        return result["total_loss"], {
            key: value.detach() for key, value in result["losses"].items()
        }
    with torch.no_grad(), representation_precision(device_type):
        embedding = representation.dynamics.encode_history(
            batch["trajectory"][:, :16, :17], batch["trajectory"][:, :16, 17:]
        )
    loss = model.loss(
        batch["trajectory"],
        embedding=embedding[:, None].expand(-1, 20, 23),
        weighted_regression_tensor=batch["weight"],
    )
    return loss, {}

def optimization_step(stage, model, representation, batch, optimizer):
    device_type = next(representation.parameters()).device.type
    execution = (
        representation_precision(device_type) if stage == "representation" else nullcontext()
    )
    # Keep TF32 disabled through representation backward as well as forward.
    with execution:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=stage == "policy"):
            loss, _ = stage_loss(stage, model, representation, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("FATAL_DADP: nonfinite training loss")
        loss.backward()
        parameters = (
            representation.parameters() if stage == "representation" else model.model.parameters()
        )
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if stage == "policy":
            model.ema_update()
    return loss.detach(), norm.detach()

@torch.no_grad()
def dev_loss(stage, model, representation, sampler, batch_size):
    old = sampler.rng.bit_generator.state
    sampler.rng = np.random.default_rng(813_013)
    representation.eval()
    if stage == "policy":
        model.eval()
        original_model = model.model
        model.model = model.model_ema
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(813_013)
            values = []
            for _ in range(8):
                batch = to_device(getattr(sampler, stage)(batch_size))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = stage_loss(stage, model, representation, batch)
                values.append(float(loss))
    finally:
        sampler.rng.bit_generator.state = old
        if stage == "policy":
            model.model = original_model
    return float(np.mean(values))

def load_representation(path, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=False)
    if state["stage"] != "representation":
        raise ValueError("A selected DADP representation checkpoint is required")
    representation = make_representation(device)
    representation.load_state_dict(state["model"])
    representation.eval()
    return representation, state
