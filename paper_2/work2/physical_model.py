"""Explicit physical features for a preserved shared diffusion checkpoint.

No original source, checkpoint, normalization, network parameter, or sampler
schedule is changed. Native integer queries remain available. Continuous queries
must explicitly use physical_features = [(gravity - 1) / 2, friction_bit, joint_bit].
Importing this module constructs neither a model nor an environment.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
import torch


from work2.native.model import SharedDiffusion
from work2.native.optimized_proposal_control import _finite, _noise, _numpy, _tensor, public_history
from work2.native.unified_comparison_control import UnifiedComparisonEngine, UnifiedOraclePolicy
from work2.native.environment import PaperEnv

ADAPTER_SCHEMA = 'explicit_gravity_coordinate_v1'
ENV_GRAVITIES = (1., 1.5, 3., 5.)


def physical_features(gravity, friction_bit, joint_bit):
    """Explicit query coordinate; no integer label casting or clipping."""
    if isinstance(gravity, (bool, np.bool_)) or not math.isfinite(float(gravity)) or not 1 <= float(gravity) <= 5:
        raise ValueError('Declared gravity coordinates cover finite physical gravity from 1× through 5×')
    if friction_bit not in (0, 1) or joint_bit not in (0, 1):
        raise ValueError('Friction and joint features must be binary')
    return np.array([(float(gravity) - 1) / 2, friction_bit, joint_bit], dtype=np.float32)


def feature_tensor(value, n, device):
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.shape != (n, 3) or not torch.isfinite(result).all():
        raise ValueError('One finite physical feature vector [N,3] is required per query')
    if ((result[:, 0] < 0) | (result[:, 0] > 2)).any() or not ((result[:, 1:] == 0) | (result[:, 1:] == 1)).all():
        raise ValueError('Gravity coordinate must be in [0,2], and friction/joint features must be binary')
    return result


def endpoint_features(labels):
    if labels.dtype != torch.long or labels.ndim != 1 or ((labels < 0) | (labels > 7)).any():
        raise ValueError('Endpoint conversion requires native integer labels in [0,7]')
    return torch.stack(((labels >> 2) & 1, (labels >> 1) & 1, labels & 1), -1).float()


class PhysicalSharedDiffusion(SharedDiffusion):
    """Same state_dict and native loss; only explicit condition features extend it."""

    @classmethod
    def from_native(cls, native):
        if not isinstance(native, SharedDiffusion) or not native.config.condition_input:
            raise ValueError('A native conditional shared model is required')
        if native._forward_pre_hooks or native._forward_hooks:
            raise ValueError('Attach the physical model before installing query-statistics hooks')
        model = copy.deepcopy(native)
        # No constructor, new parameter initialization, or RNG consumption.
        model.__class__ = cls
        original_state = native.state_dict()
        if model.state_dict().keys() != original_state.keys():
            raise AssertionError('Physical adapter changed checkpoint keys')
        for key, value in model.state_dict().items():
            if not torch.equal(value, original_state[key]):
                raise AssertionError(f'Physical adapter changed tensor {key}')
        return model

    def encode(self, batch):
        if 'physical_features' not in batch:
            return super().encode(batch)
        n = len(batch['actions'])
        features = feature_tensor(batch['physical_features'], n, batch['actions'].device)
        native = dict(batch)
        # This placeholder is not a physical label. forward uses the explicit
        # features exclusively when present; all native masks/context stay exact.
        native.setdefault('condition', torch.zeros(n, dtype=torch.long, device=features.device))
        encoded = super().encode(native)
        encoded['physical_features'] = features
        return encoded

    def forward(self, noisy, encoded, time):
        if 'physical_features' not in encoded:
            return super().forward(noisy, encoded, time)
        bits = feature_tensor(encoded['physical_features'], len(noisy), noisy.device)
        known = torch.where(encoded['known'], encoded['x'], 0)
        features = torch.stack((time, torch.sin(math.pi * time), torch.cos(math.pi * time), time.square()), -1)
        context = (self.context(encoded['context']) + self.condition(bits) + self.role(encoded['role'])
                   + self.quality(encoded['quality']) + self.horizon(encoded['horizon']) + self.time(features))
        x = (self.input(torch.cat((noisy, known, encoded['known'].float()), -1))
             + self.position[:, :noisy.shape[1]] + context[:, None])
        for block in self.blocks:
            x = block(x, context)
        x = self.norm(x)
        return self.output(x), self.termination(x[:, 1::2]).squeeze(-1)

    @torch.no_grad()
    def sample_physical(self, history, physical_features, horizon=8, fixed_actions=None,
                        noise=None, generator=None, quality=1, temperature=1.0, steps=None):
        """Native cosine sampler with an explicit physical-feature batch field."""
        self.eval()
        device = next(self.parameters()).device
        h = horizon
        if not 1 <= h <= self.config.max_horizon:
            raise ValueError('Unsupported sampling horizon')
        n = len(physical_features)
        features = feature_tensor(physical_features, n, device)
        fixed = fixed_actions is not None
        if fixed:
            if fixed_actions.shape != (n, h, self.config.action_dim):
                raise ValueError('Invalid imposed-action shape')
            if not torch.isfinite(fixed_actions).all() or (fixed_actions.abs() > 1.000001).any():
                raise ValueError('Imposed actions must be finite and physically bounded')
        actions = fixed_actions.float() if fixed else torch.zeros(n, h, self.config.action_dim, device=device)
        zeros = torch.zeros(n, h, device=device)
        batch = dict(history=history, physical_features=features,
                     condition=torch.zeros(n, dtype=torch.long, device=device), actions=actions,
                     future_observations=torch.zeros(n, h, self.config.observation_dim, device=device),
                     rewards=zeros, valid=torch.ones_like(zeros, dtype=torch.bool),
                     horizon=torch.full((n,), h, dtype=torch.long, device=device),
                     role=torch.full((n,), (1 if h == 1 else 2) if fixed else 0, dtype=torch.long, device=device),
                     quality=torch.full((n,), 0 if fixed else quality, dtype=torch.long, device=device))
        encoded = self.encode(batch)
        if noise is None:
            noise = torch.randn(encoded['x'].shape, device=device, generator=generator)
        if noise.shape != encoded['x'].shape:
            raise ValueError('Sampling noise must match the complete query shape')
        x = torch.where(encoded['known'], encoded['x'], noise * temperature) * encoded['semantic']
        grid = torch.linspace(.999, 0, (steps or self.config.sampling_steps) + 1, device=device)
        for time, next_time in zip(grid[:-1], grid[1:], strict=True):
            prediction, _ = self(x, encoded, time.expand(n))
            prediction = prediction.float()
            alpha, sigma = torch.cos(time * math.pi / 2), torch.sin(time * math.pi / 2)
            epsilon = (x - alpha * prediction) / sigma.clamp_min(1e-5)
            x = torch.cos(next_time * math.pi / 2) * prediction + torch.sin(next_time * math.pi / 2) * epsilon
            x = torch.where(encoded['known'], encoded['x'], x) * encoded['semantic']
        _, terminal = self(x, encoded, torch.zeros(n, device=device))
        returned_actions = x[:, ::2, :self.config.action_dim].clamp(-1, 1)
        if fixed:
            returned_actions = fixed_actions.clone()
        observations = (x[:, 1::2, :self.config.observation_dim] * self.delta_std
                        + self.delta_mean + history['observations'][:, -1, None])
        rewards = x[:, 1::2, -1] * self.reward_std + self.reward_mean
        result = dict(actions=returned_actions, termination_probability=terminal.float().sigmoid())
        prefix = '' if fixed else 'joint_'
        result[prefix + 'observations'] = observations
        result[prefix + 'rewards'] = rewards
        return result
