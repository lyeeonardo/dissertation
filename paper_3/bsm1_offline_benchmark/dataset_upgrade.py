"""Development-only public policy selection and a broader shared offline bank.

The original bank is read-only. Fresh test forcing parameters are frozen before
development and never simulated by this module. No privileged state enters a
collection controller.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from .environment import ACTION_SCALE

NEW_BEHAVIORS = ("development_expert", "perturbed_expert", "full_range_excitation", "boundary_holds")

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)

class PublicExpertPolicy:
    """Feedback and optional current-flow feedforward from public57 only."""

    def __init__(self, specification, seed=17, perturb=False):
        self.specification = dict(specification)
        self.rng = np.random.default_rng(seed)
        self.previous_error = np.zeros(2)
        self.feedback = np.array([55338., 84.])
        self.previous_flow = None
        self.perturb = bool(perturb)
        self.target_offset = np.zeros(2)
        self.step = 0

    def action(self, step, observation):
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (57,) or not np.isfinite(obs).all():
            raise ValueError("Public controller requires exactly 57 finite measured channels")
        spec = self.specification
        do_target = spec["do_target"]
        if "ammonium_gain" in spec:
            do_target = np.clip(do_target + spec["ammonium_gain"] * (obs[38] - 1.5), 2., 5.)
        if self.perturb and step % 96 == 0:
            self.target_offset = self.rng.uniform([-.15, -.3], [.15, .3])
        targets = np.array([spec["no_target"], do_target]) + self.target_offset
        error = targets - obs[[42, 41]]
        gains = np.array([3000., 18.])
        integral_time_days = np.array([.08, .03])
        update = gains * (error - self.previous_error) + gains / 96 / integral_time_days * error
        if self.previous_flow is not None and "flow_feedforward" in spec:
            update[1] += spec["flow_feedforward"] * self.feedback[1] * (
                obs[-1] - self.previous_flow) / max(self.previous_flow, 1.)
        self.feedback = np.clip(self.feedback + update, 0, ACTION_SCALE)
        self.previous_error = error
        self.previous_flow = obs[-1]
        action = self.feedback / ACTION_SCALE
        if self.perturb:
            action += self.rng.normal(0, .015, 2)
        return np.clip(action, 0, 1)

    def act(self, observation, previous_action=None):
        action = self.action(self.step, observation)
        self.step += 1
        return action

class UpgradeCollectionPolicy:
    def __init__(self, name, expert_specification, seed=17):
        if name not in NEW_BEHAVIORS:
            raise ValueError(name)
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.expert = PublicExpertPolicy(expert_specification, seed, name == "perturbed_expert")
        self.target = np.array([.6, .35])
        self.boundary_schedule = np.array([[q, a] for a in (0., 1.) for q in (0., .5, 1.)])
        self.rng.shuffle(self.boundary_schedule)

    def action(self, step, observation):
        obs = np.asarray(observation)
        if obs.shape != (57,):
            raise ValueError("Collection policy receives public57 only")
        if self.name in ("development_expert", "perturbed_expert"):
            return self.expert.action(step, obs)
        if self.name == "full_range_excitation":
            if step % 20 == 0:
                self.target = self.rng.uniform(0, 1, 2)
            # Han-style full-range mean and 5% independent action perturbation;
            # unlike v1, there is no mixing with feedback to narrow the range.
            return np.clip(self.target + self.rng.normal(0, .05, 2), 0, 1)
        return self.boundary_schedule[(step // 20) % len(self.boundary_schedule)].copy()
