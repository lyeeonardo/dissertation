"""Bounded shared offline bank: 32 forcing parents crossed with four behaviors.

All methods receive the same public transition bank. Full physical labels are
kept in separately named privileged arrays for the authorized diffusion oracle.
Split assignments are made before collection; official source weather templates
are shared across splits, so these are new synthetic realizations, not new weather.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np

from .environment import ACTION_SCALE, COMPONENT_NAMES, OBSERVATION_NAMES, SCHEMA, STATE_NAMES

from simulator.inputs import load_official

ROOT = Path(__file__).resolve().parents[1]

BEHAVIORS = ("nominal_pi", "economical_pi", "ammonium_feedback", "smooth_excitation")

WEATHERS = ("dry", "rain", "storm")

SPLIT_COUNTS = {"train": 20, "validation": 6, "test": 6}

def forcing_for_parent(parent, episode_days=14):
    name = {"dry": "dry", "rain": "rain", "storm": "strm"}[parent["weather"]]
    original = load_official(input_path(f"forcing/Inf_{name}_2006.txt"))
    time = np.arange(round(episode_days * 96) + 1, dtype=float) / 96
    source_time = (time + parent["phase_days"]) % 14
    indices = np.maximum(0, np.searchsorted(original[:, 0], source_time, side="right") - 1)
    values = original[indices].copy()
    values[:, 0] = time
    wave = 1 + parent["wave_amplitude"] * np.sin(2 * np.pi * time / 3.7 + parent["wave_phase"])
    values[:, 15] *= parent["flow_scale"] * wave
    values[:, 1:8] *= parent["carbon_scale"] * wave[:, None]
    values[:, [10, 11, 12]] *= parent["nitrogen_scale"] * (2 - wave[:, None])
    values[:, 14] = .75 * values[:, 3:8].sum(axis=1)
    return values

class CollectionPolicy:
    """Public feedback with slower PI tuning for the 15-minute control interval."""

    def __init__(self, name, seed=17):
        if name not in BEHAVIORS:
            raise ValueError(name)
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.previous_error = np.zeros(2)
        self.feedback = np.array([55338., 84.])
        self.random_target = self.feedback / ACTION_SCALE
        self.do_target = float(self.rng.uniform(1.0, 1.7)) if name == "economical_pi" else 2.0
        self.no_target = float(self.rng.uniform(.8, 1.2)) if name == "economical_pi" else 1.0

    def action(self, step, observation):
        obs = np.asarray(observation)
        do_target = self.do_target
        if self.name == "ammonium_feedback":
            do_target = float(np.clip(1.2 + .28 * (obs[38] - 1.5), .8, 2.5))
        error = np.array([self.no_target - obs[42], do_target - obs[41]])
        gains = np.array([3000., 18.])
        integral_time_days = np.array([.08, .03])
        self.feedback = np.clip(self.feedback + gains * (error - self.previous_error)
                                + gains * (1 / 96) / integral_time_days * error, 0, ACTION_SCALE)
        self.previous_error = error
        action = self.feedback / ACTION_SCALE
        if self.name == "smooth_excitation":
            if step % 20 == 0:
                self.random_target = self.rng.uniform(0, 1, 2)
            action = .25 * action + .75 * self.random_target + self.rng.normal(0, .025, 2)
        return np.clip(action, 0, 1)

from work3_paths import input_path
