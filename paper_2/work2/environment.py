"""Isolated Paper2 physics with gravity {1,3}; original source stays unchanged."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

from work2.native.environment import PaperEnv
import mujoco

GRAVITY_LEVELS = (1., 3.)
LAW = dict(gravity_levels=GRAVITY_LEVELS, sliding_friction_levels=(1., .5), joint_range_levels=(1., .8),
           env_id='HalfCheetah-v5', native_dt=.05)


@dataclass(frozen=True)
class Gravity3Condition:
    index: int
    gravity: float
    sliding_friction: float
    joint_range: float

    @classmethod
    def from_index(cls, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < 8:
            raise ValueError('Condition must be an integer from0 through7')
        return cls(int(index), GRAVITY_LEVELS[(index >> 2) & 1],
                   (1., .5)[(index >> 1) & 1], (1., .8)[index & 1])

    @property
    def bits(self):
        return np.array([(self.index >> bit) & 1 for bit in (2, 1, 0)], np.float32)


class Gravity3Env(PaperEnv):
    """Native observations/reward/timing; evaluator-only condition metadata."""

    def __init__(self, condition=0, max_episode_steps=1000, render_mode=None):
        super().__init__('HalfCheetah-v5', condition, max_episode_steps, render_mode)
        if not np.isclose(self.unwrapped.dt, .05, rtol=0, atol=1e-12):
            raise ValueError('Expected native HalfCheetah50ms transitions')

    def apply_condition(self, condition):
        spec = Gravity3Condition.from_index(condition)
        model, data = self.unwrapped.model, self.unwrapped.data
        # All mutations derive from captured nominal fields, never the current law.
        model.opt.gravity[:] = self.nominal['gravity'] * spec.gravity
        model.geom_friction[:] = self.nominal['friction']
        model.geom_friction[:, 0] *= spec.sliding_friction
        model.jnt_range[:] = self.nominal['joint_range']
        ranges = self.nominal['joint_range'][self.joint_ids]
        center = ranges.mean(1)
        half_width = (ranges[:, 1] - ranges[:, 0]) * .5 * spec.joint_range
        model.jnt_range[self.joint_ids] = np.stack([center - half_width, center + half_width], 1)
        self.condition = spec
        warmstart = data.qacc_warmstart.copy()
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = warmstart

    def audit_condition(self):
        """Read actual simulator parameters; no simulation or dynamics labels in observation."""
        spec = self.condition
        model = self.unwrapped.model
        np.testing.assert_allclose(model.opt.gravity, self.nominal['gravity'] * spec.gravity, rtol=0, atol=1e-12)
        expected_friction = self.nominal['friction'].copy()
        expected_friction[:, 0] *= spec.sliding_friction
        np.testing.assert_allclose(model.geom_friction, expected_friction, rtol=0, atol=1e-12)
        expected_range = self.nominal['joint_range'].copy()
        ranges = expected_range[self.joint_ids]
        centers = ranges.mean(1)
        widths = (ranges[:, 1] - ranges[:, 0]) * .5 * spec.joint_range
        expected_range[self.joint_ids] = np.stack([centers-widths, centers+widths], 1)
        np.testing.assert_allclose(model.jnt_range, expected_range, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(model.body_mass, self.nominal['body_mass'])
        np.testing.assert_array_equal(model.actuator_gear, self.nominal['actuator_gear'])
        return dict(condition=spec.index, bits=f'{spec.index:03b}', gravity_scale=spec.gravity,
                    gravity_vector=model.opt.gravity.tolist(), sliding_friction_scale=spec.sliding_friction,
                    joint_range_scale=spec.joint_range, native_dt=float(self.unwrapped.dt),
                    frame_skip=int(self.unwrapped.frame_skip), observation_dim=17, action_dim=6,
                    body_mass_unchanged=True, actuator_gear_unchanged=True)


def physical_checks():
    """Native simulator parameter/state checks with zero env.step calls."""
    env = Gravity3Env()
    checks = []
    try:
        observation, _ = env.reset(seed=391000099)
        assert observation.shape == (17,) and env.action_space.shape == (6,)
        data = env.unwrapped.data
        preserved = {k: getattr(data, k).copy() for k in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart')}
        clock = float(data.time)
        # Repeated/reversed settings verify no compounding and no hidden resets.
        for condition in list(range(8)) + list(reversed(range(8))):
            env.apply_condition(condition)
            row = env.audit_condition()
            assert float(data.time) == clock and env.elapsed == 0
            for key, expected in preserved.items():
                np.testing.assert_array_equal(getattr(data, key), expected)
            if len(checks) < 8:
                checks.append(row)
        state = env.snapshot()
        restored = env.restore(state, condition=7)
        np.testing.assert_array_equal(restored, observation)
        np.testing.assert_array_equal(env.snapshot()['integration'], state['integration'])
        assert env.condition.gravity == 3. and env.condition.index == 7
        env.audit_condition()
        # Reset restores the declared initial condition rather than the last changed law.
        env.reset(seed=391000099)
        assert env.condition.index == 0
        return dict(status='passed', physical_transitions=0, conditions=checks,
                    no_state_or_time_advance_on_switch=True, snapshot_restore_exact=True,
                    reset_restores_initial_condition=True,
                    original_environment_source=str(Path(__file__).resolve().parent / 'native/environment.py'))
    finally:
        env.close()
