"""Native actual3×/5× physics for the included specialist collection."""
from dataclasses import dataclass
import mujoco
import numpy as np
from work2.native.environment import PaperEnv

@dataclass(frozen=True)
class ExpertCondition:
    index: int
    gravity: float
    sliding_friction: float
    joint_range: float

    @classmethod
    def from_index(cls, gravity, index):
        if isinstance(gravity, (bool, np.bool_)) or float(gravity) not in (3., 5.):
            raise ValueError('This isolated expert study supports actual gravity 3× or 5× only')
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 4 <= index <= 7:
            raise ValueError('Expert condition must be an integer from 4 through 7')
        return cls(int(index), float(gravity), (1., .5)[(index >> 1) & 1], (1., .8)[index & 1])

    @property
    def bits(self):
        # The first bit means target-gravity slot, never the physical magnitude.
        return np.array([(self.index >> bit) & 1 for bit in (2, 1, 0)], dtype=np.float32)


class ExpertGravityEnv(PaperEnv):
    """Original native HalfCheetah interface with explicit fixed target gravity."""

    def __init__(self, gravity, condition=4, max_episode_steps=1000, render_mode=None):
        spec = ExpertCondition.from_index(gravity, condition)
        self.target_gravity = spec.gravity
        super().__init__('HalfCheetah-v5', condition, max_episode_steps, render_mode)
        if not np.isclose(self.unwrapped.dt, .05, rtol=0, atol=1e-12):
            raise ValueError('Expected native HalfCheetah 50 ms transitions')
        self.audit_condition()

    def apply_condition(self, condition):
        spec = ExpertCondition.from_index(self.target_gravity, condition)
        model, data = self.unwrapped.model, self.unwrapped.data
        model.opt.gravity[:] = self.nominal['gravity'] * spec.gravity
        model.geom_friction[:] = self.nominal['friction']
        model.geom_friction[:, 0] *= spec.sliding_friction
        model.jnt_range[:] = self.nominal['joint_range']
        ranges = self.nominal['joint_range'][self.joint_ids]
        centers = ranges.mean(1)
        widths = (ranges[:, 1] - ranges[:, 0]) * .5 * spec.joint_range
        model.jnt_range[self.joint_ids] = np.stack([centers - widths, centers + widths], 1)
        self.condition = spec
        warmstart = data.qacc_warmstart.copy()
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = warmstart

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.audit_condition()
        return observation, info

    def snapshot(self):
        result = super().snapshot()
        result['actual_gravity_scale'] = self.target_gravity
        return result

    def restore(self, snapshot, condition=None):
        if snapshot.get('actual_gravity_scale') != self.target_gravity:
            raise ValueError('Expert snapshot gravity missing or mismatched; no implicit cross-gravity restore')
        observation = super().restore(snapshot, condition=condition)
        self.audit_condition()
        return observation

    def audit_condition(self):
        spec, model = self.condition, self.unwrapped.model
        np.testing.assert_allclose(model.opt.gravity, self.nominal['gravity'] * spec.gravity, rtol=0, atol=1e-12)
        friction = self.nominal['friction'].copy(); friction[:, 0] *= spec.sliding_friction
        np.testing.assert_allclose(model.geom_friction, friction, rtol=0, atol=1e-12)
        ranges = self.nominal['joint_range'].copy()
        limited = ranges[self.joint_ids]
        centers = limited.mean(1)
        widths = (limited[:, 1] - limited[:, 0]) * .5 * spec.joint_range
        ranges[self.joint_ids] = np.stack([centers - widths, centers + widths], 1)
        np.testing.assert_allclose(model.jnt_range, ranges, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(model.body_mass, self.nominal['body_mass'])
        np.testing.assert_array_equal(model.actuator_gear, self.nominal['actuator_gear'])
        return dict(env_id='HalfCheetah-v5', condition=spec.index, nominal_bits=f'{spec.index:03b}',
                    gravity_scale=spec.gravity, gravity_vector=model.opt.gravity.tolist(),
                    sliding_friction_scale=spec.sliding_friction, joint_range_scale=spec.joint_range,
                    native_dt=float(self.unwrapped.dt), frame_skip=int(self.unwrapped.frame_skip),
                    observation_dim=17, action_dim=6, body_mass_unchanged=True, actuator_gear_unchanged=True,
                    condition_index_alone_does_not_encode_gravity=True)
