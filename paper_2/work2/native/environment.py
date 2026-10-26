"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import copy

from dataclasses import asdict, dataclass

import gymnasium as gym

import mujoco

import numpy as np

JOINTS = {
    "HalfCheetah-v5": ("bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"),

}

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

@dataclass(frozen=True)
class Condition:
    index: int
    gravity: float
    sliding_friction: float
    joint_range: float

    @classmethod
    def from_index(cls, index: int) -> Condition:
        if not isinstance(index, (int, np.integer)) or not 0 <= index < 8:
            raise ValueError(f"Condition must be an integer in [0, 7], got {index}")
        return cls(
            int(index),
            (1.0, 1.5)[(index >> 2) & 1],
            (1.0, 0.5)[(index >> 1) & 1],
            (1.0, 0.8)[index & 1],
        )

    @property
    def bits(self) -> np.ndarray:
        return np.array([(self.index >> bit) & 1 for bit in (2, 1, 0)], dtype=np.float32)

class PaperEnv(gym.Wrapper):
    """Truth stays in this evaluator wrapper; observations and rewards stay native."""

    def __init__(
        self,
        env_id: str = "HalfCheetah-v5",
        condition: int = 0,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        *,
        frame_skip: int | None = None,
    ):
        if env_id not in JOINTS:
            raise ValueError(f"Unsupported Paper 2 environment: {env_id}")
        if frame_skip is not None and (
            isinstance(frame_skip, (bool, np.bool_))
            or not isinstance(frame_skip, (int, np.integer))
            or frame_skip < 1
        ):
            raise ValueError("frame_skip must be a positive integer or None")
        timing = {} if frame_skip is None else {"frame_skip": int(frame_skip)}
        super().__init__(
            gym.make(env_id, max_episode_steps=max_episode_steps, render_mode=render_mode, **timing)
        )
        self.env_id = env_id
        self.initial_condition = int(condition)
        self.condition = Condition.from_index(condition)
        model = self.unwrapped.model
        self.nominal = {
            "gravity": model.opt.gravity.copy(),
            "friction": model.geom_friction.copy(),
            "joint_range": model.jnt_range.copy(),
            "actuator_gear": model.actuator_gear.copy(),
            "body_mass": model.body_mass.copy(),
        }
        self.joint_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS[env_id]]
        )
        if (self.joint_ids < 0).any() or not model.jnt_limited[self.joint_ids].all():
            raise ValueError("The expected limited leg hinges are absent")
        if model.npair:
            raise ValueError("Explicit contact pairs need a declared friction adapter")
        self.elapsed = 0
        self.apply_condition(condition)

    def apply_condition(self, condition: int) -> None:
        spec = Condition.from_index(condition)
        model, data = self.unwrapped.model, self.unwrapped.data
        # Always derive from nominal arrays. No compounding and no state projection.
        model.opt.gravity[:] = self.nominal["gravity"] * spec.gravity
        model.geom_friction[:] = self.nominal["friction"]
        model.geom_friction[:, 0] *= spec.sliding_friction
        model.jnt_range[:] = self.nominal["joint_range"]
        nominal_ranges = self.nominal["joint_range"][self.joint_ids]
        centers = nominal_ranges.mean(axis=1)
        widths = (nominal_ranges[:, 1] - nominal_ranges[:, 0]) * 0.5 * spec.joint_range
        model.jnt_range[self.joint_ids] = np.stack([centers - widths, centers + widths], 1)
        self.condition = spec
        # mj_forward refreshes forces/contacts without advancing qpos/qvel/time.
        warmstart = data.qacc_warmstart.copy()
        mujoco.mj_forward(model, data)
        data.qacc_warmstart[:] = warmstart

    def reset(self, *, seed=None, options=None):
        self.apply_condition(self.initial_condition)
        observation, info = self.env.reset(seed=seed, options=options)
        self.elapsed = 0
        return observation.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Action has invalid shape or nonfinite entries")
        if (action < self.action_space.low - 1e-6).any() or (
            action > self.action_space.high + 1e-6
        ).any():
            raise ValueError("Action violates the common physical bounds")
        data = self.unwrapped.data
        warning_ids = [
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
            mujoco.mjtWarning.mjWARN_BADCTRL,
        ]
        warning_counts = [data.warning[int(kind)].number for kind in warning_ids]
        before = float(data.time)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed += 1
        if (
            not np.isfinite(observation).all()
            or not np.isfinite(reward)
            or data.time < before
            or any(
                data.warning[int(kind)].number > count
                for kind, count in zip(warning_ids, warning_counts, strict=True)
            )
        ):
            raise FloatingPointError("FATAL_SIMULATION: nonfinite or reset MuJoCo transition")
        return observation.astype(np.float32), float(reward), terminated, truncated, info

    def diagnostics(self) -> dict:
        """Offline diagnostics; never included in the controller's public history."""
        base = self.unwrapped
        torso = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        up = float(base.data.xmat[torso].reshape(3, 3)[2, 2])
        angles = base.data.qpos[base.model.jnt_qposadr[self.joint_ids]]
        limits = base.model.jnt_range[self.joint_ids]
        return {
            "inverted": up < 0,
            "torso_up": up,
            "near_joint_limit": bool(
                np.minimum(abs(angles - limits[:, 0]), abs(angles - limits[:, 1])).min() < 0.05
            ),
            "contacts": int(base.data.ncon),
        }

    def snapshot(self) -> dict:
        base = self.unwrapped
        state = np.empty(mujoco.mj_stateSize(base.model, STATE_SPEC), dtype=np.float64)
        mujoco.mj_getState(base.model, base.data, state, STATE_SPEC)
        return {
            "integration": state,
            "condition": self.condition.index,
            "elapsed": self.elapsed,
            "rng": copy.deepcopy(base.np_random.bit_generator.state),
        }

    def restore(self, snapshot: dict, condition: int | None = None) -> np.ndarray:
        self.apply_condition(snapshot["condition"] if condition is None else condition)
        base = self.unwrapped
        mujoco.mj_setState(base.model, base.data, snapshot["integration"], STATE_SPEC)
        mujoco.mj_forward(base.model, base.data)
        # Restore warmstart and other integration fields changed by a forward pass.
        mujoco.mj_setState(base.model, base.data, snapshot["integration"], STATE_SPEC)
        base.np_random.bit_generator.state = copy.deepcopy(snapshot["rng"])
        self.elapsed = int(snapshot["elapsed"])
        wrapper = self.env
        while wrapper is not base:
            if isinstance(wrapper, gym.wrappers.TimeLimit):
                wrapper._elapsed_steps = self.elapsed
            wrapper = wrapper.env
        return base._get_obs().astype(np.float32)

    def specification(self) -> dict:
        model = self.unwrapped.model
        return {
            "env_id": self.env_id,
            "observation_dim": self.observation_space.shape[0],
            "action_dim": self.action_space.shape[0],
            "dt": self.unwrapped.dt,
            "frame_skip": self.unwrapped.frame_skip,
            "condition": asdict(self.condition),
            "joint_names": list(JOINTS[self.env_id]),
            "nominal_joint_ranges": self.nominal["joint_range"][self.joint_ids],
            "actual_joint_ranges": model.jnt_range[self.joint_ids],
            "gravity": model.opt.gravity,
            "geom_friction": model.geom_friction,
            "geom_priority": model.geom_priority,
            "geom_condim": model.geom_condim,
            "explicit_contact_pairs": model.npair,
            "observation_has_condition": False,
        }


def make_monitored_env(env_id: str, condition: int, seed: int):
    from stable_baselines3.common.monitor import Monitor

    env = PaperEnv(env_id, condition)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return Monitor(env)
