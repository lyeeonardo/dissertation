from __future__ import annotations


import functools

import json

import os

import time

import traceback

from pathlib import Path

import numpy as np

import torch

from stable_baselines3 import SAC

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .common import append_json, atomic_json, atomic_torch_save, require_cuda, restore_rng, rng_state, seed_all

from .environment import PaperEnv, make_monitored_env

def evaluate_expert(model, env_id: str, condition: int, seeds: list[int]) -> dict:
    env = PaperEnv(env_id, condition)
    episodes = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total, speeds, inversions, saturations = 0.0, [], [], []
            for _ in range(1000):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                speeds.append(float(info.get("x_velocity", 0.0)))
                inversions.append(env.diagnostics()["inverted"])
                saturations.append(float(np.mean(abs(action) > 0.99)))
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "seed": seed,
                    "return": total,
                    "length": len(speeds),
                    "mean_speed": float(np.mean(speeds)),
                    "inverted_fraction": float(np.mean(inversions)),
                    "saturated_action_fraction": float(np.mean(saturations)),
                    "native_termination": bool(terminated),
                }
            )
    finally:
        env.close()
    summary = {
        key: float(np.mean([row[key] for row in episodes]))
        for key in (
            "return",
            "length",
            "mean_speed",
            "inverted_fraction",
            "saturated_action_fraction",
        )
    }
    summary["episodes"] = episodes
    # Declared before any fit. This qualifies intended locomotion, not just native reward.
    summary["qualified"] = bool(
        summary["mean_speed"] >= 1.0
        and summary["return"] > 0
        and summary["inverted_fraction"] <= 0.05
        and all(row["length"] == 1000 for row in episodes)
    )
    return summary

class Progress(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        env_id: str,
        condition: int,
        target: int,
        eval_interval: int,
        eval_episodes: int,
    ):
        super().__init__()
        self.run_dir, self.env_id, self.condition = run_dir, env_id, condition
        self.target = target
        self.eval_interval, self.eval_episodes = eval_interval, eval_episodes
        self.started = time.monotonic()
        self.start_steps = 0
        self.last_status = 0
        self.next_eval = eval_interval
        self.metrics = []
        self.best_score = (-1, -float("inf"))
        self.best_step = None
        progress = run_dir / "progress.json"
        if progress.exists():
            previous = json.loads(progress.read_text())
            self.metrics = previous.get("development", [])
            if self.metrics:
                best = max(self.metrics, key=lambda r: (int(r["qualified"]), r["return"]))
                self.best_score = (int(best["qualified"]), best["return"])
                self.best_step = best["steps"]

    def _on_training_start(self):
        # SB3 calls this again for the declared extension. Keep numerator and
        # elapsed time over the same process interval instead of resetting only
        # the transition count, which would understate extension throughput.
        if not getattr(self, "has_started", False):
            self.start_steps = self.num_timesteps
            self.started = time.monotonic()
            self.has_started = True
        self.next_eval = (self.num_timesteps // self.eval_interval + 1) * self.eval_interval

    def status(self, state="running"):
        elapsed = time.monotonic() - self.started
        completed = max(0, self.num_timesteps - self.start_steps)
        rate = completed / max(elapsed, 1e-6)
        return {
            "state": state,
            "pid": os.getpid(),
            "condition": self.condition,
            "env_id": self.env_id,
            "seed": 13,
            "steps": self.num_timesteps,
            "target_steps": self.target,
            "elapsed_s_this_process": elapsed,
            "transitions_per_second": rate,
            "estimated_remaining_s": (self.target - self.num_timesteps) / rate
            if rate > 0
            else None,
            "gradient_updates": self.model._n_updates,
            "updated_at": time.time(),
            "best_step": self.best_step,
            "development": self.metrics,
        }

    def checkpoint(self):
        step = int(self.num_timesteps)
        directory = self.run_dir / "checkpoints" / f"step_{step:09d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "policy.zip")
        # Publish the pointer only after all files of this numbered transaction exist.
        previous_path = self.run_dir / "resume.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
        replay_path = directory / "replay.pkl"
        replay_tmp = directory / "replay.next.pkl"
        self.model.save_replay_buffer(replay_tmp)
        replay_tmp.replace(replay_path)
        rng_path = directory / "rng.pt"
        atomic_torch_save(rng_path, {"rng": rng_state(), "steps": step})
        atomic_json(
            previous_path,
            {
                "steps": step,
                "policy": str(directory / "policy.zip"),
                "replay": str(replay_path),
                "rng": str(rng_path),
                "environment_resume": "Fresh reset; policy/optimizer/replay/RNG retained",
            },
        )
        if previous and Path(previous["replay"]) != replay_path:
            Path(previous["replay"]).unlink(missing_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_status >= 5000:
            self.last_status = self.num_timesteps
            gpu = require_cuda()
            if not all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters()):
                raise FloatingPointError("FATAL_TRAINING: nonfinite SAC parameter")
            status = self.status()
            status["gpu"] = gpu
            status["training_metrics"] = {
                key: float(value)
                for key, value in self.model.logger.name_to_value.items()
                if isinstance(value, (float, int, np.floating)) and np.isfinite(value)
            }
            atomic_json(self.run_dir / "progress.json", status)
            append_json(self.run_dir / "progress.jsonl", status)
        if self.num_timesteps >= self.next_eval:
            result = evaluate_expert(
                self.model,
                self.env_id,
                self.condition,
                [810_000 + i for i in range(self.eval_episodes)],
            )
            result["steps"] = int(self.num_timesteps)
            self.metrics.append(result)
            score = (int(result["qualified"]), result["return"])
            if score > self.best_score:
                self.best_score, self.best_step = score, int(self.num_timesteps)
                self.model.save(self.run_dir / "selected.zip")
            self.checkpoint()
            self.next_eval += self.eval_interval
            atomic_json(self.run_dir / "progress.json", self.status())
        return True

def train(args) -> None:
    seed_all(13, threads=2)
    gpu = require_cuda()
    run_dir = args.output / "experts" / args.env_id / f"condition_{args.condition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "env_id": args.env_id,
        "condition": args.condition,
        "seed": 13,
        "initial_steps": args.steps,
        "maximum_steps": args.max_steps,
        "workers": args.workers,
        "batch": 512,
        "learning_starts": args.warmup,
        "learning_rate": 3e-4,
        "network": [256, 256],
        "gamma": 0.99,
        "gradient_updates_per_environment_transition": 1 / args.workers,
        "device": "cuda",
        "gpu": gpu,
        "qualification": {
            "min_mean_speed": 1.0,
            "min_mean_return_exclusive": 0.0,
            "max_inverted_fraction": 0.05,
            "all_episodes_full_length": True,
        },
    }
    if (run_dir / "result.json").exists():
        raise FileExistsError(f"Completed expert already exists: {run_dir}")
    if (run_dir / "config.json").exists():
        prior = json.loads((run_dir / "config.json").read_text())
        for key in ("env_id", "condition", "seed", "workers", "batch", "learning_starts"):
            if config[key] != prior[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
    atomic_json(run_dir / "config.json", config)
    env_fns = [
        functools.partial(make_monitored_env, args.env_id, args.condition, 13 + i)
        for i in range(args.workers)
    ]
    vector = (
        SubprocVecEnv(env_fns, start_method="spawn") if args.workers > 1 else DummyVecEnv(env_fns)
    )
    try:
        resume_path = run_dir / "resume.json"
        if resume_path.exists():
            resume = json.loads(resume_path.read_text())
            model = SAC.load(resume["policy"], env=vector, device="cuda")
            model.load_replay_buffer(resume["replay"])
            state = torch.load(resume["rng"], weights_only=False, map_location="cpu")
            if state["steps"] != resume["steps"] or model.num_timesteps != resume["steps"]:
                raise ValueError("Resume policy/replay transaction is incomplete")
            restore_rng(state["rng"])
        else:
            model = SAC(
                "MlpPolicy",
                vector,
                learning_rate=3e-4,
                buffer_size=1_000_000,
                learning_starts=args.warmup,
                batch_size=512,
                train_freq=1,
                gradient_steps=1,
                gamma=0.99,
                policy_kwargs={"net_arch": [256, 256]},
                seed=13,
                device="cuda",
                verbose=0,
            )
        callback = Progress(
            run_dir, args.env_id, args.condition, args.steps, args.eval_interval, args.eval_episodes
        )
        callback.init_callback(model)
        callback.num_timesteps = model.num_timesteps
        if model.num_timesteps < args.steps:
            model.learn(
                args.steps - model.num_timesteps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        # One declared extension, same seed and optimizer, if still improving or unqualified.
        recent = callback.metrics[-2:]
        improving = len(recent) == 2 and recent[-1]["return"] > recent[-2]["return"] * 1.02
        unqualified = callback.best_score[0] < 1
        if args.max_steps > args.steps and (improving or unqualified):
            callback.target = args.max_steps
            atomic_json(
                run_dir / "extension.json",
                {
                    "reason": "expert_unqualified" if unqualified else "development_improving",
                    "initial_steps": args.steps,
                    "extended_steps": args.max_steps,
                },
            )
            model.learn(
                args.max_steps - model.num_timesteps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        model.save(run_dir / "final.zip")
        if not (run_dir / "selected.zip").exists():
            # Only reachable for an implementation/profile run shorter than an evaluation interval.
            model.save(run_dir / "selected.zip")
            callback.best_step = int(model.num_timesteps)
        selected = SAC.load(run_dir / "selected.zip", device="cuda", custom_objects={"seed": None})
        qualification = evaluate_expert(
            selected,
            args.env_id,
            args.condition,
            [820_000 + i for i in range(args.qualification_episodes)],
        )
        callback.checkpoint()
        atomic_json(run_dir / "progress.json", callback.status("completed"))
        atomic_json(
            run_dir / "result.json",
            {
                "state": "completed",
                "seed": 13,
                "steps": int(model.num_timesteps),
                "gradient_updates": model._n_updates,
                "selected_step": callback.best_step,
                "selected_policy": str(run_dir / "selected.zip"),
                "qualification": qualification,
                "is_profile": args.steps < 1_000_000,
            },
        )
    except BaseException as error:
        atomic_json(
            run_dir / "fatal.json",
            {
                "state": "stopped_error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "time": time.time(),
                "pid": os.getpid(),
            },
        )
        raise
    finally:
        vector.close()
