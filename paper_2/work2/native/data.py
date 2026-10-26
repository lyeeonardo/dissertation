from __future__ import annotations


import json

import time

from pathlib import Path

import h5py

import numpy as np

from stable_baselines3 import SAC

from .common import atomic_json, require_cuda, seed_all

from .environment import PaperEnv

SPLITS = ("train", "development", "calibration", "test")

def split_for_episode(index: int) -> str:
    slot = index % 10
    return SPLITS[0 if slot < 7 else slot - 6]

def public_history(episode, index: int, length=8) -> dict:
    """Past transition t ends at observation t+1; branch prefixes preserve ancestry."""
    observations = episode["observations"][: index + 1]
    actions = episode["actions"][:index]
    rewards = episode["rewards"][:index]
    if "prefix_observations" in episode:
        # Prefix's final observation is the branch root, already in observations.
        prefix_mask = episode["prefix_mask"][:].astype(bool)
        prefix_obs = episode["prefix_observations"][:][prefix_mask]
        transition_mask = prefix_mask[:-1] & prefix_mask[1:]
        observations = np.concatenate((prefix_obs[:-1], observations))
        actions = np.concatenate((episode["prefix_actions"][:][transition_mask], actions))
        rewards = np.concatenate((episode["prefix_rewards"][:][transition_mask], rewards))
    count = min(length, len(observations))
    result = {
        "observations": np.zeros((length, 17), np.float32),
        "actions": np.zeros((length - 1, 6), np.float32),
        "rewards": np.zeros(length - 1, np.float32),
        "mask": np.zeros(length, np.float32),
    }
    result["observations"][-count:] = observations[-count:]
    result["mask"][-count:] = 1
    if count > 1:
        result["actions"][-(count - 1) :] = actions[-(count - 1) :]
        result["rewards"][-(count - 1) :] = rewards[-(count - 1) :]
    return result

def write_episode(handle, name: str, arrays: dict, metadata: dict) -> None:
    if name in handle:
        del handle[name]
    group = handle.create_group(name)
    for key, value in arrays.items():
        value = np.asarray(value)
        group.create_dataset(key, data=value, compression="lzf" if value.ndim else None)
    for key, value in metadata.items():
        group.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    group.attrs["complete"] = True
    handle.flush()

def expert_result(output: Path, env_id: str, condition: int) -> dict:
    if env_id != "HalfCheetah-v5": raise ValueError("Only the dissertation HalfCheetah task is archived")
    directory = output / "experts" / env_id / f"condition_{condition}"
    return json.loads((directory / "result.json").read_text())

def expert_path(output: Path, env_id: str, condition: int) -> Path:
    result = expert_result(output, env_id, condition)
    if result["is_profile"] or result["steps"] < 1_000_000:
        raise ValueError(f"Expert {condition} has not received the declared full budget")
    if not result["qualification"]["qualified"]:
        raise ValueError(f"EXPERT_CAPABILITY_INCOMPLETE: condition {condition}")
    return Path(result["selected_policy"])

def expert_paths(output: Path, env_id: str) -> list[Path]:
    return [expert_path(output, env_id, condition) for condition in range(8)]

def collect_rollout(
    env,
    models,
    condition,
    episode_index,
    kind,
    *,
    max_length=None,
    reset_seed=None,
    random_seed=None,
):
    recovery = kind == "recovery"
    seed = (
        1_000_000 + condition * 10_000 + episode_index + (200_000 if recovery else 0)
        if reset_seed is None
        else reset_seed
    )
    generator = np.random.default_rng(
        np.random.SeedSequence([13, condition, episode_index, 1 if recovery else 0])
        if random_seed is None
        else random_seed
    )
    length = 600 if recovery else 1000
    if max_length is not None:
        if max_length < 1:
            raise ValueError("Collection must request at least one real transition")
        length = min(length, max_length)
    observation, _ = env.reset(seed=seed)
    snapshot = env.snapshot()
    observations, integrations = [observation.copy()], [snapshot["integration"]]
    actions, rewards, terminateds, truncateds, eligible, interventions = [], [], [], [], [], []
    speeds, inverted = [], []
    # Fixed-condition interventions only: no hidden parameter changes in trunk data.
    periods = []
    if recovery:
        for start in (int(generator.integers(100, 151)), int(generator.integers(375, 426))):
            periods.append((start, start + int(generator.integers(5, 21))))
    wrong = condition ^ (1 << int(generator.integers(0, 3)))
    for step in range(length):
        intervene = any(start <= step < end for start, end in periods)
        # Alternate wrong-label behavior and smooth bounded noise across parent episodes.
        use_wrong = intervene and episode_index % 2 == 0
        source = models[wrong] if use_wrong else models[condition]
        action, _ = source.predict(observation, deterministic=True)
        if intervene and not use_wrong:
            action = np.clip(0.3 * action + generator.normal(0, 0.5, 6), -1, 1).astype(np.float32)
        observation, reward, terminal, truncated, info = env.step(action)
        diagnostics = env.diagnostics()
        speed = float(info.get("x_velocity", 0))
        observations.append(observation.copy())
        integrations.append(env.snapshot()["integration"])
        actions.append(action.copy())
        rewards.append(reward)
        terminateds.append(terminal)
        truncateds.append(truncated)
        interventions.append(intervene)
        eligible.append(not intervene and not diagnostics["inverted"] and speed >= 0.5)
        speeds.append(speed)
        inverted.append(diagnostics["inverted"])
        if terminal or truncated:
            break
    arrays = {
        "observations": np.asarray(observations, np.float32),
        "actions": np.asarray(actions, np.float32),
        "rewards": np.asarray(rewards, np.float32),
        "terminated": np.asarray(terminateds, bool),
        "truncated": np.asarray(truncateds, bool),
        "proposal_eligible": np.asarray(eligible, bool),
        "intervention": np.asarray(interventions, bool),
        "integration": np.asarray(integrations, np.float64),
        "speed": np.asarray(speeds, np.float32),
        "inverted": np.asarray(inverted, bool),
    }
    metadata = {
        "condition": condition,
        "kind": kind,
        "episode_index": episode_index,
        "parent_id": condition * 1_000_000 + (100_000 if recovery else 0) + episode_index,
        "split": split_for_episode(episode_index),
        "reset_seed": seed,
        "rng": snapshot["rng"],
        "source_condition": condition,
        "length": len(actions),
        "requested_length": length,
        "intervention_windows": periods,
        "wrong_condition": wrong,
        "action_noise_seed": [13, condition, episode_index, int(recovery)]
        if random_seed is None
        else random_seed,
        "initial_elapsed": 0,
        "collection_cutoff": not (terminateds[-1] or truncateds[-1]),
        "proposal_rule": "expert-active, noninverted, instantaneous forward speed >= 0.5 m/s",
    }
    return arrays, metadata

def collect_trunks(
    output: Path, env_id="HalfCheetah-v5", conditions=None, kinds=("expert", "recovery")
):
    seed_all(13)
    require_cuda()
    conditions = list(range(8)) if conditions is None else list(conditions)
    if (
        not conditions
        or not set(conditions) <= set(range(8))
        or len(set(conditions)) != len(conditions)
    ):
        raise ValueError("Select distinct physical conditions from 0 through 7")
    if not kinds or not set(kinds) <= {"expert", "recovery"}:
        raise ValueError("Collect expert and/or recovery parents")
    needed = set(conditions)
    if "recovery" in kinds:
        needed.update(condition ^ bit for condition in conditions for bit in (1, 2, 4))
    models = {
        condition: SAC.load(
            expert_path(output, env_id, condition), device="cuda", custom_objects={"seed": None}
        )
        for condition in sorted(needed)
    }
    directory = output / "data" / env_id
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for condition in conditions:
        env = PaperEnv(env_id, condition)
        try:
            with h5py.File(directory / f"cell_{condition}.h5", "a") as handle:
                for kind, count in (("expert", 80), ("recovery", 20)):
                    if kind not in kinds:
                        continue
                    for index in range(count):
                        name = f"{kind}/{index:04d}"
                        if name in handle and handle[name].attrs.get("complete", False):
                            continue
                        arrays, metadata = collect_rollout(env, models, condition, index, kind)
                        write_episode(handle, name, arrays, metadata)
                        if len(arrays["actions"]) != (600 if kind == "recovery" else 1000):
                            # Keep the exact native parent length.
                            raise ValueError(
                                "EARLY_TERMINAL_COLLECTION: retain prefix and add balanced parents"
                            )
                        atomic_json(
                            output / "data_progress.json",
                            {
                                "state": "collecting_trunks",
                                "condition": condition,
                                "kind": kind,
                                "episode": index,
                                "updated_at": time.time(),
                                "started_at": started,
                            },
                        )
                        if index % 10 == 0:
                            require_cuda()
        finally:
            env.close()

    atomic_json(
        output / "data_progress.json",
        {
            "state": "trunks_completed",
            "conditions": conditions,
            "kinds": list(kinds),
            "updated_at": time.time(),
            "started_at": started,
            "remaining": "Remaining cells/kinds, paired branches and full corpus validation",
        },
    )

def anchor_table(output: Path, env_id="HalfCheetah-v5") -> list[dict]:
    """250 parents, stratified 175/25/25/25; selected without outcome-based filtering."""
    directory = output / "data" / env_id
    pools = {split: [] for split in SPLITS}
    for condition in range(8):
        with h5py.File(directory / f"cell_{condition}.h5", "r") as handle:
            for kind, count in (("expert", 80), ("recovery", 20)):
                for index in range(count):
                    group = handle[f"{kind}/{index:04d}"]
                    pools[group.attrs["split"]].append(
                        {
                            "cell": condition,
                            "episode": group.name,
                            "parent_id": int(group.attrs["parent_id"]),
                            "split": group.attrs["split"],
                            "length": int(group.attrs["length"]),
                        }
                    )
    rng = np.random.default_rng(230_013)
    anchors = []
    for split, count in zip(SPLITS, (175, 25, 25, 25), strict=True):
        pool = pools[split]
        chosen = rng.choice(len(pool), size=count, replace=False)
        for selected in chosen:
            row = dict(pool[selected])
            row["step"] = int(rng.integers(16, row["length"] - 16))
            row["anchor_id"] = len(anchors)
            anchors.append(row)
    atomic_json(output / "splits" / "paired_anchors.json", {"anchors": anchors})
    return anchors

def branch_action_chunks(model, wrong_model, env, state, observation, seed, horizon=8):
    rng = np.random.default_rng(seed)
    chunks = []
    for teacher in (model, wrong_model):
        obs = env.restore(state)
        actions = []
        for _ in range(horizon):
            action, _ = teacher.predict(obs, deterministic=True)
            obs, _, terminal, truncated, _ = env.step(action)
            actions.append(action)
            if terminal or truncated:
                break
        while len(actions) < horizon:
            actions.append(np.zeros(6, np.float32))
        chunks.append(np.asarray(actions, np.float32))
    noise = rng.normal(0, 0.5, (horizon, 6))
    for t in range(1, horizon):
        noise[t] = 0.7 * noise[t - 1] + 0.3 * noise[t]
    chunks.append(np.clip(chunks[0] + noise, -1, 1).astype(np.float32))
    probe = np.zeros((horizon, 6), np.float32)
    probe[:, int(rng.integers(0, 6))] = rng.choice([-0.35, 0.35])
    chunks.append(probe)
    return chunks

def collect_branches(output: Path, env_id="HalfCheetah-v5"):
    require_cuda()
    paths = expert_paths(output, env_id)
    models = [SAC.load(path, device="cuda", custom_objects={"seed": None}) for path in paths]
    anchors = anchor_table(output, env_id)
    directory = output / "data" / env_id
    handles = [h5py.File(directory / f"cell_{i}.h5", "a") for i in range(8)]
    env = PaperEnv(env_id)
    env.reset(seed=13)
    try:
        for anchor in anchors:
            source = handles[anchor["cell"]][anchor["episode"]]
            index = anchor["step"]
            # Retain a longer public prefix for DADP's lagged 16-step encoder and
            # the declared history sensitivity checks. Primary models still use 8.
            history = public_history(source, index, length=33)
            state = {
                "condition": anchor["cell"],
                "elapsed": index,
                "integration": source["integration"][index],
                "rng": json.loads(source.attrs["rng"]),
            }
            chunks = branch_action_chunks(
                models[anchor["cell"]],
                models[anchor["cell"] ^ 7],
                env,
                state,
                history["observations"][-1],
                240_000 + anchor["anchor_id"],
            )
            for chunk_id, chunk in enumerate(chunks):
                for target in range(8):
                    name = f"paired/{anchor['anchor_id']:04d}_{chunk_id}"
                    if name in handles[target] and handles[target][name].attrs.get(
                        "complete", False
                    ):
                        continue
                    obs = env.restore(state, target)
                    observations, integrations = [obs], [env.snapshot()["integration"]]
                    rewards, terminals, truncateds = [], [], []
                    for action in chunk:
                        obs, reward, terminal, truncated, _ = env.step(action)
                        observations.append(obs)
                        integrations.append(env.snapshot()["integration"])
                        rewards.append(reward)
                        terminals.append(terminal)
                        truncateds.append(truncated)
                        if terminal or truncated:
                            break
                    n = len(rewards)
                    arrays = {
                        "observations": np.asarray(observations, np.float32),
                        "actions": chunk[:n],
                        "rewards": np.asarray(rewards, np.float32),
                        "terminated": np.asarray(terminals, bool),
                        "truncated": np.asarray(truncateds, bool),
                        "proposal_eligible": np.zeros(n, bool),
                        "integration": np.asarray(integrations),
                        **{f"prefix_{key}": value for key, value in history.items()},
                    }
                    metadata = {
                        "condition": target,
                        "source_condition": anchor["cell"],
                        "kind": "paired",
                        "parent_id": anchor["parent_id"],
                        "split": anchor["split"],
                        "anchor_id": anchor["anchor_id"],
                        "chunk_id": chunk_id,
                        "length": n,
                        "initial_elapsed": index,
                        "rng": state["rng"],
                        "collection_cutoff": not (terminals[-1] or truncateds[-1]),
                    }
                    write_episode(handles[target], name, arrays, metadata)
                    if n < 8:
                        raise ValueError(
                            "EARLY_TERMINAL_BRANCH: prefix retained; balanced top-up needed"
                        )
            if anchor["anchor_id"] % 10 == 0:
                require_cuda()
                atomic_json(
                    output / "data_progress.json",
                    {
                        "state": "collecting_branches",
                        "anchor": anchor["anchor_id"],
                        "total_anchors": len(anchors),
                        "updated_at": time.time(),
                    },
                )
    finally:
        env.close()
        for handle in handles:
            handle.close()

def finalize_dataset(output: Path, env_id="HalfCheetah-v5"):
    directory = output / "data" / env_id
    index = {"R": []}
    sums = {"observation": np.zeros(17), "delta": np.zeros(17), "reward": np.zeros(1)}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    counts = {key: 0 for key in sums}
    ancestry = {}
    for condition in range(8):
        path = directory / f"cell_{condition}.h5"
        with h5py.File(path, "r") as handle:
            for kind in ("expert", "recovery", "paired"):
                for key in sorted(handle[kind]):
                    group = handle[kind][key]
                    if not group.attrs.get("complete", False):
                        raise ValueError(f"Incomplete episode: {path}:{group.name}")
                    n, parent = int(group.attrs["length"]), int(group.attrs["parent_id"])
                    split = group.attrs["split"]
                    if parent in ancestry and ancestry[parent] != split:
                        raise ValueError("Parent ancestry crosses data splits")
                    ancestry[parent] = split
                    for field in ("observations", "actions", "rewards"):
                        if not np.isfinite(group[field][:]).all():
                            raise ValueError(f"Nonfinite dataset field: {field}")
                    row = {
                        "path": str(path.resolve()),
                        "episode": group.name,
                        "length": n,
                        "condition": condition,
                        "parent_id": parent,
                        "split": split,
                        "kind": kind,
                        "proposal_eligible": int(group["proposal_eligible"][:].sum()),
                    }
                    if kind != "expert" or int(key) < 80:
                        index["R"].append(row)
                    if kind == "expert" and int(key) < 80 and split == "train":
                        values = {
                            "observation": group["observations"][:-1],
                            "delta": np.diff(group["observations"][:], axis=0),
                            "reward": group["rewards"][:].reshape(-1, 1),
                        }
                        for field, value in values.items():
                            value = value.astype(np.float64)
                            sums[field] += value.sum(0)
                            squares[field] += np.square(value).sum(0)
                            counts[field] += len(value)
    normalization = {}
    for key in sums:
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(squares[key] / counts[key] - mean * mean, 1e-6))
        normalization[key + "_mean"] = mean.item() if key == "reward" else mean
        normalization[key + "_std"] = std.item() if key == "reward" else std
    totals = {}
    for arm, rows in index.items():
        total = sum(row["length"] for row in rows)
        if total != 800_000:
            raise ValueError(f"{arm}: expected 800,000 valid transitions, got {total}")
        totals[arm] = {
            "transitions": total,
            "splits": {s: sum(r["length"] for r in rows if r["split"] == s) for s in SPLITS},
            "conditions": {
                str(c): sum(r["length"] for r in rows if r["condition"] == c) for c in range(8)
            },
            "proposal_eligible": sum(row["proposal_eligible"] for row in rows),
        }
        if set(totals[arm]["conditions"].values()) != {100_000}:
            raise ValueError("Unbalanced condition counts")
        atomic_json(output / "splits" / f"{arm}_index.json", {"episodes": rows})
    atomic_json(output / "normalization.json", normalization)
    atomic_json(
        output / "data_summary.json",
        {
            "state": "complete",
            "env_id": env_id,
            "arms": totals,
            "normalization_source": "Shared stationary TRAIN parents; first 80 experts/cell",
            "proposal_rule": "expert-active, noninverted, instantaneous forward speed >= 0.5 m/s",
            "ancestry_groups": len(ancestry),
            "seed": 13,
        },
    )
    atomic_json(output / "data_progress.json", {"state": "completed", "updated_at": time.time()})
