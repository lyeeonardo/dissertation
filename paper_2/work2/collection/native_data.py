"""Native corpus primitives for the final HalfCheetah study only."""
from pathlib import Path
import json
import h5py
import numpy as np
from work2.collection.common import save as atomic_json
SPLITS = ('train', 'development', 'calibration', 'test')

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
