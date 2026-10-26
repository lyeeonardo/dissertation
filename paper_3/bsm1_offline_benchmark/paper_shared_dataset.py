"""Shared offline data with the Han initial state and hydraulic parameters.

Reuse only the original paper TRAIN/VAL transitions. Supplement them with four
fixed public controllers on the previously assigned20TRAIN/6VAL forcing parents.
The original paper TEST file and old common-physics trajectories are not read.
"""

from __future__ import annotations

import copy

from concurrent.futures import ProcessPoolExecutor, as_completed

import json

import multiprocessing as mp

from pathlib import Path

import time

import numpy as np

from bsm1_repro.rewards import inventory, rates

from .dataset import CollectionPolicy, forcing_for_parent

from .dataset_upgrade import UpgradeCollectionPolicy, write_json

from .environment import CanonicalBSM1Env, OBSERVATION_NAMES, STATE_NAMES, DISTURBANCE_NAMES, COMPONENT_NAMES

from .paper_protocol import PaperBSM1Env, PAPER_DECISIONS, PAPER_RATE_NAMES, PAPER_RETURN_FLOW, PAPER_WASTE_FLOW

BEHAVIORS = ("nominal_pi", "development_expert", "perturbed_expert", "boundary_holds")

SOURCE_COUNTS = {"train": 80000, "validation": 10000}

PARENT_COUNTS = {"train": 20, "validation": 6}

WEATHER_IDS = {"dry": 0, "rain": 1, "storm": 2}

RAW_FIELDS = ("observations", "next_observations", "privileged", "next_privileged", "actions",
              "previous_actions", "costs", "components", "root_rates", "next_rates", "root_inventory", "next_inventory")

def public57_from_paper(public55, current159):
    """Insert only current public DO5/NO2 instruments; retain all55 original values."""
    public, state = np.asarray(public55), np.asarray(current159)
    if public.shape[-1:] != (55,) or state.shape[-1:] != (159,) or public.shape[:-1] != state.shape[:-1]:
        raise ValueError("Aligned public55/current159 packets are required")
    if not np.isfinite(public).all() or not np.isfinite(state).all():
        raise ValueError("Public projection requires finite packets")
    # Physical145 begins with five13-species reactor states.
    return np.concatenate((public[..., :41], state[..., [4*13+7, 13+8]], public[..., 41:]), axis=-1)

class PaperPublic57Env(PaperBSM1Env):
    """Paper physics with clean public57; no privileged controller interface."""
    observation_dim = 57

    def clean_observation(self):
        return CanonicalBSM1Env.clean_observation(self)

def _load(path):
    return json.loads(Path(path).read_text())

def _npz(path, arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)

def boundary_metadata(step, sequence_end, episode_steps=PAPER_DECISIONS):
    """Keep sequence boundaries distinct from continuing-task Bellman terminals."""
    step, end = np.asarray(step), np.asarray(sequence_end, dtype=bool)
    if step.ndim != 1 or end.shape != step.shape or np.any(step < 0) or np.any(step >= episode_steps):
        raise ValueError("Aligned valid episode steps and sequence boundaries required")
    time_limit = step == episode_steps - 1
    return dict(terminated=end.copy(), physical_terminal=np.zeros(len(step), dtype=bool),
                time_limit=time_limit, split_boundary=end & ~time_limit,
                bootstrap_mask=np.ones(len(step), dtype=np.float32))

def prepare(root, paper, benchmark, seed=17):
    root, paper, benchmark = (Path(path).resolve() for path in (root, paper, benchmark))
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Prepare a fresh shared-data root; no source outputs are overwritten")
    paper_manifest = _load(paper / "data/manifest.json")
    source = _load(benchmark / "data/manifest.json")
    if paper_manifest["split_ranges"]["train"] != [0, 80000] or paper_manifest["split_ranges"]["validation"] != [80000, 90000]:
        raise ValueError("Expected the original80k/10k paper TRAIN/VAL split")
    for split in SOURCE_COUNTS:
        if not (paper / "data" / (split + ".npz")).is_file():
            raise FileNotFoundError(split)
    if paper_manifest["return_flow"] != PAPER_RETURN_FLOW or paper_manifest["waste_flow"] != PAPER_WASTE_FLOW:
        raise ValueError("Source paper hydraulics do not match the shared physical protocol")
    parents = [copy.deepcopy(parent) for parent in source["parents"] if parent["split"] in PARENT_COUNTS]
    if {split: sum(parent["split"] == split for parent in parents) for split in PARENT_COUNTS} != PARENT_COUNTS:
        raise ValueError("Use exactly the existing20TRAIN and6VAL forcing parents")
    if len({parent["parent_number"] for parent in parents}) != len(parents) or any(parent["parent_number"] < 0 for parent in parents):
        raise ValueError("Supplement parent numbers must be unique and nonnegative")
    expert = copy.deepcopy(source["expert_specification"])
    jobs = []
    for parent in parents:
        parent["initial_state"] = "Han published initial states; paper_protocol.published_initial_state"
        for behavior_index, behavior in enumerate(BEHAVIORS):
            jobs.append(dict(episode_id=1000 + len(jobs), parent_id=parent["parent_id"],
                parent_number=parent["parent_number"], split=parent["split"], weather=parent["weather"],
                behavior=behavior, behavior_seed=int(seed + 1931*parent["parent_number"] + 19*behavior_index)))
    original_episodes = [copy.deepcopy(episode) for episode in paper_manifest["episodes"] if episode["start"] < 90000]
    if max(episode["episode_id"] for episode in original_episodes) >= 1000:
        raise ValueError("Original episode IDs overlap the supplement namespace")
    manifest = dict(schema="han_shared_clean57_privileged159_v2", status="prepared", seed=seed,
        source_paper=str(paper), source_benchmark=str(benchmark),
        original_sources={split: str(paper / "data" / (split + ".npz")) for split in SOURCE_COUNTS},
        original_episodes=original_episodes, source_transitions=SOURCE_COUNTS, parents=parents, jobs=jobs,
        supplement_parents=PARENT_COUNTS, supplement_behaviors=list(BEHAVIORS), supplement_episodes=len(jobs),
        split_transitions={split: SOURCE_COUNTS[split] + PARENT_COUNTS[split]*len(BEHAVIORS)*PAPER_DECISIONS for split in SOURCE_COUNTS},
        episode_days=14., episode_steps=PAPER_DECISIONS, decision_minutes=15,
        primary="Whole14-day mean EQ+0.3OCI; no day7 reset", initial_state="Han published rounded initial states",
        return_flow=PAPER_RETURN_FLOW, waste_flow=PAPER_WASTE_FLOW, sensor_noise=False,
        expert_specification=expert, expert_selection="Previously selected v2 expert, unchanged; no retuning",
        observation_names=list(OBSERVATION_NAMES), privileged_names=list(STATE_NAMES+DISTURBANCE_NAMES),
        component_names=list(COMPONENT_NAMES), rate_names=list(PAPER_RATE_NAMES),
        public_projection="Paper55[:41] + current159[59,21] for public DO5/NO2 + Paper55[41:]",
        label_scope="Only the two explicitly measured channels are added to public inputs; current159 remains privileged",
        quality="Constant1 compatibility field for every row, not an expert-quality claim; no global episode ranking",
        parent_identity="Original parent=−(episode_id+1), supplement=existing nonnegative parent_number; only supplement has paired behaviors",
        boundaries="terminated is a sequence boundary; physical_terminal is false; time_limit marks actual episode end; split_boundary marks sliced-file end; bootstrap_mask1 uses stored physical next state",
        original_split_limitation="Original paper episode59 spans TRAIN/VAL. Preserve the supplied sequential split; do not call it parent-disjoint",
        excluded="Original paper TEST10k, all v2 physical trajectories, fresh-test parents and evaluator outputs",
        runtime="CPU collection, no automatic retry/resume; all learned methods receive the same merged TRAIN/VAL bank",
        prepared_at=time.time())
    (root / "data/episodes").mkdir(parents=True)
    write_json(root / "data/manifest.json", manifest)
    return manifest

def _episode_path(output, job):
    return Path(output) / "episodes" / f"{job['episode_id']:04d}_{job['parent_id']}_{job['behavior']}.npz"

def collect_episode(output, parent, job, expert, episode_days=14.):
    path = _episode_path(output, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = path.with_suffix(".json")
    if path.exists() or receipt.exists():
        raise FileExistsError("Existing collection attempt preserved; no automatic retry")
    started = time.time()
    write_json(receipt, dict(status="running", job=job, completed_steps=0, updated_at=time.time()))
    try:
        env = PaperPublic57Env(parent["weather"], forcing=forcing_for_parent(parent, episode_days),
                              seed=job["behavior_seed"], episode_days=episode_days)
        policy = (CollectionPolicy("nominal_pi", job["behavior_seed"]) if job["behavior"] == "nominal_pi"
                  else UpgradeCollectionPolicy(job["behavior"], expert, job["behavior_seed"]))
        rows = {key: [] for key in RAW_FIELDS}
        mapping = dict(observations="observation", next_observations="next_observation", privileged="privileged",
                       next_privileged="next_privileged", actions="action", previous_actions="previous_action",
                       costs="cost", components="components")
        for step in range(env.episode_steps):
            rate = rates(env.plant)
            rows["root_rates"].append([rate[name] for name in PAPER_RATE_NAMES])
            rows["root_inventory"].append(inventory(env.plant))
            env.step(policy.action(step, env.observation()))
            for key, source in mapping.items():
                rows[key].append(env.last_transition[source])
            rate = rates(env.plant)
            rows["next_rates"].append([rate[name] for name in PAPER_RATE_NAMES])
            rows["next_inventory"].append(inventory(env.plant))
            if (step+1) % 32 == 0:
                write_json(receipt, dict(status="running", job=job, completed_steps=step+1, updated_at=time.time()))
        arrays = {key: np.asarray(value, dtype=np.float64) for key, value in rows.items()}
        n = env.episode_steps
        arrays.update(episode_id=np.full(n, job["episode_id"], dtype=np.int32), step=np.arange(n, dtype=np.int32),
            parent_id=np.full(n, parent["parent_number"], dtype=np.int32), weather_id=np.full(n, WEATHER_IDS[parent["weather"]], dtype=np.int8),
            source_kind=np.ones(n, dtype=np.int8), quality=np.ones(n, dtype=np.float32),
            clean_observations=arrays["observations"].copy(), next_clean_observations=arrays["next_observations"].copy())
        arrays.update(boundary_metadata(arrays["step"], np.arange(n) == n-1, n))
        summary = env.summary()
        if not np.isclose(arrays["costs"].mean(), summary["economic_cost_per_day"], rtol=1e-9, atol=1e-6):
            raise ArithmeticError("Interval costs disagree with whole-episode inventory accounting")
        _npz(path, arrays)
        result = dict(status="complete", **job, transitions=n, path=str(path.resolve()),
            mean_daily_cost=float(arrays["costs"].mean()), primary_mean_daily_cost=float(arrays["costs"].mean()),
            primary="Whole episode", evaluation=summary, elapsed_seconds=time.time()-started)
        write_json(receipt, result)
        return result
    except BaseException as exc:
        write_json(receipt, dict(status="failed", job=job, failure=f"{type(exc).__name__}: {exc}", updated_at=time.time()))
        raise

def project_original(source, manifest):
    """Convert an already selected original TRAIN/VAL file, preserving physical labels."""
    arrays = {key: np.array(source[key], copy=True) for key in RAW_FIELDS}
    arrays["observations"] = public57_from_paper(arrays["observations"], arrays["privileged"])
    arrays["next_observations"] = public57_from_paper(arrays["next_observations"], arrays["next_privileged"])
    episode, step = np.asarray(source["episode_id"], dtype=np.int32), np.asarray(source["step"], dtype=np.int32)
    weather = {row["episode_id"]: WEATHER_IDS[row["weather"]] for row in manifest["original_episodes"]}
    arrays.update(episode_id=episode.copy(), step=step.copy(), parent_id=-(episode+1),
        weather_id=np.array([weather[int(value)] for value in episode], dtype=np.int8),
        source_kind=np.zeros(len(step), dtype=np.int8), quality=np.ones(len(step), dtype=np.float32),
        clean_observations=arrays["observations"].copy(), next_clean_observations=arrays["next_observations"].copy())
    arrays.update(boundary_metadata(step, source["terminated"]))
    return arrays

def assemble(root):
    root = Path(root).resolve()
    manifest = _load(root / "data/manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("Assembly requires the original prepared manifest")
    for split in SOURCE_COUNTS:
        if (root / "data" / (split + ".npz")).exists():
            raise FileExistsError("Existing assembled split preserved")
    metadata = []
    for split in SOURCE_COUNTS:
        with np.load(manifest["original_sources"][split], allow_pickle=False) as source:
            original = project_original(source, manifest)
        if len(original["costs"]) != manifest["source_transitions"][split]:
            raise ValueError("Original source split length changed")
        chunks = [original]
        for job in manifest["jobs"]:
            if job["split"] != split:
                continue
            path = _episode_path(root / "data", job)
            result = _load(path.with_suffix(".json"))
            if result.get("status") != "complete":
                raise ValueError(f"Incomplete supplement episode: {path}")
            with np.load(path, allow_pickle=False) as source:
                chunk = {key: source[key].copy() for key in source.files}
            if set(chunk) != set(original) or len(chunk["costs"]) != manifest["episode_steps"]:
                raise ValueError("Supplement schema or complete-episode length mismatch")
            chunks.append(chunk)
            metadata.append(result)
        arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in original}
        if len(arrays["costs"]) != manifest["split_transitions"][split]:
            raise ValueError("Merged split count differs from frozen manifest")
        _npz(root / "data" / (split + ".npz"), arrays)
    write_json(root / "data/episode_metadata.json", metadata)
    result = dict(manifest, status="complete", completed_supplement_episodes=len(metadata), completed_at=time.time())
    write_json(root / "data/manifest.json", result)
    write_json(root / "data/completion.json", dict(status="complete", split_transitions=result["split_transitions"],
        original_sources_unchanged=True, original_test_loaded=False, completed_at=time.time()))
    return result

def collect(root, workers=12):
    root = Path(root).resolve()
    manifest = _load(root / "data/manifest.json")
    progress_path = root / "data/collection_progress.json"
    if manifest["status"] != "prepared" or progress_path.exists():
        raise FileExistsError("Collection was already attempted; no automatic retry or resume")
    if workers < 1:
        raise ValueError("At least one CPU worker is required")
    parents = {parent["parent_id"]: parent for parent in manifest["parents"]}
    write_json(progress_path, dict(status="running", completed=0, total=len(manifest["jobs"]), updated_at=time.time()))
    completed = 0
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = [pool.submit(collect_episode, root / "data", parents[job["parent_id"]], job,
                                   manifest["expert_specification"]) for job in manifest["jobs"]]
            for future in as_completed(futures):
                future.result()
                completed += 1
                write_json(progress_path, dict(status="running", completed=completed, total=len(futures), updated_at=time.time()))
        result = assemble(root)
        write_json(progress_path, dict(status="complete", completed=completed, total=len(manifest["jobs"]), updated_at=time.time()))
        return result
    except BaseException as exc:
        write_json(progress_path, dict(status="failed", completed=completed, failure=f"{type(exc).__name__}: {exc}", updated_at=time.time()))
        raise
