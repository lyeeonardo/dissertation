from __future__ import annotations


import json

import time

from contextlib import ExitStack

from pathlib import Path

import h5py

import numpy as np

from .common import atomic_json

from .evaluation import atomic_npz

from .replay import trace_histories

KIND_QUOTAS = {
    "calibration": {"expert": 16, "recovery": 8, "paired": 8},
    "development": {"expert": 8, "recovery": 4, "paired": 4},
}

def read(path):
    return json.loads(Path(path).read_text())

def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen CPU pilot declaration differs: {path}")
    else:
        atomic_json(path, value)

def select_transitions(rows, split, index_path):
    """Fixed composition; spread each kind over parents before reusing one."""
    selected = []
    for condition in range(8):
        for kind, count in KIND_QUOTAS[split].items():
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        13,
                        153001,
                        condition,
                        list(KIND_QUOTAS).index(split),
                        list(KIND_QUOTAS[split]).index(kind),
                    ]
                )
            )
            pool = [
                (i, row)
                for i, row in enumerate(rows)
                if row["split"] == split
                and row["condition"] == condition
                and row["kind"] == kind
                and row["length"] > 0
            ]
            parents = sorted({row["parent_id"] for _, row in pool})
            if not parents:
                raise ValueError(f"No existing support for {split}/{condition}/{kind}")
            parents = [parents[i] for i in rng.permutation(len(parents))]
            parent_rows = {
                parent: [(i, row) for i, row in pool if row["parent_id"] == parent]
                for parent in parents
            }
            used = set()
            for position in range(count):
                parent = parents[position % len(parents)]
                available = [
                    (i, row)
                    for i, row in parent_rows[parent]
                    if sum(a == i for a, _ in used) < row["length"]
                ]
                if not available:
                    raise ValueError("Insufficient unique transitions in selected parent")
                i, row = available[int(rng.integers(len(available)))]
                steps = [s for s in range(row["length"]) if (i, s) not in used]
                step = int(steps[int(rng.integers(len(steps)))])
                used.add((i, step))
                selected.append(
                    dict(
                        transition_id=f"{split}:row{i}:step{step}",
                        source_type="indexed_hdf5",
                        role=split,
                        index_path=str(index_path),
                        row_index=i,
                        step=step,
                        path=row["path"],
                        episode=row["episode"],
                        length=row["length"],
                        kind=kind,
                        parent_id=parent,
                        condition_evaluator_only=condition,
                    )
                )
    return selected

def _hdf_transition(ref, group):
    step = ref["step"]
    if not 0 <= step < ref["length"] or group["observations"].shape[0] != ref["length"] + 1:
        raise ValueError("Frozen transition index no longer matches native source")
    positions = np.arange(step - 7, step + 1)
    observations = np.zeros((8, 17), np.float32)
    actions = np.zeros((7, 6), np.float32)
    rewards = np.zeros(7, np.float32)
    mask = np.zeros(8, np.float32)
    prefix = "prefix_observations" in group
    for i, at in enumerate(positions):
        if at >= 0:
            observations[i] = group["observations"][at]
            mask[i] = 1
        elif prefix:
            # Retained prefix includes the current branch-root observation as its final row.
            index = len(group["prefix_observations"]) - 1 + at
            if index < 0:
                raise ValueError("Retained branch prefix is too short")
            observations[i] = group["prefix_observations"][index]
            mask[i] = group["prefix_mask"][index]
        if i < 7:
            if at >= 0:
                actions[i], rewards[i] = group["actions"][at], group["rewards"][at]
            elif prefix:
                index = len(group["prefix_actions"]) + at
                actions[i], rewards[i] = (
                    group["prefix_actions"][index],
                    group["prefix_rewards"][index],
                )
    return dict(
        history=dict(observations=observations, actions=actions, rewards=rewards, mask=mask),
        actions=group["actions"][step],
        next_observations=group["observations"][step + 1],
        rewards=float(group["rewards"][step]),
        labels_evaluator_only=ref["condition_evaluator_only"],
    )

def read_batch(references):
    """Return native physical variables; labels are separate from the public history."""
    rows = []
    with ExitStack() as stack:
        handles = {}
        for ref in references:
            key = (ref["source_type"], ref["path"])
            if key not in handles:
                handles[key] = stack.enter_context(
                    h5py.File(ref["path"], "r")
                    if ref["source_type"] == "indexed_hdf5"
                    else np.load(ref["path"], allow_pickle=False)
                )
            source = handles[key]
            if ref["source_type"] == "indexed_hdf5":
                row = _hdf_transition(ref, source[ref["episode"]])
            elif ref["source_type"] == "replay_npz":
                step = ref["step"]
                row = dict(
                    history={k: v[0] for k, v in trace_histories(source, [step]).items()},
                    actions=source["actions"][step],
                    next_observations=source["observations"][step + 1],
                    rewards=source["rewards"][step],
                    labels_evaluator_only=source["conditions"][step],
                )
            else:
                raise ValueError("Unknown frozen transition source")
            rows.append(row)
    if not rows:
        raise ValueError("Cannot load an empty transition batch")
    return {
        k: (
            {name: np.stack([r[k][name] for r in rows]) for name in rows[0][k]}
            if k == "history"
            else np.asarray([r[k] for r in rows])
        )
        for k in rows[0]
    }

def _tensor_batch(batch):
    import torch

    return {
        key: (
            {name: torch.tensor(value, dtype=torch.float32) for name, value in item.items()}
            if key == "history"
            else torch.tensor(
                item, dtype=(torch.int64 if key == "labels_evaluator_only" else torch.float32)
            )
        )
        for key, item in batch.items()
    }

def _count_delta(before, after):
    return {key: int(after[key] - before[key]) for key in ("forward_calls", "forward_examples")}

def _cache_group(output, name, references, engine, *, batch_size=32, check_budget=None):
    """Cache each method independently, preserving completed neural work on failure."""
    import torch

    directory = Path(output) / "experiment_a" / "cache" / name
    directory.mkdir(parents=True, exist_ok=True)
    freeze_json(
        directory / "manifest.json",
        dict(
            references=references,
            batch_size=batch_size,
            probes=[0.2, 0.4, 0.6, 0.8],
            gaussian_particles=16,
            device="cpu",
            dtype="float32",
            evidence=["observations", "rewards"],
        ),
    )
    groups = {"denoising": [], "gaussian": []}
    timings = {"denoising": [], "gaussian": []}
    for start in range(0, len(references), batch_size):
        refs = references[start : start + batch_size]
        batch = None
        for method in ("denoising", "gaussian"):
            path = directory / f"batch_{start:04d}_{method}.npz"
            record_path = path.with_suffix(".json")
            if path.exists() != record_path.exists():
                raise RuntimeError(f"Incomplete cache commit requires explicit recovery: {path}")
            if path.exists():
                metadata = read(record_path)
                if metadata["transition_ids"] != [r["transition_id"] for r in refs]:
                    raise ValueError("Cached transition identities differ")
                with np.load(path, allow_pickle=False) as saved:
                    values = {k: saved[k].copy() for k in saved.files}
            else:
                if check_budget is not None:
                    check_budget()
                if batch is None:
                    batch = _tensor_batch(read_batch(refs))
                # Stable stage/batch/method seed; batching declared before querying.
                domain = {"calibration": 0, "development": 1}.get(name, 2)
                window = next(
                    (
                        i
                        for i, value in enumerate(
                            ("stationary", "gravity", "sliding_friction", "joint_range")
                        )
                        if name == "replay_" + value
                    ),
                    0,
                )
                seed = int(
                    np.random.SeedSequence(
                        [13, 154001, domain, window, start, int(method == "gaussian")]
                    ).generate_state(1)[0]
                )
                generator = torch.Generator(device="cpu").manual_seed(seed)
                before = engine.stats.snapshot()
                started = time.perf_counter()
                if method == "denoising":
                    errors = engine.denoising_errors(
                        batch["history"],
                        batch["actions"],
                        batch["next_observations"],
                        batch["rewards"],
                        generator=generator,
                    )
                    values = dict(
                        errors=errors.numpy(), labels=batch["labels_evaluator_only"].numpy()
                    )
                else:
                    moments = engine.gaussian_moments(
                        batch["history"], batch["actions"], particles=16, generator=generator
                    )
                    observed = engine.normalize_outcomes(
                        batch["history"], batch["next_observations"], batch["rewards"]
                    )
                    values = dict(
                        mean=moments["mean"].numpy(),
                        variance=moments["variance"].numpy(),
                        observed=observed.numpy(),
                        labels=batch["labels_evaluator_only"].numpy(),
                    )
                seconds = time.perf_counter() - started
                after = engine.stats.snapshot()
                if not all(np.isfinite(value).all() for value in values.values()):
                    raise FloatingPointError("Nonfinite CPU evidence cache")
                metadata = dict(
                    state="completed_cpu_evidence_batch",
                    transition_ids=[r["transition_id"] for r in refs],
                    method=method,
                    seconds=seconds,
                    neural_queries=_count_delta(before, after),
                    seed=seed,
                    scope=(
                        "Complete evidence computation including query preparation "
                        "and outcome normalization; excludes disk I/O"
                    ),
                )
                atomic_npz(path, values)
                atomic_json(record_path, metadata)
            groups[method].append(values)
            timings[method].append(metadata)
    arrays = {
        method: {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        for method, parts in groups.items()
    }
    totals = {
        method: dict(
            seconds=sum(r["seconds"] for r in records),
            forward_calls=sum(r["neural_queries"]["forward_calls"] for r in records),
            forward_examples=sum(r["neural_queries"]["forward_examples"] for r in records),
            transitions=len(references),
        )
        for method, records in timings.items()
    }
    return arrays, totals

def _evidence(arrays, calibration):
    import torch

    from .cpu_denoising_evidence import denoising_log_evidence, gaussian_log_evidence

    started = time.perf_counter()
    denoising = denoising_log_evidence(
        torch.tensor(arrays["denoising"]["errors"]), calibration["denoising"]["temperature"]
    )
    denoising_s = time.perf_counter() - started
    started = time.perf_counter()
    gaussian = gaussian_log_evidence(
        torch.tensor(arrays["gaussian"]["observed"]),
        {k: torch.tensor(arrays["gaussian"][k]) for k in ("mean", "variance")},
        calibration["gaussian"],
    )
    gaussian_s = time.perf_counter() - started
    return {"denoising": denoising, "gaussian": gaussian}, {
        "denoising": denoising_s,
        "gaussian": gaussian_s,
    }
