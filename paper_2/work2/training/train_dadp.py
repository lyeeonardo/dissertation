"""Fresh native DADP fits on the isolated 3x-gravity R corpus.

Without --execute this prints the declared recipe and does no model/data work.
The selected.pt/best.pt checkpoints retain the native load_representation and
load_policy schema. final.pt preserves the terminal, potentially non-best fit.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]

from .plateau import Plateau


def recipe(stage):
    return {
        "experiment": "halfcheetah_gravity3x_20261016",
        "stage": stage,
        "arm": "R",
        "seed": 13,
        "fresh_initialization": True,
        "gravity_levels": [1.0, 3.0],
        "history": 16,
        "future_actions": 4,
        "batch_size": 128 if stage == "representation" else 256,
        "hard_cap_updates": 100_000 if stage == "representation" else 300_000,
        "check_every_updates": 2_000,
        "periodic_checkpoint_updates": 10_000,
        "validation_split": "development",
        "validation_seed": 813_013,
        "validation_batches": 8,
        "monitor": "forward_plus_inverse" if stage == "representation" else "ema_diffusion_loss",
        "plateau": {"minimum_updates": 10_000, "patience": 5, "relative_improvement": 0.005},
        "representation_precision": {"precision": "float32", "attention_backend": "math", "tf32": False},
        "policy_precision": "native BF16 forward / FP32 residual loss",
        "policy_ema": 0.9999,
        "selection": "lowest raw fixed-DEV monitored loss; not control qualification",
    }


@contextmanager
def training_lock(root):
    """Coordinate with the other model-family fits, without a persistent lease."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another model-family fit owns training.lock") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} stage=DADP\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def data_identity(root):
    """Check parent split metadata only; never load CAL/TEST observations."""
    summary_path = root / "data_summary.json"
    completion_path = root / "data_collection_result.json"
    summary = json.loads(summary_path.read_text())
    completion = json.loads(completion_path.read_text())
    if (summary.get("state") != "complete" or
            summary.get("env_id") != "HalfCheetah-v5" or
            summary.get("gravity_levels") != [1.0, 3.0] or
            set(summary.get("arms", {})) != {"R"} or
            summary["arms"]["R"].get("transitions") != 800_000 or
            summary.get("normalization_transitions") != 448_000 or
            summary.get("old_high_gravity_transitions_relabelled") != 0 or
            summary.get("train_dev_cal_test_parent_disjoint") is not True or
            completion.get("state") != "completed" or
            completion.get("dataset_transitions") != 800_000):
        raise ValueError("The completed isolated 1x/3x R corpus is required")
    index_path = root / "splits/R_index.json"
    rows = json.loads(index_path.read_text())["episodes"]
    parents = {}
    for row in rows:
        parent, split = int(row["parent_id"]), row["split"]
        if parent in parents and parents[parent] != split:
            raise ValueError(f"Parent {parent} crosses ancestry splits")
        parents[parent] = split
    if not {"train", "development"}.issubset(set(parents.values())):
        raise ValueError("TRAIN and DEV parents are required")
    stat = index_path.stat()
    return {
        "root": str(root),
        "index_path": str(index_path),
        "index_size_bytes": stat.st_size,
        "index_mtime_ns": stat.st_mtime_ns,
        "parent_splits_disjoint": True,
        "normalization": json.loads((root / "normalization.json").read_text()),
        "data_summary": summary,
        "collection_completion": completion,
    }


def fixed_development(stage, model, representation, sampler, batch_size, config):
    """Native objectives on identical balanced anchors and Torch noise each time.

    Policy validation uses EMA. Representation preserves the repaired FP32/math
    context through native stage_loss; only forward+inverse controls stopping.
    All RNG objects, module modes and the online/EMA pointer are restored.
    """
    import numpy as np
    import torch
    from work2.native.dadp_training import stage_loss
    from work2.native.windows import to_device

    device = next(representation.parameters()).device
    devices = [] if device.type == "cpu" else [device.index if device.index is not None else torch.cuda.current_device()]
    old_sampler_rng = sampler.rng
    old_base_rng = sampler.base.rng
    modules = list(representation.modules())
    if stage == "policy":
        modules += list(model.model.modules()) + list(model.model_ema.modules())
    modes = [(module, module.training) for module in modules]
    original_model = model.model if stage == "policy" else None
    values = []
    try:
        sampler.rng = np.random.default_rng(config["validation_seed"])
        sampler.base.rng = sampler.rng
        representation.eval()
        if stage == "policy":
            model.model = model.model_ema
            model.model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            # Native MixedDDIM draws its own times/noise, so reseed the scoped
            # global generator rather than relying on its unused eps argument.
            torch.manual_seed(config["validation_seed"])
            for _ in range(config["validation_batches"]):
                batch = to_device(getattr(sampler, stage)(batch_size), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=stage == "policy"):
                    total, components = stage_loss(stage, model, representation, batch)
                row = {"total_loss": float(total)}
                row.update({key: float(value) for key, value in components.items()})
                row["monitored_loss"] = row["forward"] + row["inverse"] if stage == "representation" else row["total_loss"]
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError("FATAL_DADP: nonfinite fixed-DEV loss")
                values.append(row)
    finally:
        sampler.rng = old_sampler_rng
        sampler.base.rng = old_base_rng
        if stage == "policy":
            model.model = original_model
        for module, mode in modes:
            module.training = mode
    return {key: float(np.mean([row[key] for row in values])) for key in values[0]}


def train(root, stage):
    import torch
    from work2.native.common import (
        append_json, atomic_json, atomic_torch_save, require_cuda, restore_rng,
        rng_state, seed_all,
    )
    from work2.native.dadp_adapter import make_planner, make_representation
    from work2.native.dadp_training import DADPWindows, load_representation, optimization_step
    from work2.native.windows import to_device

    config = recipe(stage)
    directory = root / "models/dadp_R" / stage
    completed_path = directory / "completed.json"
    if completed_path.exists():
        result = json.loads(completed_path.read_text())
        if result["recipe"] != config:
            raise ValueError("Completed DADP recipe differs; preserve it in its own root")
        print(json.dumps({"state": "already_completed_not_restarted", **result}, indent=2))
        return
    identity = data_identity(root)
    if (directory / "config.json").exists() and not (directory / "resume.pt").exists():
        raise RuntimeError("Earlier DADP attempt has no committed checkpoint; no silent restart")
    directory.mkdir(parents=True, exist_ok=True)
    gpu = require_cuda()  # One real startup probe, not a repeating observer.
    seed_all(config["seed"])
    sampler = DADPWindows(root, "R", "train", config["seed"])
    development = DADPWindows(root, "R", "development", config["validation_seed"], sampler.value_bounds)
    rep_path = root / "models/dadp_R/representation/selected.pt"
    representation_selection = None
    if stage == "representation":
        representation = make_representation()
        model, optimizer = representation, representation.optimizer
    else:
        rep_completed = root / "models/dadp_R/representation/completed.json"
        if not rep_completed.exists():
            raise ValueError("The new representation stage must finish before fitting policy")
        representation, rep_state = load_representation(rep_path)
        if (rep_state["recipe"] != recipe("representation") or
                rep_state["data_identity"] != identity or
                rep_state["normalization"] != sampler.normalization):
            raise ValueError("Selected DADP representation does not match this 3x dataset/recipe")
        representation_selection = {"path": str(rep_path), "updates": rep_state["updates"], "loss": rep_state["best_dev_loss"]}
        representation.requires_grad_(False).eval()
        model = make_planner()
        optimizer = model.optimizer
        del rep_state

    completed, records, elapsed_before = 0, [], 0.0
    plateau = Plateau(**config["plateau"])
    resume_path = directory / "resume.json"
    if (directory / "resume.pt").exists():
        # The binary commit is authoritative even if an interrupt prevented the
        # first small JSON receipt from being created.
        if resume_path.exists() and json.loads(resume_path.read_text())["checkpoint"] != str(directory / "resume.pt"):
            raise ValueError("Resume receipt points outside the rolling stage checkpoint")
        state = torch.load(directory / "resume.pt", map_location="cuda", weights_only=False)
        for key, expected in (("recipe", config), ("data_identity", identity),
                              ("normalization", sampler.normalization),
                              ("value_bounds", sampler.value_bounds),
                              ("representation_selection", representation_selection),
                              ("stage", stage), ("arm", "R"), ("seed", 13),
                              ("batch_size", config["batch_size"])):
            if state[key] != expected:
                raise ValueError(f"DADP exact-resume incompatibility: {key}")
        plateau = Plateau.from_state(state["plateau"])
        for key, expected in config["plateau"].items():
            if getattr(plateau, key) != expected:
                raise ValueError(f"DADP resume changed plateau rule: {key}")
        if stage == "representation":
            representation.load_state_dict(state["model"])
        else:
            representation.load_state_dict(state["representation"])
            model.model.load_state_dict(state["model"])
            model.model_ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        state["rng"]["torch"] = state["rng"]["torch"].cpu()
        state["rng"]["cuda"] = [item.cpu() for item in state["rng"]["cuda"]]
        restore_rng(state["rng"])
        sampler.rng.bit_generator.state = state["sampler_rng"]
        completed, records = int(state["updates"]), state["development"]
        elapsed_before = state["training_wall_seconds"]
        optimizer_steps = {int(value["step"].item()) for value in state["optimizer"]["state"].values() if "step" in value}
        if not 0 < completed <= config["hard_cap_updates"] or optimizer_steps != {completed}:
            raise ValueError("Saved optimizer counters differ from completed updates")
        if completed != plateau.last_update or not records or records[-1]["updates"] != completed:
            raise ValueError("Resume must be from a completed fixed-DEV/checkpoint boundary")
        if state["best_dev_loss"] != plateau.best_value:
            raise ValueError("Saved best DEV loss differs from stopping state")
        if plateau.best_update == completed:
            # The rolling checkpoint commits first. Repair aliases if the process
            # exited between that commit and the subsequent best/selected saves.
            atomic_torch_save(directory / "best.pt", state)
            atomic_torch_save(directory / "selected.pt", state)
        else:
            for filename in ("best.pt", "selected.pt"):
                saved_best = torch.load(directory / filename, map_location="cpu", weights_only=False)
                if (saved_best["updates"] != plateau.best_update or
                        saved_best["best_dev_loss"] != plateau.best_value):
                    raise ValueError(f"{filename} differs from resumed raw-best selection")
                del saved_best
        # A process can exit after checkpoint commit but before the JSONL append.
        # The checkpoint's records are authoritative for this resumed prefix.
        log_path = directory / "development.jsonl"
        log_temp = log_path.with_suffix(".jsonl.tmp")
        log_temp.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
        log_temp.replace(log_path)
        del state

    atomic_json(directory / "config.json", {
        **config, "data_identity": identity, "value_bounds_train_only": sampler.value_bounds,
        "representation_selection": representation_selection, "gpu_at_start": gpu,
        "representation_anchors_train": [len(pool) for pool in sampler.representation_pools],
        "representation_anchors_development": [len(pool) for pool in development.representation_pools],
        "policy_anchors_train": [len(pool) for pool in sampler.policy_pools],
        "policy_anchors_development": [len(pool) for pool in development.policy_pools],
    })
    started = time.monotonic()
    initial = completed
    losses = []

    def checkpoint(*, best=False, periodic=False, terminal=False):
        state = {
            "stage": stage, "arm": "R", "seed": 13, "updates": completed,
            "batch_size": config["batch_size"], "normalization": sampler.normalization,
            "optimizer": optimizer.state_dict(), "rng": rng_state(),
            "sampler_rng": copy.deepcopy(sampler.rng.bit_generator.state),
            "best_dev_loss": plateau.best_value, "development": records,
            "value_bounds": sampler.value_bounds, "recipe": config,
            "data_identity": identity, "plateau": plateau.state_dict(),
            "representation_selection": representation_selection,
            "representation_checkpoint": str(rep_path) if stage == "policy" else None,
            "training_wall_seconds": elapsed_before + time.monotonic() - started,
            "numerical_history": [{"from_updates": 0, "numerics": {
                "representation": config["representation_precision"],
                "planner": config["policy_precision"] if stage == "policy" else None}}],
        }
        if stage == "representation":
            state["model"] = representation.state_dict()
        else:
            state.update(model=model.model.state_dict(), ema=model.model_ema.state_dict(), representation=representation.state_dict())
        # One rolling full-state checkpoint per DEV check. At least the previous
        # completed boundary remains intact if saving the next one is interrupted.
        path = directory / "resume.pt"
        # Commit before aliases; the resume path repairs a missing/stale best alias
        # from this exact checkpoint if interruption occurs during alias saves.
        atomic_torch_save(path, state)
        if best:
            atomic_torch_save(directory / "best.pt", state)
            atomic_torch_save(directory / "selected.pt", state)
        if periodic:
            atomic_torch_save(directory / "checkpoints" / f"step_{completed:07d}.pt", state)
        if terminal:
            atomic_torch_save(directory / "final.pt", state)
        atomic_json(resume_path, {"checkpoint": str(path), "updates": completed})

    try:
        stop = plateau.monitoring_started and plateau.bad_checks >= plateau.patience
        while completed < config["hard_cap_updates"] and not stop:
            representation.train(stage == "representation")
            if stage == "policy":
                model.train()
            batch = to_device(getattr(sampler, stage)(config["batch_size"]))
            loss, _ = optimization_step(stage, model, representation, batch, optimizer)
            losses.append(loss)
            completed += 1
            if completed % config["check_every_updates"]:
                continue
            validation = fixed_development(stage, model, representation, development, config["batch_size"], config)
            decision = plateau.observe(completed, validation["monitored_loss"])
            stop = decision["stop"]
            row = {"updates": completed, **validation, **decision,
                   "train_loss_interval": float(torch.stack(losses).mean()),
                   "updates_per_second_this_process": (completed - initial) / (time.monotonic() - started),
                   "training_wall_seconds": elapsed_before + time.monotonic() - started}
            records.append(row)
            checkpoint(best=decision["is_best"], periodic=completed % config["periodic_checkpoint_updates"] == 0)
            append_json(directory / "development.jsonl", row)
            progress_metrics = dict(row)
            if "state" in progress_metrics:
                progress_metrics["state_loss"] = progress_metrics.pop("state")
            atomic_json(directory / "progress.json", {**progress_metrics, "stage": stage, "state": "training"})
            print(json.dumps({"stage": stage, **row}), flush=True)
            losses = []
        checkpoint(terminal=True)
        result = {
            "state": "completed", "stage": stage,
            "stop_reason": "validation_plateau" if stop else "budget_limited",
            "updates": completed, "selected_updates": plateau.best_update,
            "selected_update": plateau.best_update,
            "selected_monitored_loss": plateau.best_value,
            "selected_checkpoint": str(directory / "selected.pt"),
            "final_checkpoint": str(directory / "final.pt"),
            "training_wall_seconds": elapsed_before + time.monotonic() - started,
            "recipe": config, "plateau": plateau.state_dict(),
            "control_qualification": "not evaluated; loss selection is not control success",
        }
        atomic_json(directory / "selected.json", {
            "state": "selected", "stage": stage,
            "selected_update": plateau.best_update,
            "selected_checkpoint": str(directory / "selected.pt"),
            "monitored_loss": plateau.best_value, "monitor": config["monitor"],
            "selection_split": "development", "control_qualified": None,
            "qualification_note": "Control capability has not been tested",
        })
        atomic_json(directory / "result.json", result)
        atomic_json(completed_path, result)
        atomic_json(directory / "progress.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        atomic_json(directory / "fatal.json", {
            "stage": stage, "updates": completed, "error": repr(error),
            "traceback": traceback.format_exc(), "updated_at": time.time(),
            "automatic_retry": False,
            "resume_note": "Last fully saved DEV boundary preserves optimizer, RNG and stopping state",
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--stage", choices=("representation", "policy"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.execute:
        print(json.dumps({"state": "prepared_not_launched", "root": str(root),
                          "recipe": recipe(args.stage), "training_updates": 0}, indent=2))
        return
    with training_lock(root):
        train(root, args.stage)


if __name__ == "__main__":
    main()
