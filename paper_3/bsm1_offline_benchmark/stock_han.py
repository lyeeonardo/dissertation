"""Stock Han architecture/settings on the shared public55 dataset.

A fresh paper-derived reimplementation, not the contextual/prefix adaptations.
Eq.21 sums time points; costs use declared TRAIN-only normalization and the
previous completed interval as the root label. Historical modules are untouched.
"""

from __future__ import annotations

import argparse

from dataclasses import asdict, dataclass

import json

from pathlib import Path

import time

import numpy as np

import torch

from torch import nn

from .koopman import DIOKO, KoopmanConfig, atomic_checkpoint, atomic_json, device_data, gather_windows, load_data

MODEL_CLASS = "StockHanDIOKO"

@dataclass(frozen=True)
class StockHanConfig(KoopmanConfig):
    observation_dim: int = 55
    batch_size: int = 128

class StockHanDIOKO(DIOKO):
    def __init__(self, config=StockHanConfig(), **kwargs):
        super().__init__(config, **kwargs)

    def loss(self, observations, actions, costs, root_costs):
        if root_costs.shape != (len(observations),):
            raise ValueError("One previous-interval root cost is required per window")
        _, base = super().loss(observations, actions, costs)
        horizon = actions.shape[1]
        root_prediction = self.normalized_cost(self.encode(observations[:, 0]))
        root_loss = (root_prediction - root_costs / self.cost_scale).square().mean()
        # Eq.21 includes j=k..k+H. Root latent-consistency error is exactly zero.
        transition = base["transition"] * horizon
        cost = base["cost"] * horizon + root_loss
        regularization = base["regularization"]
        total = transition + cost + self.config.l2_coefficient * regularization
        return total, dict(transition=transition, cost=cost, root_cost=root_loss,
                           future_cost=base["cost"], regularization=regularization,
                           selection_score=transition + cost)

def load_paper_data(path, config):
    data = load_data(path, config)
    starts = data["starts"]
    starts = starts[starts > 0]
    valid_previous = ((data["episode_id"][starts] == data["episode_id"][starts - 1]) &
                      (data["step"][starts] == data["step"][starts - 1] + 1) &
                      ~data["terminated"][starts - 1].astype(bool))
    data["starts"] = starts[valid_previous]
    if not len(data["starts"]):
        raise ValueError("No complete windows with a same-episode previous interval")
    return data

def paper_windows(data, starts, horizon):
    return (*gather_windows(data, starts, horizon), data["costs"][starts - 1])

@torch.no_grad()
def validate(model, data, starts):
    model.eval()
    names = ("transition", "cost", "root_cost", "future_cost", "regularization", "selection_score")
    totals = dict.fromkeys(names, 0.)
    for offset in range(0, len(starts), model.config.batch_size):
        selected = starts[offset:offset + model.config.batch_size]
        _, values = model.loss(*paper_windows(data, selected, model.config.horizon))
        for key in totals:
            totals[key] += float(values[key]) * len(selected)
    return {key: value / len(starts) for key, value in totals.items()}

def train(args):
    if not getattr(args, "implementation_test", False):
        fixed = dict(epochs=400, batch_size=128, learning_rate=.001, horizon=30, seed=17)
        if any(getattr(args, name) != value for name, value in fixed.items()):
            raise ValueError("Production stock Han settings are frozen; small CPU tests require --implementation-test")
        if torch.device(args.device).type != "cuda":
            raise ValueError("Production training requires CUDA; CPU is only for explicit implementation tests")
    elif torch.device(args.device).type != "cpu":
        raise ValueError("Implementation tests must use CPU")
    config = StockHanConfig(observation_dim=55, epochs=args.epochs,
                           batch_size=args.batch_size, learning_rate=args.learning_rate,
                           seed=args.seed, horizon=args.horizon,
                           validation_windows=args.validation_windows)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no CPU fallback")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output, data_root = Path(args.output), Path(args.data)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh paper training output directory")
    output.mkdir(parents=True, exist_ok=True)
    training_raw = load_paper_data(data_root / "train.npz", config)
    validation_raw = load_paper_data(data_root / "validation.npz", config)
    mean = training_raw["observations"].mean(0, dtype=np.float64)
    scale = training_raw["observations"].std(0, dtype=np.float64).clip(1e-4)
    cost_scale = float(training_raw["costs"].mean(dtype=np.float64))
    if cost_scale <= 0:
        raise ValueError("Positive TRAIN mean economic cost required")
    model = StockHanDIOKO(config, observation_mean=mean, observation_scale=scale,
                       cost_scale=cost_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed + 4103)
    val_starts = rng.choice(validation_raw["starts"],
                            min(config.validation_windows, len(validation_raw["starts"])), replace=False)
    starts = torch.as_tensor(training_raw["starts"], device=device, dtype=torch.long)
    val_starts_tensor = torch.as_tensor(val_starts, device=device, dtype=torch.long)
    training, validation = device_data(training_raw, device), device_data(validation_raw, device)
    metadata = dict(method="Han2024_stock_public55_shared_data", model_class=MODEL_CLASS,
        config=asdict(config), source="https://arxiv.org/html/2405.12478v2",
        data=str(data_root.resolve()), train_transitions=len(training_raw["costs"]),
        train_windows=len(starts), validation_windows=val_starts.tolist(),
        expected_updates=config.epochs * ((len(starts) + config.batch_size - 1) // config.batch_size),
        loss="Eq21 root and H future costs plus H lifted-consistency terms; SUM over H+1 instants; batch expectation",
        cost_time="c_j is the interval-average economic cost ending at state j; root uses previous interval",
        excluded_roots="episode reset, split start, or discontinuous previous transition",
        normalization="TRAIN observation mean/std, std floor1e-4; TRAIN mean cost; no cost centering; paper unspecified",
        l2="0.1 times mean encoder/dynamics squared weights, no biases/economic-head penalty; paper tensors unspecified",
        gradient_check="finite gradients required; no gradient clipping",
        controller_checkpoint="final.pt after400epochs by default; best.pt validation-only diagnostic",
        checkpoint_selection="final completed epoch; no closed-loop test selection",
        input="clean55: declared41 sensor reconstruction plus14 current inlet; no action memory",
        runtime_device=str(device), paper_horizon_ambiguity="Table7 H30; Figure3 reports16-step error",
        unresolved="author sensor indices, training normalization, SP instantaneous labeling and QP solver unspecified",
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    atomic_json(output / "config.json", metadata)
    started, last_progress, updates = time.time(), 0., 0
    history, best = [], float("inf")
    atomic_json(output / "progress.json", dict(status="running", epoch=0, updates=0, started_unix=started))
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = starts[torch.randperm(len(starts), device=device)]
        total_loss, seen = 0., 0
        for offset in range(0, len(permutation), config.batch_size):
            selected = permutation[offset:offset + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.loss(*paper_windows(training, selected, config.horizon))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite paper DIOKO loss; preserve latest completed epoch")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach()) * len(selected)
            seen += len(selected)
            now = time.time()
            if now - last_progress >= 10:
                atomic_json(output / "progress.json", dict(status="running", epoch=epoch,
                    updates=updates, completed_epoch=epoch - 1, epoch_windows=seen,
                    elapsed_seconds=now - started, last_progress_unix=now))
                last_progress = now
        metrics = validate(model, validation, val_starts_tensor)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite paper validation loss")
        row = dict(epoch=epoch, updates=updates, train_loss=total_loss / seen,
                   validation=metrics, elapsed_seconds=time.time() - started)
        history.append(row)
        payload = dict(model_class=MODEL_CLASS, config=asdict(config),
            model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(), epoch=epoch, updates=updates,
            validation=metrics, metadata=metadata, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        atomic_checkpoint(output / "latest.pt", payload)
        if metrics["selection_score"] < best:
            best = metrics["selection_score"]
            atomic_checkpoint(output / "best.pt", payload)
        if epoch == config.epochs:
            atomic_checkpoint(output / "final.pt", payload)
        atomic_json(output / "history.json", history)
        atomic_json(output / "progress.json", dict(status="complete" if epoch == config.epochs else "running",
            epoch=epoch, completed_epoch=epoch, updates=updates, best_selection=best,
            elapsed_seconds=time.time() - started, last_progress_unix=time.time(), validation=metrics))
        print(json.dumps(row), flush=True)
    atomic_json(output / "completion.json", dict(status="complete", method="stock_han", model_class=MODEL_CLASS, epochs=config.epochs,
        updates=updates, best_selection=best, elapsed_seconds=time.time() - started,
        controller_checkpoint=str((output / "final.pt").resolve()),
        selected_checkpoint=str((output / "final.pt").resolve()),
        train_transitions=len(training_raw["costs"]), validation_transitions=len(validation_raw["costs"]),
        train_windows=len(starts), fresh_initialization=True,
        data=str(data_root.resolve()), best_is_diagnostic_only=True))

def load_controller(path, device="cpu"):
    from .koopman import KoopmanController
    path = Path(path)
    if path.is_dir():
        path = path / "final.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_class") != MODEL_CLASS:
        raise ValueError("Expected the independent stock Han checkpoint")
    config = StockHanConfig(**payload["config"])
    if config.observation_dim != 55 or config.latent_dim != 60 or config.width != 128:
        raise ValueError("Stock Han checkpoint architecture differs from the declared model")
    model = StockHanDIOKO(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.requires_grad_(False)
    return KoopmanController(model)

def main(argv=None):
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    parser = commands.add_parser("train")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--validation-windows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--implementation-test", action="store_true")
    args = root.parse_args(argv)
    try:
        train(args)
    except Exception as exc:
        progress = Path(args.output) / "progress.json"
        if progress.exists():
            value = json.loads(progress.read_text())
            value.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
            atomic_json(progress, value)
        raise

if __name__ == "__main__":
    main()
