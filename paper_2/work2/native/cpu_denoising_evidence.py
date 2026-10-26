"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import math

from contextlib import contextmanager

from dataclasses import dataclass, field

from time import perf_counter

import numpy as np

import torch

from scipy.optimize import minimize_scalar

from scipy.special import logsumexp

from .robust_observation_law import (
    BIAS_BOUNDS,
    SCALE_SQUARED_BOUNDS,
    GaussianResidual,
)

PROBE_TIMES = (0.2, 0.4, 0.6, 0.8)

HYPOTHESES = 8

GAUSSIAN_FLOOR = 1e-4

TEMPERATURE_LOG_BOUNDS = (-12.0, 16.0)

def assert_cpu(value, name="value"):
    """Reject accidental CUDA, lower precision, nonfinite values or active autograd."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert_cpu(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_cpu(item, f"{name}[{index}]")
    elif isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must remain on CPU")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"{name} must use FP32")
        if value.requires_grad:
            raise ValueError(f"{name} must have gradients disabled")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains nonfinite values")

def _float_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be an FP32 CPU tensor")
    assert_cpu(value, name)
    if not value.numel():
        raise ValueError(f"{name} cannot be empty")
    return value

def denoising_log_evidence(errors, temperature):
    errors = _float_tensor(errors, "errors")
    if errors.shape[-1:] != (HYPOTHESES,) or (errors < 0).any():
        raise ValueError("Errors need eight nonnegative entries")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be strictly positive and finite")
    result = -errors / temperature
    assert_cpu(result, "denoising_log_evidence")
    return result

def _cal_labels(labels, count):
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64:
        raise ValueError("Offline CAL labels must be int64 CPU tensors")
    assert_cpu(labels, "CAL labels")
    if labels.shape != (count,) or (labels < 0).any() or (labels >= HYPOTHESES).any():
        raise ValueError("One valid CAL label is required per transition")
    counts = torch.bincount(labels, minlength=HYPOTHESES)
    if (counts == 0).any():
        raise ValueError("CAL must contain every condition")
    return labels, (1 / (HYPOTHESES * counts[labels].float()))

def fit_temperature(errors, true_labels, *, split="CAL"):
    """One positive scalar, fitting condition-balanced CAL posterior NLL only.

    A uniform prior is used for these independently selected CAL transitions.
    Fixed broad numerical bounds prevent under/overflow; this is scalar fitting,
    not a probe-count or controller-parameter sweep. Returns a JSON-safe receipt.
    """
    if split != "CAL":
        raise ValueError("Temperature fitting requires split='CAL'")
    errors = _float_tensor(errors, "CAL errors")
    if errors.ndim != 2 or errors.shape[1] != HYPOTHESES or (errors < 0).any():
        raise ValueError("CAL errors must have shape [transitions,8]")
    labels, weights = _cal_labels(true_labels, len(errors))
    values = errors.numpy().astype(np.float64)
    values -= values.min(axis=1, keepdims=True)
    target, weights_np = labels.numpy(), weights.numpy().astype(np.float64)

    def objective(log_temperature):
        logits = -values / math.exp(log_temperature)
        losses = logsumexp(logits, axis=1) - logits[np.arange(len(values)), target]
        return float((losses * weights_np).sum())

    fitted = minimize_scalar(
        objective,
        bounds=TEMPERATURE_LOG_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 200},
    )
    if not fitted.success or not math.isfinite(fitted.fun):
        raise RuntimeError("CAL temperature optimization failed")
    return {
        "temperature": math.exp(float(fitted.x)),
        "split": split,
        "cal_count": len(errors),
        "cal_nll": float(fitted.fun),
        "method": "bounded_scalar_log_temperature_CAL_NLL",
        "log_temperature_bounds": list(TEMPERATURE_LOG_BOUNDS),
        "at_numerical_bound": min(abs(float(fitted.x) - bound) for bound in TEMPERATURE_LOG_BOUNDS)
        < 1e-4,
        "weighting": "equal_condition_then_equal_transition",
        "approximate_evidence": True,
    }

def _moments(moments):
    if not isinstance(moments, dict) or not {"mean", "variance"} <= moments.keys():
        raise ValueError("Moments require mean and variance")
    mean = _float_tensor(moments["mean"], "mean")
    variance = _float_tensor(moments["variance"], "variance")
    if mean.ndim != 3 or mean.shape[1] != HYPOTHESES or mean.shape != variance.shape:
        raise ValueError("Moments must have matching [transitions,8,features] shapes")
    if (variance < 0).any():
        raise ValueError("Predictive variance cannot be negative")
    return mean, variance

def fit_gaussian_calibration(moments, observed, true_labels, *, split="CAL", floor=GAUSSIAN_FLOOR):
    """Fresh joint observation/reward residual calibration; no old law is reused.

    First add a CAL residual-variance correction to the sample variance. Then
    fit the existing GaussianResidual location and scale parameterization
    analytically on those standardized CAL residuals, with its existing bounds.
    """
    if split != "CAL":
        raise ValueError("Gaussian fitting requires split='CAL'")
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("A strictly positive normalized variance floor is required")
    mean, variance = _moments(moments)
    observed = _float_tensor(observed, "observed")
    if observed.shape != (len(mean), mean.shape[-1]):
        raise ValueError("CAL observations must have matching [transitions,features] shape")
    labels, weights = _cal_labels(true_labels, len(mean))
    index = torch.arange(len(mean))
    mean, variance = mean[index, labels], variance[index, labels]
    residual = observed - mean
    extra = ((residual.square() - variance) * weights[:, None]).sum(0).clamp_min(floor)
    standardized = residual / (variance + extra).sqrt()
    bias = (standardized * weights[:, None]).sum(0).clamp(*BIAS_BOUNDS)
    scale = ((standardized - bias).square() * weights[:, None]).sum(0)
    scale = scale.clamp(*SCALE_SQUARED_BOUNDS)
    return {
        "family": "cpu_matched_gaussian_observation_reward",
        "split": split,
        "cal_count": len(observed),
        "feature_count": observed.shape[-1],
        "features": "checkpoint_normalized_next_observation_delta_and_reward",
        "weighting": "equal_condition_then_equal_transition",
        "extra_variance": extra.tolist(),
        "normalized_variance_floor": float(floor),
        "residual_law": GaussianResidual(tuple(scale.tolist()), tuple(bias.tolist())).to_spec(),
        "denoising_weighting": "mean_observation_coordinate_error_plus_reward_error",
        "gaussian_weighting": "normalized_joint_density_sums_all_coordinates",
        "original_observation_only_law_reused": False,
    }

def gaussian_log_evidence(observed, moments, calibration):
    mean, variance = _moments(moments)
    observed = _float_tensor(observed, "observed")
    if observed.shape != (len(mean), mean.shape[-1]):
        raise ValueError("Observed variables must match the predictive feature dimensions")
    if calibration.get("family") != "cpu_matched_gaussian_observation_reward":
        raise ValueError("This pilot requires its fresh observation-and-reward calibration")
    extra = torch.tensor(calibration["extra_variance"], dtype=torch.float32)
    if extra.shape != mean.shape[-1:] or (extra <= 0).any():
        raise ValueError("CAL residual variance must match all public evidence variables")
    law = GaussianResidual.from_spec(calibration["residual_law"])
    result = law.log_prob(observed[:, None], mean, variance + extra)
    assert_cpu(result, "gaussian_log_evidence")
    return result

@dataclass
class QueryStats:
    forward_calls: int = 0
    forward_examples: int = 0
    maximum_forward_batch: int = 0
    operation_calls: dict = field(default_factory=dict)
    operation_seconds: dict = field(default_factory=dict)
    total_seconds: float = 0.0

    def snapshot(self):
        return {
            "forward_calls": self.forward_calls,
            "forward_examples": self.forward_examples,
            "maximum_forward_batch": self.maximum_forward_batch,
            "operation_calls": dict(self.operation_calls),
            "operation_seconds": dict(self.operation_seconds),
            "total_seconds": self.total_seconds,
            "seconds_semantics": "outermost_engine_calls_wall_time_including_preparation",
            "forward_semantics": "actual_backbone_calls_including_final_sampler_event_pass",
        }

class CPUQueryEngine:
    """Chunk all actual neural forwards to <=32, with CPU assertions and accounting."""

    def __init__(self, model, batch_size=32):
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("CPU query batch_size must be in [1,32]")
        model.eval()
        model.requires_grad_(False)
        assert_cpu(dict(model.named_parameters()), "parameters")
        assert_cpu(dict(model.named_buffers()), "buffers")
        self.model, self.batch_size = model, batch_size
        self.stats = QueryStats()
        self._depth = 0
        self._hook = model.register_forward_pre_hook(self._before_forward)

    def close(self):
        self._hook.remove()

    def _before_forward(self, model, args):
        assert_cpu(args, "neural_query")
        size = len(args[0])
        if size > self.batch_size:
            raise ValueError("Neural query exceeded the configured CPU batch limit")
        if model.training or torch.is_grad_enabled():
            raise RuntimeError("CPU pilot queries require evaluation mode and disabled gradients")
        self.stats.forward_calls += 1
        self.stats.forward_examples += size
        self.stats.maximum_forward_batch = max(self.stats.maximum_forward_batch, size)

    @contextmanager
    def _operation(self, name):
        started, outer = perf_counter(), self._depth == 0
        self._depth += 1
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self._depth -= 1
            self.stats.operation_calls[name] = self.stats.operation_calls.get(name, 0) + 1
            self.stats.operation_seconds[name] = self.stats.operation_seconds.get(name, 0) + elapsed
            if outer:
                self.stats.total_seconds += elapsed

    def _history(self, history):
        if set(history) != {"observations", "actions", "rewards", "mask"}:
            raise ValueError("History must contain only public observations/actions/rewards/mask")
        assert_cpu(history, "history")
        n, length, dimensions = history["observations"].shape
        c = self.model.config
        expected = {
            "observations": (n, c.history, c.observation_dim),
            "actions": (n, c.history - 1, c.action_dim),
            "rewards": (n, c.history - 1),
            "mask": (n, c.history),
        }
        if n < 1 or length != c.history or dimensions != c.observation_dim:
            raise ValueError("Invalid batched public history")
        if any(history[key].shape != shape for key, shape in expected.items()):
            raise ValueError("Public history shapes must match the checkpoint")
        if not ((history["mask"] == 0) | (history["mask"] == 1)).all():
            raise ValueError("History mask must be binary")
        if not (history["mask"][:, -1] == 1).all():
            raise ValueError("Current observation must be present")
        return n

    def _encode_outcomes(self, history, actions, observations, rewards):
        n = self._history(history)
        c = self.model.config
        assert_cpu((actions, observations, rewards), "transition")
        if actions.shape == (n, c.action_dim):
            actions = actions[:, None]
        if observations.shape == (n, c.observation_dim):
            observations = observations[:, None]
        if rewards.shape == (n,):
            rewards = rewards[:, None]
        if (
            actions.shape != (n, 1, c.action_dim)
            or observations.shape != (n, 1, c.observation_dim)
            or rewards.shape != (n, 1)
        ):
            raise ValueError("Evidence requires one actual H1 action, next observation and reward")
        if (actions.abs() > 1.000001).any():
            raise ValueError("Executed actions must obey physical bounds")
        return self.model.encode(
            {
                "history": history,
                "actions": actions,
                "future_observations": observations,
                "rewards": rewards,
                "condition": torch.zeros(n, dtype=torch.int64),
                "horizon": torch.ones(n, dtype=torch.int64),
                "role": torch.ones(n, dtype=torch.int64),
                "quality": torch.zeros(n, dtype=torch.int64),
                "valid": torch.ones(n, 1, dtype=torch.bool),
            }
        )

    @torch.inference_mode()
    def normalize_outcomes(self, history, next_observations, rewards):
        with self._operation("normalize_outcomes"):
            n = self._history(history)
            actions = torch.zeros(n, 1, self.model.config.action_dim)
            encoded = self._encode_outcomes(history, actions, next_observations, rewards)
            result = encoded["x"][:, 1].clone()
            assert_cpu(result, "normalized_outcomes")
            return result

    @torch.inference_mode()
    def denoising_errors(
        self, history, actions, next_observations, rewards, *, noise=None, generator=None
    ):
        with self._operation("denoising_errors"):
            encoded = self._encode_outcomes(history, actions, next_observations, rewards)
            n, _, width = encoded["x"].shape
            if noise is None:
                noise = torch.randn(n, len(PROBE_TIMES), 2, width, generator=generator)
            assert_cpu(noise, "probe_noise")
            if noise.shape != (n, len(PROBE_TIMES), 2, width):
                raise ValueError("Probe noise must have shape [histories,4,2,token_width]")
            times = torch.tensor(PROBE_TIMES, dtype=torch.float32)
            errors = torch.empty(n * len(PROBE_TIMES) * HYPOTHESES)
            for start in range(0, len(errors), self.batch_size):
                index = torch.arange(start, min(start + self.batch_size, len(errors)))
                source, probe, condition = index // 32, index // 8 % 4, index % 8
                current = {key: value[source] for key, value in encoded.items()}
                current["condition"] = condition
                time = times[probe]
                alpha, sigma = torch.cos(time * math.pi / 2), torch.sin(time * math.pi / 2)
                noisy = alpha[:, None, None] * current["x"]
                noisy = noisy + sigma[:, None, None] * noise[source, probe]
                noisy = torch.where(current["known"], current["x"], noisy) * current["semantic"]
                predicted, _ = self.model(noisy, current, time)
                assert_cpu(predicted, "denoised_x0")
                squared = (predicted[:, 1] - current["x"][:, 1]).square()
                block_error = squared[:, : self.model.config.observation_dim].mean(-1)
                block_error = block_error + squared[:, -1]
                errors[index] = alpha.square() / sigma.square() * block_error
            result = errors.reshape(n, len(PROBE_TIMES), HYPOTHESES).mean(1)
            assert_cpu(result, "denoising_errors")
            return result

    @torch.inference_mode()
    def predict(self, history, condition, actions=None, *, horizon=4, noise=None, generator=None):
        with self._operation("predict"):
            n = self._history(history)
            assert_cpu((condition, actions, noise), "prediction_arguments")
            if condition.shape != (n,) or condition.dtype != torch.int64:
                raise ValueError("One integer enumerated hypothesis is required per prediction")
            if (condition < 0).any() or (condition >= HYPOTHESES).any():
                raise ValueError("Unknown enumerated hypothesis")
            if type(horizon) is not int or not 1 <= horizon <= self.model.config.max_horizon:
                raise ValueError("Unsupported prediction horizon")
            if actions is not None and actions.shape != (n, horizon, self.model.config.action_dim):
                raise ValueError("Imposed actions must match [queries,horizon,action_dim]")
            shape = (n, 2 * horizon, self.model.token_width)
            if noise is None:
                noise = torch.randn(shape, generator=generator)
            if noise.shape != shape:
                raise ValueError("Prediction noise must match all token coordinates")
            assert_cpu(noise, "prediction_noise")
            parts = {}
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                result = self.model.sample(
                    {key: value[start:stop] for key, value in history.items()},
                    condition[start:stop],
                    horizon=horizon,
                    fixed_actions=None if actions is None else actions[start:stop],
                    noise=noise[start:stop],
                    quality=1 if actions is None else 0,
                )
                assert_cpu(result, "sample_output")
                for key, value in result.items():
                    parts.setdefault(key, []).append(value)
            return {key: torch.cat(value) for key, value in parts.items()}

    @torch.inference_mode()
    def gaussian_moments(self, history, actions, *, particles=16, noise=None, generator=None):
        with self._operation("gaussian_moments"):
            n = self._history(history)
            c = self.model.config
            assert_cpu(actions, "actual_actions")
            if actions.shape == (n, c.action_dim):
                actions = actions[:, None]
            if actions.shape != (n, 1, c.action_dim):
                raise ValueError("Gaussian evidence requires actual H1 actions")
            if type(particles) is not int or particles < 2:
                raise ValueError("At least two Gaussian particles are required")
            shape = (n, particles, 2, self.model.token_width)
            if noise is None:
                noise = torch.randn(shape, generator=generator)
            assert_cpu(noise, "gaussian_common_noise")
            if noise.shape != shape:
                raise ValueError("Gaussian noise must be [histories,particles,2,token_width]")
            # Particle noise is identical across all eight tested hypotheses.
            index = torch.arange(n).repeat_interleave(HYPOTHESES * particles)
            expanded_history = {key: value[index] for key, value in history.items()}
            conditions = torch.arange(HYPOTHESES).repeat_interleave(particles).repeat(n)
            expanded_noise = noise[:, None].expand(-1, HYPOTHESES, -1, -1, -1)
            predictions = self.predict(
                expanded_history,
                conditions,
                actions[index],
                horizon=1,
                noise=expanded_noise.reshape(-1, 2, self.model.token_width),
            )
            normalized = self.normalize_outcomes(
                expanded_history, predictions["observations"], predictions["rewards"]
            ).reshape(n, HYPOTHESES, particles, self.model.token_width)
            return {
                "mean": normalized.mean(2),
                "variance": normalized.var(2, unbiased=False),
            }


def categorical_update(prior, log_evidence):
    """Exact categorical VFE minimizer relative to the supplied approximate law."""
    prior = _probability(prior)
    log_evidence = _float_tensor(log_evidence, "log_evidence")
    if log_evidence.shape != prior.shape:
        raise ValueError("Prior and evidence must have identical [...,8] shapes")
    logits = torch.where(prior > 0, prior.log(), -torch.inf) + log_evidence
    return torch.softmax(logits, dim=-1)


def entropy(prior):
    prior = _probability(prior)
    return -torch.special.xlogy(prior, prior).sum(-1)

def _probability(prior):
    prior = _float_tensor(prior, "prior")
    if prior.shape[-1:] != (HYPOTHESES,) or (prior < 0).any():
        raise ValueError("A prior needs eight nonnegative entries")
    total = prior.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("A prior must have positive total mass")
    return prior / total
