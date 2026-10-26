"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import math

from contextlib import contextmanager

from numbers import Integral

import numpy as np

import torch

from .cpu_denoising_control import FORCE_SCALES, REAL_HAZARD, _rng

from .cpu_denoising_evidence import PROBE_TIMES, QueryStats

from .cpu_proposal_control import EXTRA_NOISE_TAGS, REFINEMENT_INDEX, SAMPLING_STEPS

METHODS = tuple(
    f"{proposal}_{selector}"
    for proposal in ("scaled", "independent", "refinement")
    for selector in ("passive", "h1")
)

FROZEN_TEMPERATURE = 0.007202860241169122

FROZEN_INFORMATION_WEIGHT = 0.21594711941402683

PUBLIC_SHAPES = dict(observations=(8, 17), actions=(7, 6), rewards=(7,), mask=(8,))

def _finite(value, name="value"):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{name}.{key}")
    elif isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise FloatingPointError(f"FATAL_CONTROLLER: nonfinite {name}")

def _tensor(value, device):
    value = torch.as_tensor(value, device=device)
    if value.requires_grad:
        raise ValueError("Inference inputs must have gradients disabled")
    value = value.to(dtype=torch.float32)
    _finite(value)
    return value

def public_history(history, device):
    if set(history) != set(PUBLIC_SHAPES):
        raise ValueError("Only public observations, executed actions/rewards and mask are allowed")
    result = {}
    for key, shape in PUBLIC_SHAPES.items():
        value = _tensor(history[key], device)
        if value.ndim == len(shape):
            value = value.unsqueeze(0)
        if value.ndim != len(shape) + 1 or value.shape[1:] != shape:
            raise ValueError(f"Invalid public {key} shape")
        result[key] = value.clone()
    if len({len(value) for value in result.values()}) != 1 or not len(result["mask"]):
        raise ValueError("Public history batch lengths differ or are empty")
    mask = result["mask"]
    if not ((mask == 0) | (mask == 1)).all() or not (mask[:, -1] == 1).all():
        raise ValueError("Public mask must be binary with the current observation present")
    return result

def _probability(prior):
    if prior.shape[-1:] != (8,) or (prior < 0).any():
        raise ValueError("Belief needs eight nonnegative probabilities")
    _finite(prior, "belief")
    total = prior.sum(-1, keepdim=True)
    if (total <= 0).any():
        raise ValueError("Belief must have positive mass")
    return prior / total

def _belief(value, device):
    value = _tensor(value, device)
    if (
        value.shape != (8,)
        or (value < 0).any()
        or not torch.isclose(value.sum(), value.new_tensor(1.0), atol=1e-6, rtol=0)
    ):
        raise ValueError("Expected normalized belief over eight conditions")
    # Match the pilot: validation here does not renormalize an existing prior.
    return value.clone()

def predict_prior(prior):
    prior = _probability(prior)
    return prior * (1 - REAL_HAZARD) + (1 - prior) * REAL_HAZARD / 7

def categorical_update(prior, log_evidence):
    prior = _probability(prior)
    if prior.shape != log_evidence.shape:
        raise ValueError("Prior and evidence shapes differ")
    _finite(log_evidence, "log_evidence")
    return torch.softmax(torch.where(prior > 0, prior.log(), -torch.inf) + log_evidence, -1)

def entropy(prior):
    prior = _probability(prior)
    return -torch.special.xlogy(prior, prior).sum(-1)

def _noise(shape, seed, stage, device):
    return torch.randn(*shape, generator=_rng(seed, stage)).to(device)

class OptimizedProposalEngine:
    """Frozen FP32 model queries; one case may batch up to 768 H1 evidence rows.

    The caller measures synchronized decision/update latency. This engine does
    not synchronize or scan for nonfinite tensors inside every backbone call.
    """

    def __init__(self, model, batch_size=256):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Query batch_size must be a positive integer")
        self.device = next(model.parameters()).device
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU parity and CUDA execution are supported")
        c = model.config
        if (c.history, c.observation_dim, c.action_dim, c.sampling_steps) != (8, 17, 6, 8):
            raise ValueError("Frozen history8/17-observation/6-action/eight-step model required")
        if c.max_horizon < 4 or not c.condition_input:
            raise ValueError("The frozen conditional model must support H4")
        model.eval().requires_grad_(False)
        for value in (*model.parameters(), *model.buffers()):
            if value.device != self.device or (
                value.is_floating_point() and value.dtype != torch.float32
            ):
                raise ValueError("All frozen model tensors must be colocated FP32")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model, self.batch_size = model, batch_size
        self.stats = QueryStats()
        self._hook = model.register_forward_pre_hook(self._before_forward)

    def close(self):
        self._hook.remove()

    def _before_forward(self, model, args):
        if model.training or torch.is_grad_enabled() or args[0].dtype != torch.float32:
            raise RuntimeError("Optimized queries require frozen evaluation and FP32 inference")
        size = len(args[0])
        if size > self.batch_size:
            raise ValueError("Neural query exceeded configured batch limit")
        self.stats.forward_calls += 1
        self.stats.forward_examples += size
        self.stats.maximum_forward_batch = max(self.stats.maximum_forward_batch, size)

    @contextmanager
    def _fp32(self):
        with torch.autocast(device_type=self.device.type, enabled=False):
            yield

    def _encode(self, history, conditions, horizon, actions=None, observations=None, rewards=None):
        n, c, device = len(conditions), self.model.config, self.device
        fixed = actions is not None
        zeros = torch.zeros(n, horizon, device=device)
        return self.model.encode(
            dict(
                history=history,
                condition=conditions,
                actions=torch.zeros(n, horizon, c.action_dim, device=device)
                if actions is None
                else actions,
                future_observations=torch.zeros(n, horizon, c.observation_dim, device=device)
                if observations is None
                else observations,
                rewards=zeros if rewards is None else rewards,
                valid=torch.ones(n, horizon, dtype=torch.bool, device=device),
                horizon=torch.full((n,), horizon, dtype=torch.long, device=device),
                role=torch.full(
                    (n,),
                    (1 if horizon == 1 else 2) if fixed else 0,
                    dtype=torch.long,
                    device=device,
                ),
                quality=torch.full((n,), 0 if fixed else 1, dtype=torch.long, device=device),
            )
        )

    def _finish(self, x, encoded, history, grid):
        n = len(x)
        for current, following in zip(grid[:-1], grid[1:], strict=True):
            prediction, _ = self.model(x, encoded, current.expand(n))
            prediction = prediction.float()
            alpha, sigma = torch.cos(current * math.pi / 2), torch.sin(current * math.pi / 2)
            epsilon = (x - alpha * prediction) / sigma.clamp_min(1e-5)
            x = (
                torch.cos(following * math.pi / 2) * prediction
                + torch.sin(following * math.pi / 2) * epsilon
            )
            x = torch.where(encoded["known"], encoded["x"], x) * encoded["semantic"]
        _, terminal = self.model(x, encoded, torch.zeros(n, device=self.device))
        c = self.model.config
        return dict(
            actions=x[:, ::2, : c.action_dim].clamp(-1, 1),
            joint_observations=x[:, 1::2, : c.observation_dim] * self.model.delta_std
            + self.model.delta_mean
            + history["observations"][:, -1, None],
            joint_rewards=x[:, 1::2, -1] * self.model.reward_std + self.model.reward_mean,
            termination_probability=terminal.float().sigmoid(),
            clean_joint_tokens=x.clone(),
        )

    @torch.inference_mode()
    def joint_sample(self, history, condition, *, noise, clean_joint_tokens=None):
        history = public_history(history, self.device)
        n = len(history["mask"])
        condition = torch.as_tensor(condition, device=self.device)
        if (
            condition.shape != (n,)
            or condition.dtype != torch.long
            or ((condition < 0) | (condition > 7)).any()
        ):
            raise ValueError("One valid integer hypothesis is required per joint sample")
        noise = _tensor(noise, self.device)
        if noise.shape != (n, 8, 18):
            raise ValueError("Joint noise must have shape [N,8,18]")
        if clean_joint_tokens is not None:
            clean_joint_tokens = _tensor(clean_joint_tokens, self.device)
            if clean_joint_tokens.shape != noise.shape:
                raise ValueError("Refinement requires all clean joint tokens")
        grid = torch.linspace(0.999, 0, SAMPLING_STEPS + 1, device=self.device)
        if clean_joint_tokens is not None:
            grid = grid[REFINEMENT_INDEX:]
        parts = {}
        with self._fp32():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                context = {key: value[start:stop] for key, value in history.items()}
                encoded = self._encode(context, condition[start:stop], 4)
                x = noise[start:stop]
                if clean_joint_tokens is not None:
                    x = (
                        torch.cos(grid[0] * math.pi / 2) * clean_joint_tokens[start:stop]
                        + torch.sin(grid[0] * math.pi / 2) * x
                    )
                x = torch.where(encoded["known"], encoded["x"], x) * encoded["semantic"]
                output = self._finish(x, encoded, context, grid)
                if clean_joint_tokens is not None:
                    output["forward_noised_tokens"] = x
                for key, value in output.items():
                    parts.setdefault(key, []).append(value)
        result = {key: torch.cat(value) for key, value in parts.items()}
        if clean_joint_tokens is not None:
            result["refinement_time_grid"] = grid
        _finite(result, "joint_sample")
        return result

    def refine_joint(self, history, condition, clean_joint_tokens, *, noise):
        return self.joint_sample(
            history, condition, noise=noise, clean_joint_tokens=clean_joint_tokens
        )

    @torch.inference_mode()
    def predict(self, history, condition, actions=None, *, horizon=4, noise):
        history = public_history(history, self.device)
        condition = torch.as_tensor(condition, device=self.device)
        n = len(history["mask"])
        if (
            condition.shape != (n,)
            or condition.dtype != torch.long
            or ((condition < 0) | (condition > 7)).any()
        ):
            raise ValueError("One valid integer hypothesis is required per prediction")
        if horizon != 4:
            raise ValueError("The optimized pilot preserves its H4 prediction query")
        noise = _tensor(noise, self.device)
        if noise.shape != (n, 8, 18):
            raise ValueError("Prediction noise must cover all H4 joint tokens")
        if actions is not None:
            actions = _tensor(actions, self.device)
            if actions.shape != (n, 4, 6) or (actions.abs() > 1).any():
                raise ValueError("Imposed actions must be bounded [N,4,6]")
        parts = {}
        with self._fp32():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                result = self.model.sample(
                    {key: value[start:stop] for key, value in history.items()},
                    condition[start:stop],
                    horizon=4,
                    fixed_actions=None if actions is None else actions[start:stop],
                    noise=noise[start:stop],
                    quality=1 if actions is None else 0,
                )
                for key, value in result.items():
                    parts.setdefault(key, []).append(value)
        result = {key: torch.cat(value) for key, value in parts.items()}
        _finite(result, "prediction")
        return result

    @torch.inference_mode()
    def denoising_errors(self, history, actions, next_observations, rewards, *, noise):
        history = public_history(history, self.device)
        n, device = len(history["mask"]), self.device
        actions = _tensor(actions, device).reshape(n, 1, 6)
        observations = _tensor(next_observations, device).reshape(n, 1, 17)
        rewards = _tensor(rewards, device).reshape(n, 1)
        if (actions.abs() > 1.000001).any():
            raise ValueError("Evidence must use bounded executed actions")
        encoded = self._encode(
            history,
            torch.zeros(n, dtype=torch.long, device=device),
            1,
            actions,
            observations,
            rewards,
        )
        noise = _tensor(noise, device)
        if noise.shape != (n, 4, 2, 18):
            raise ValueError("Evidence noise must have shape [N,4,2,18]")
        times = torch.tensor(PROBE_TIMES, dtype=torch.float32, device=device)
        errors = torch.empty(n * 32, device=device)
        with self._fp32():
            for start in range(0, len(errors), self.batch_size):
                index = torch.arange(
                    start, min(start + self.batch_size, len(errors)), device=device
                )
                source, probe, condition = index // 32, index // 8 % 4, index % 8
                current = {key: value[source] for key, value in encoded.items()}
                current["condition"] = condition
                time = times[probe]
                alpha, sigma = torch.cos(time * math.pi / 2), torch.sin(time * math.pi / 2)
                noisy = alpha[:, None, None] * current["x"]
                noisy = noisy + sigma[:, None, None] * noise[source, probe]
                noisy = torch.where(current["known"], current["x"], noisy) * current["semantic"]
                predicted, _ = self.model(noisy, current, time)
                squared = (predicted[:, 1] - current["x"][:, 1]).square()
                block_error = squared[:, :17].mean(-1) + squared[:, -1]
                errors[index] = alpha.square() / sigma.square() * block_error
        result = errors.reshape(n, 4, 8).mean(1)
        _finite(result, "denoising_errors")
        return result

@torch.inference_mode()
def generate_candidate_bank(engine, history, belief, *, method, seed):
    if method not in ("scaled", "independent", "refinement"):
        raise ValueError("Unknown frozen proposal method")
    device = engine.device
    history = public_history(history, device)
    belief = _belief(belief, device)
    if belief.shape != (8,) or len(history["mask"]) != 1:
        raise ValueError("Proposal bank requires one public root and one belief")
    condition = torch.multinomial(belief.cpu(), 1, generator=_rng(seed, "condition")).to(device)
    noise = _noise((1, 8, 18), seed, "proposal", device)
    cache = {}
    if method == "scaled":
        original = engine.predict(history, condition, noise=noise)
        actions = original["actions"].repeat(3, 1, 1)
        actions[:, 0] = (actions[:, 0] * torch.tensor(FORCE_SCALES, device=device)[:, None]).clamp(
            -1, 1
        )
        original_index = 1
    else:
        original = engine.joint_sample(history, condition, noise=noise)
        context = {key: value.expand(2, *value.shape[1:]).clone() for key, value in history.items()}
        generator = torch.Generator(device="cpu").manual_seed(int(seed) ^ EXTRA_NOISE_TAGS[method])
        extra_noise = torch.randn(2, 8, 18, generator=generator).to(device)
        if method == "independent":
            extra = engine.joint_sample(context, condition.expand(2), noise=extra_noise)
        else:
            extra = engine.refine_joint(
                context,
                condition.expand(2),
                original["clean_joint_tokens"].expand(2, 8, 18),
                noise=extra_noise,
            )
            cache.update(
                refinement_forward_noised_tokens=extra["forward_noised_tokens"],
                refinement_time_grid=extra["refinement_time_grid"],
            )
        actions = torch.cat((original["actions"], extra["actions"]))
        original_index = 0
        cache.update(
            proposal_clean_joint_tokens=torch.cat(
                (original["clean_joint_tokens"], extra["clean_joint_tokens"])
            ),
            proposal_joint_observations=torch.cat(
                (original["joint_observations"], extra["joint_observations"])
            ),
            proposal_joint_rewards=torch.cat((original["joint_rewards"], extra["joint_rewards"])),
            original_initial_noise=noise,
            additional_noise=extra_noise,
        )
    _finite(actions, "candidate bank")
    if actions.shape != (3, 4, 6) or (actions.abs() > 1).any():
        raise ValueError("Three bounded H4 candidate sequences are required")
    return dict(
        candidate_actions=actions,
        original_index=original_index,
        sampled_condition=int(condition[0]),
        proposal_method=method,
        first_action_distance_from_original=torch.linalg.vector_norm(
            actions[:, 0] - actions[original_index, 0], dim=-1
        ),
        full_sequence_distance_from_original=torch.linalg.vector_norm(
            (actions - actions[original_index]).flatten(1), dim=-1
        ),
        **cache,
    )

@torch.inference_mode()
def evaluate_candidates(
    engine, history, belief, candidates, *, temperature, seed, information_horizon
):
    if information_horizon not in (0, 1):
        raise ValueError("Only frozen passive/H1 selection is supported")
    device = engine.device
    history = public_history(history, device)
    belief = _belief(belief, device)
    candidates = _tensor(candidates, device)
    if len(history["mask"]) != 1 or belief.shape != (8,) or candidates.shape != (3, 4, 6):
        raise ValueError("Scoring requires one root, eight hypotheses and three H4 candidates")
    if not math.isfinite(temperature) or temperature <= 0 or (candidates.abs() > 1).any():
        raise ValueError("Invalid evidence temperature or bounded candidates")
    context = {key: value.expand(24, *value.shape[1:]).clone() for key, value in history.items()}
    actions = candidates[:, None].expand(3, 8, 4, 6).reshape(24, 4, 6)
    noise = _noise((1, 8, 18), seed, "prediction", device)
    prediction = engine.predict(
        context,
        torch.arange(8, device=device).repeat(3),
        actions=actions,
        noise=noise.expand(24, 8, 18),
    )
    observations = prediction["observations"].reshape(3, 8, 4, 17)
    rewards = prediction["rewards"].reshape(3, 8, 4)
    probability = prediction["termination_probability"].reshape(3, 8, 4)
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid modeled termination probability")
    uniforms = torch.rand(1, 1, 4, generator=_rng(seed, "termination")).to(device).expand(3, 8, 4)
    terminated = uniforms < probability
    valid = torch.cat(
        (torch.ones_like(terminated[..., :1]), (~terminated[..., :-1]).cumprod(-1).bool()), -1
    )
    returns = (rewards * valid * 0.99 ** torch.arange(4, device=device)).sum(-1)
    result = dict(
        expected_reward=(returns * belief[None]).sum(-1),
        information_h1=None,
        information_h4=None,
        information_horizon_computed=information_horizon,
        observations=observations,
        rewards=rewards,
        termination_probability=probability,
        termination_uniforms=uniforms,
        terminated=terminated & valid,
        valid=valid,
        candidate_actions=candidates.clone(),
        belief=belief.clone(),
        fixed_condition_during_imagination=True,
        predictive_futures=1,
    )
    if information_horizon:
        posteriors = belief.expand(24, 8).clone()
        before = posteriors.reshape(3, 8, 8).clone()
        indices = (valid[:, :, 0] & (belief[None] > 0)).reshape(24).nonzero().flatten()
        error_cache = torch.zeros(24, 1, 8, device=device)
        queried = torch.zeros(24, 1, dtype=torch.bool, device=device)
        if len(indices):
            noise = _noise((1, 4, 2, 18), seed, "imagination", device).expand(24, 4, 2, 18)
            errors = engine.denoising_errors(
                {key: value[indices] for key, value in context.items()},
                actions[indices, 0],
                observations.reshape(24, 4, 17)[indices, 0],
                rewards.reshape(24, 4)[indices, 0],
                noise=noise[indices],
            )
            error_cache[indices, 0], queried[indices, 0] = errors, True
            # No hazard here; the imagined physical condition is held fixed.
            posteriors[indices] = categorical_update(posteriors[indices], -errors / temperature)
        after = posteriors.reshape(3, 8, 8)
        result.update(
            information_h1=entropy(belief) - (entropy(after) * belief[None]).sum(-1),
            posterior_traces=torch.stack((before, after), 2),
            denoising_errors=error_cache.reshape(3, 8, 1, 8),
            evidence_queried=queried.reshape(3, 8, 1),
        )
    _finite(result, "candidate evaluation")
    return result

class OptimizedProposalController:
    """The original pilot state machine, with frozen surrogate evidence on device."""

    def __init__(
        self,
        engine,
        proposal_method,
        *,
        selection="passive",
        temperature,
        information_weight=0.0,
        prior=None,
    ):
        if f"{proposal_method}_{selection}" not in METHODS:
            raise ValueError("Unknown optimized method")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Evidence temperature must be positive")
        if not math.isfinite(information_weight) or information_weight < 0:
            raise ValueError("Information weight must be finite and nonnegative")
        if selection == "h1" and information_weight <= 0:
            raise ValueError("H1 selection requires its positive fixed weight")
        self.engine, self.proposal_method, self.selection = engine, proposal_method, selection
        self.temperature = float(temperature)
        self.information_weight = float(information_weight) if selection == "h1" else 0.0
        value = torch.full((8,), 1 / 8) if prior is None else prior
        self.belief = _tensor(value, engine.device).clone()
        if self.belief.shape != (8,) or not torch.isclose(
            self.belief.sum(), self.belief.new_tensor(1.0), atol=1e-6, rtol=0
        ):
            raise ValueError("Initial belief must be normalized over eight conditions")
        _probability(self.belief)
        self._pending = None

    @torch.inference_mode()
    def decide(self, history, *, seed):
        if self._pending is not None:
            raise RuntimeError("Observe the preceding actual transition before replanning")
        public = public_history(history, self.engine.device)
        prior = predict_prior(self.belief)
        bank = generate_candidate_bank(
            self.engine, public, prior, method=self.proposal_method, seed=seed
        )
        result = evaluate_candidates(
            self.engine,
            public,
            prior,
            bank["candidate_actions"],
            temperature=self.temperature,
            seed=seed,
            information_horizon=int(self.selection == "h1"),
        )
        score = result["expected_reward"].clone()
        if self.information_weight:
            score += self.information_weight * result["information_h1"]
        _finite(score, "selection score")
        selected = int(score.argmax())
        action = bank["candidate_actions"][selected, 0].clone()
        self._pending = dict(prior=prior.clone(), history=public, action=action, seed=int(seed))
        result.update(bank)
        result.update(
            prior=prior,
            selected_index=selected,
            selected=selected,
            selected_action=action,
            selection_score=score,
            query_seed=int(seed),
            selected_force_scale=FORCE_SCALES[selected]
            if self.proposal_method == "scaled"
            else 1.0,
            force_scale_applicable=self.proposal_method == "scaled",
            selected_original_action_l2=float(
                bank["first_action_distance_from_original"][selected]
            ),
        )
        return result

    @torch.inference_mode()
    def observe(self, history, action, next_observation, reward, *, seed):
        if self._pending is None:
            raise RuntimeError("A real transition requires its preceding decision")
        pending, device = self._pending, self.engine.device
        public = public_history(history, device)
        action = _tensor(action, device).reshape(1, 6)
        if any(not torch.equal(public[key], pending["history"][key]) for key in public):
            raise ValueError("Actual history differs from the decision history")
        if not torch.allclose(action[0], pending["action"], atol=1e-6, rtol=0):
            raise ValueError("Evidence must use the selected executed action")
        if int(seed) != pending["seed"]:
            raise ValueError("Actual evidence seed differs from its decision seed")
        errors = self.engine.denoising_errors(
            public,
            action,
            _tensor(next_observation, device).reshape(1, 17),
            _tensor(reward, device).reshape(1),
            noise=_noise((1, 4, 2, 18), seed, "actual", device),
        )
        log_evidence = -errors / self.temperature
        posterior = categorical_update(pending["prior"][None], log_evidence)[0]
        self.belief = posterior.clone()
        self._pending = None
        return dict(
            prior=pending["prior"],
            posterior=posterior,
            belief=posterior,
            log_evidence=log_evidence[0],
            log_likelihood=log_evidence[0],
            denoising_errors=errors[0],
        )

def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: _numpy(item) for key, item in value.items()}
    return value

class OptimizedProposalPolicy:
    """Runner adapter; real reward is taken from the appended public history."""

    receives_truth = False
    records_beliefs = True
    history_length = 8

    def __init__(
        self,
        method,
        engine,
        seed,
        *,
        temperature=FROZEN_TEMPERATURE,
        information_weight=FROZEN_INFORMATION_WEIGHT,
    ):
        if (
            method not in METHODS
            or isinstance(seed, bool)
            or not isinstance(seed, Integral)
            or seed < 0
        ):
            raise ValueError("Unknown method or invalid case seed")
        proposal, selection = method.rsplit("_", 1)
        self.method, self.engine, self.seed = method, engine, int(seed)
        self.device, self.model = engine.device, engine.model
        self.controller = OptimizedProposalController(
            engine,
            proposal,
            selection=selection,
            temperature=temperature,
            information_weight=information_weight,
        )
        self._step = 0

    @property
    def belief(self):
        return self.controller.belief

    def recipe(self):
        return dict(
            schema_version=1,
            adapter="optimized_proposal_fp32",
            method=self.method,
            history=8,
            horizon=4,
            execute_actions=1,
            candidate_count=3,
            sampling_steps=8,
            diffusion_temperature=1.0,
            evidence_temperature=self.controller.temperature,
            information_weight=self.controller.information_weight,
            real_hazard=0.06,
            imagined_hazard=0.0,
            discount=0.99,
            evidence_variables=["next_observations", "rewards"],
            probe_times=list(PROBE_TIMES),
            refinement_schedule_index=4,
            refinement_time=0.4995,
            prediction_replicates=1,
            original_retained=True,
            approximate_evidence=True,
            information="approximate expected H1 entropy reduction; negative retained",
            normalized_observation_likelihood=False,
            receives_truth=False,
            weights_frozen=True,
            dtype="float32",
            autocast=False,
            tf32=False,
            noise=(
                "sampling_seed + native_step; original CPU keyed RNG draws "
                "transferred to model device"
            ),
            query_batch_size=self.engine.batch_size,
            latency="caller-synchronized decision and actual observation update separately",
        )

    to_spec = recipe

    def act(self, history, step, oracle_condition=None):
        if oracle_condition is not None:
            raise ValueError("Optimized hidden controller must not receive evaluator truth")
        if (
            isinstance(step, (bool, np.bool_))
            or not isinstance(step, Integral)
            or step != self._step
        ):
            raise ValueError("Act step must be the next consecutive native transition")
        result = self.controller.decide(history, seed=self.seed + int(step))
        return _numpy(result["selected_action"]), _numpy(result)

    def observe(self, action, next_observation, updated_history):
        if self.controller._pending is None:
            raise RuntimeError("Observe requires a pending actual decision")
        public = public_history(updated_history, self.device)
        actual = _tensor(action, self.device)
        observed = _tensor(next_observation, self.device)
        if len(public["mask"]) != 1 or not torch.allclose(
            public["actions"][0, -1], actual, atol=1e-6, rtol=0
        ):
            raise ValueError("Appended public history does not contain the actual action")
        if not torch.equal(public["observations"][0, -1], observed):
            raise ValueError("Appended public history does not contain the actual observation")
        pending = self.controller._pending
        result = self.controller.observe(
            pending["history"], actual, observed, public["rewards"][0, -1], seed=pending["seed"]
        )
        self._step += 1
        return _numpy(result)

def optimized_policy_actions(policies, histories, steps, oracle_conditions=None):
    """Independent-case dispatch; each decision already batches hypotheses/candidates."""
    oracle_conditions = [None] * len(policies) if oracle_conditions is None else oracle_conditions
    if not (len(policies) == len(histories) == len(steps) == len(oracle_conditions)):
        raise ValueError("Decision batch lengths differ")
    if len({id(policy) for policy in policies}) != len(policies):
        raise ValueError("Each batch entry needs a distinct policy")
    return [
        policy.act(history, step, truth)
        for policy, history, step, truth in zip(
            policies, histories, steps, oracle_conditions, strict=True
        )
    ]
