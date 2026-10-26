"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import copy

import math

from dataclasses import asdict, dataclass

from numbers import Integral

import numpy as np

import torch

from .cpu_denoising_control import FORCE_SCALES, _rng

from .cpu_denoising_evidence import PROBE_TIMES

from .cpu_proposal_control import EXTRA_NOISE_TAGS, REFINEMENT_INDEX, SAMPLING_STEPS

from .optimized_proposal_control import (
    FROZEN_INFORMATION_WEIGHT,
    FROZEN_TEMPERATURE,
    OptimizedProposalController,
    OptimizedProposalEngine,
    OptimizedProposalPolicy,
    _belief,
    _finite,
    _noise,
    _numpy,
    _tensor,
    categorical_update,
    entropy,
    optimized_policy_actions,
    predict_prior,
    public_history,
)

from .optimized_proposal_control import (
    evaluate_candidates as h4_evaluate_candidates,
)

from .optimized_proposal_control import (
    generate_candidate_bank as h4_generate_candidate_bank,
)

from .robust_observation_law import GaussianResidual

@dataclass(frozen=True)
class MethodSpec:
    horizon: int
    evidence: str
    proposal: str
    information_horizon: int

SPECIFICATIONS = {
    "sample_h8_reward": MethodSpec(8, "sample_gaussian", "scaled", 0),
    "denoise_h8_reward": MethodSpec(8, "denoising", "scaled", 0),
    "denoise_h8_h1": MethodSpec(8, "denoising", "scaled", 1),
    "denoise_h8_h8info": MethodSpec(8, "denoising", "scaled", 8),
    "refine_h4_reward": MethodSpec(4, "denoising", "refinement", 0),
    "refine_h4_h1": MethodSpec(4, "denoising", "refinement", 1),
    "refine_h8_reward": MethodSpec(8, "denoising", "refinement", 0),
    "refine_h8_h1": MethodSpec(8, "denoising", "refinement", 1),
}

def _calibration(value):
    if (
        not isinstance(value, dict)
        or value.get("family") != "cpu_matched_gaussian_observation_reward"
        or value.get("split") != "CAL"
        or value.get("feature_count") != 18
    ):
        raise ValueError("The existing CAL observation/reward Gaussian calibration is required")
    extra = torch.tensor(value["extra_variance"], dtype=torch.float32)
    if extra.shape != (18,) or not torch.isfinite(extra).all() or (extra <= 0).any():
        raise ValueError("CAL residual variance must contain 18 positive finite coordinates")
    law = GaussianResidual.from_spec(value["residual_law"])
    if len(law.scale_squared) != 18:
        raise ValueError("CAL Gaussian law must contain observation and reward coordinates")
    return copy.deepcopy(value), law

def method_recipe(
    method,
    *,
    gaussian_calibration=None,
    temperature=FROZEN_TEMPERATURE,
    information_weight=FROZEN_INFORMATION_WEIGHT,
    query_batch_size=256,
):
    if method not in SPECIFICATIONS:
        raise ValueError("Unknown unified comparison method")
    spec = SPECIFICATIONS[method]
    calibration = (
        _calibration(gaussian_calibration)[0] if spec.evidence == "sample_gaussian" else None
    )
    return dict(
        schema_version=1,
        adapter="unified_comparison_fp32",
        method=method,
        **asdict(spec),
        history_length=8,
        execute_actions=1,
        candidate_count=3,
        prediction_replicates=1,
        prediction_hypotheses=8,
        sampling_steps=8,
        diffusion_temperature=1.0,
        evidence_temperature=temperature if spec.evidence == "denoising" else None,
        evidence_particles=16 if spec.evidence == "sample_gaussian" else 0,
        probe_times=list(PROBE_TIMES) if spec.evidence == "denoising" else [],
        gaussian_calibration=calibration,
        information_weight=information_weight if spec.information_horizon else 0.0,
        real_hazard=0.06,
        imagined_hazard=0.0,
        discount=0.99,
        evidence_variables=["next_observations", "rewards"],
        refinement_schedule_index=4 if spec.proposal == "refinement" else None,
        refinement_time=0.4995 if spec.proposal == "refinement" else None,
        original_retained=True,
        normalized_observation_likelihood=spec.evidence == "sample_gaussian",
        exact_diffusion_likelihood=False,
        information_target="Terminal entropy reduction over a fixed-action imagined prefix",
        future_feedback_replanning=False,
        receives_truth=False,
        weights_frozen=True,
        dtype="float32",
        autocast=False,
        tf32=False,
        query_batch_size=query_batch_size,
        noise="Original CPU keyed streams; sampling_seed + native_step; horizon changes shapes",
        h4_implementation="Unchanged optimized controller operations"
        if spec.horizon == 4
        else None,
        latency="Caller synchronizes decision and actual evidence update separately",
    )

class UnifiedComparisonEngine(OptimizedProposalEngine):
    """Same frozen FP32 denoiser, generalized full sampling at H1/H4/H8."""

    def __init__(self, model, batch_size=256):
        if model.config.max_horizon < 8:
            raise ValueError("The unified comparison requires the existing H8-capable model")
        super().__init__(model, batch_size=batch_size)

    @torch.inference_mode()
    def joint_sample(self, history, condition, *, noise, clean_joint_tokens=None, horizon=4):
        if horizon == 4:
            return super().joint_sample(
                history, condition, noise=noise, clean_joint_tokens=clean_joint_tokens
            )
        if horizon != 8:
            raise ValueError("Joint proposals support only H4/H8")
        history = public_history(history, self.device)
        n = len(history["mask"])
        condition = self._conditions(condition, n)
        noise = _tensor(noise, self.device)
        if noise.shape != (n, 2 * horizon, 18):
            raise ValueError("Joint noise must cover every action/observation token")
        if clean_joint_tokens is not None:
            clean_joint_tokens = _tensor(clean_joint_tokens, self.device)
            if clean_joint_tokens.shape != noise.shape:
                raise ValueError("Refinement requires the complete original joint trajectory")
        grid = torch.linspace(0.999, 0, SAMPLING_STEPS + 1, device=self.device)
        if clean_joint_tokens is not None:
            grid = grid[REFINEMENT_INDEX:]
        parts = {}
        with self._fp32():
            for start in range(0, n, self.batch_size):
                stop = min(start + self.batch_size, n)
                context = {key: value[start:stop] for key, value in history.items()}
                encoded = self._encode(context, condition[start:stop], horizon)
                x = noise[start:stop]
                if clean_joint_tokens is not None:
                    x = (
                        torch.cos(grid[0] * math.pi / 2) * clean_joint_tokens[start:stop]
                        + torch.sin(grid[0] * math.pi / 2) * x
                    )
                x = torch.where(encoded["known"], encoded["x"], x) * encoded["semantic"]
                result = self._finish(x, encoded, context, grid)
                if clean_joint_tokens is not None:
                    result["forward_noised_tokens"] = x
                for key, value in result.items():
                    parts.setdefault(key, []).append(value)
        result = {key: torch.cat(value) for key, value in parts.items()}
        if clean_joint_tokens is not None:
            result["refinement_time_grid"] = grid
        _finite(result, "joint_sample")
        return result

    def _conditions(self, value, n):
        value = torch.as_tensor(value, device=self.device)
        if value.shape != (n,) or value.dtype != torch.long or ((value < 0) | (value > 7)).any():
            raise ValueError("One integer condition hypothesis in [0,7] is required per query")
        return value

    @torch.inference_mode()
    def predict(self, history, condition, actions=None, *, horizon=4, noise):
        if horizon == 4:
            return super().predict(history, condition, actions, horizon=horizon, noise=noise)
        if horizon not in (1, 8):
            raise ValueError("Only H1 evidence and H4/H8 trajectory queries are supported")
        history = public_history(history, self.device)
        n = len(history["mask"])
        condition = self._conditions(condition, n)
        noise = _tensor(noise, self.device)
        if noise.shape != (n, 2 * horizon, 18):
            raise ValueError("Prediction noise must cover the requested trajectory")
        if actions is not None:
            actions = _tensor(actions, self.device)
            if actions.shape != (n, horizon, 6) or (actions.abs() > 1).any():
                raise ValueError("Imposed actions must match the bounded prediction horizon")
        parts = {}
        with self._fp32():
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
                for key, value in result.items():
                    parts.setdefault(key, []).append(value)
        result = {key: torch.cat(value) for key, value in parts.items()}
        _finite(result, "prediction")
        return result

    @torch.inference_mode()
    def normalize_outcomes(self, history, observations, rewards):
        history = public_history(history, self.device)
        n, device = len(history["mask"]), self.device
        encoded = self._encode(
            history,
            torch.zeros(n, dtype=torch.long, device=device),
            1,
            torch.zeros(n, 1, 6, device=device),
            _tensor(observations, device).reshape(n, 1, 17),
            _tensor(rewards, device).reshape(n, 1),
        )
        result = encoded["x"][:, 1].clone()
        _finite(result, "normalized observation/reward")
        return result

    @torch.inference_mode()
    def gaussian_moments(self, history, actions, *, noise):
        history = public_history(history, self.device)
        n, device = len(history["mask"]), self.device
        actions = _tensor(actions, device).reshape(n, 1, 6)
        noise = _tensor(noise, device)
        if noise.shape != (n, 16, 2, 18):
            raise ValueError("Sample evidence requires exactly 16 H1 particles per hypothesis")
        index = torch.arange(n, device=device).repeat_interleave(8 * 16)
        expanded = {key: value[index] for key, value in history.items()}
        prediction = self.predict(
            expanded,
            torch.arange(8, device=device).repeat_interleave(16).repeat(n),
            actions[index],
            horizon=1,
            noise=noise[:, None].expand(n, 8, 16, 2, 18).reshape(-1, 2, 18),
        )
        outcomes = self.normalize_outcomes(
            expanded, prediction["observations"], prediction["rewards"]
        ).reshape(n, 8, 16, 18)
        result = dict(mean=outcomes.mean(2), variance=outcomes.var(2, unbiased=False))
        _finite(result, "Gaussian predictive moments")
        return result

@torch.inference_mode()
def generate_candidate_bank(engine, history, belief, *, method, seed, horizon):
    if horizon == 4:
        return h4_generate_candidate_bank(engine, history, belief, method=method, seed=seed)
    if horizon != 8 or method not in ("scaled", "refinement"):
        raise ValueError("Expected a frozen scaled/refined H8 proposal recipe")
    device = engine.device
    history, belief = public_history(history, device), _belief(belief, device)
    if len(history["mask"]) != 1:
        raise ValueError("A candidate bank uses one public root")
    condition = torch.multinomial(belief.cpu(), 1, generator=_rng(seed, "condition")).to(device)
    noise = _noise((1, 2 * horizon, 18), seed, "proposal", device)
    if method == "scaled":
        original = engine.predict(history, condition, horizon=horizon, noise=noise)
        actions = original["actions"].repeat(3, 1, 1)
        actions[:, 0] = (actions[:, 0] * actions.new_tensor(FORCE_SCALES)[:, None]).clamp(-1, 1)
        original_index, cache = 1, {}
    else:
        original = engine.joint_sample(history, condition, horizon=horizon, noise=noise)
        extra_noise = torch.randn(
            2,
            2 * horizon,
            18,
            generator=torch.Generator(device="cpu").manual_seed(seed ^ EXTRA_NOISE_TAGS[method]),
        ).to(device)
        extra = engine.joint_sample(
            {key: value.expand(2, *value.shape[1:]).clone() for key, value in history.items()},
            condition.expand(2),
            horizon=horizon,
            noise=extra_noise,
            clean_joint_tokens=original["clean_joint_tokens"].expand(2, 2 * horizon, 18),
        )
        actions = torch.cat((original["actions"], extra["actions"]))
        original_index = 0
        cache = dict(
            refinement_forward_noised_tokens=extra["forward_noised_tokens"],
            refinement_time_grid=extra["refinement_time_grid"],
            proposal_clean_joint_tokens=torch.cat(
                (original["clean_joint_tokens"], extra["clean_joint_tokens"])
            ),
            original_initial_noise=noise,
            additional_noise=extra_noise,
        )
    if actions.shape != (3, horizon, 6) or (actions.abs() > 1).any():
        raise ValueError("Exactly three bounded candidate trajectories are required")
    _finite(actions, "candidate bank")
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

def _append(history, action, observation, reward):
    return dict(
        observations=torch.cat((history["observations"][:, 1:], observation[:, None]), 1),
        actions=torch.cat((history["actions"][:, 1:], action[:, None]), 1),
        rewards=torch.cat((history["rewards"][:, 1:], reward[:, None]), 1),
        mask=torch.cat((history["mask"][:, 1:], torch.ones_like(reward[:, None])), 1),
    )

@torch.inference_mode()
def evaluate_candidates(
    engine, history, belief, candidates, *, temperature, seed, horizon, information_horizon
):
    if horizon == 4 and information_horizon in (0, 1):
        result = h4_evaluate_candidates(
            engine,
            history,
            belief,
            candidates,
            temperature=temperature,
            seed=seed,
            information_horizon=information_horizon,
        )
        result["information"] = result["information_h1"]
        return result
    if horizon not in (4, 8) or information_horizon not in (0, 1, horizon):
        raise ValueError("Information must use no observation, H1, or the planning horizon")
    device = engine.device
    history, belief = public_history(history, device), _belief(belief, device)
    candidates = _tensor(candidates, device)
    if len(history["mask"]) != 1 or candidates.shape != (3, horizon, 6):
        raise ValueError("Scoring requires one root and three planning-horizon sequences")
    if not math.isfinite(temperature) or temperature <= 0 or (candidates.abs() > 1).any():
        raise ValueError("Invalid evidence temperature or bounded candidates")
    context = {key: value.expand(24, *value.shape[1:]).clone() for key, value in history.items()}
    actions = candidates[:, None].expand(3, 8, horizon, 6).reshape(24, horizon, 6)
    prediction = engine.predict(
        context,
        torch.arange(8, device=device).repeat(3),
        actions=actions,
        horizon=horizon,
        noise=_noise((1, 2 * horizon, 18), seed, "prediction", device).expand(24, 2 * horizon, 18),
    )
    observations = prediction["observations"].reshape(3, 8, horizon, 17)
    rewards = prediction["rewards"].reshape(3, 8, horizon)
    probability = prediction["termination_probability"].reshape(3, 8, horizon)
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid modeled termination probability")
    uniforms = torch.rand(1, 1, horizon, generator=_rng(seed, "termination")).to(device)
    uniforms = uniforms.expand(3, 8, horizon)
    terminated = uniforms < probability
    valid = torch.cat(
        (torch.ones_like(terminated[..., :1]), (~terminated[..., :-1]).cumprod(-1).bool()), -1
    )
    returns = (rewards * valid * 0.99 ** torch.arange(horizon, device=device)).sum(-1)
    result = dict(
        expected_reward=(returns * belief[None]).sum(-1),
        information=None,
        information_h1=None,
        information_h4=None,
        information_h8=None,
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
        traces = [posteriors.reshape(3, 8, 8).clone()]
        errors_cache = torch.zeros(24, information_horizon, 8, device=device)
        queried = torch.zeros(24, information_horizon, dtype=torch.bool, device=device)
        for step in range(information_horizon):
            indices = (valid[:, :, step] & (belief[None] > 0)).reshape(24).nonzero().flatten()
            if len(indices):
                branch = {key: value[indices] for key, value in context.items()}
                action = actions[indices, step]
                observation = observations.reshape(24, horizon, 17)[indices, step]
                reward = rewards.reshape(24, horizon)[indices, step]
                noise = torch.randn(1, 4, 2, 18, generator=_rng(seed, "imagination", step))
                noise = noise.to(device).expand(24, 4, 2, 18)
                errors = engine.denoising_errors(
                    branch, action, observation, reward, noise=noise[indices]
                )
                errors_cache[indices, step], queried[indices, step] = errors, True
                posteriors[indices] = categorical_update(posteriors[indices], -errors / temperature)
                updated = _append(branch, action, observation, reward)
                for key in context:
                    context[key][indices] = updated[key]
            traces.append(posteriors.reshape(3, 8, 8).clone())
            information = entropy(belief) - (entropy(traces[-1]) * belief[None]).sum(-1)
            if step == 0:
                result["information_h1"] = information
            if step == horizon - 1:
                result[f"information_h{horizon}"] = information
        result.update(
            information=information,
            posterior_traces=torch.stack(traces, 2),
            denoising_errors=errors_cache.reshape(3, 8, information_horizon, 8),
            evidence_queried=queried.reshape(3, 8, information_horizon),
        )
    _finite(result, "candidate evaluation")
    return result

class UnifiedComparisonController(OptimizedProposalController):
    def __init__(
        self,
        engine,
        method,
        *,
        gaussian_calibration=None,
        temperature=FROZEN_TEMPERATURE,
        information_weight=FROZEN_INFORMATION_WEIGHT,
        prior=None,
    ):
        if method not in SPECIFICATIONS:
            raise ValueError("Unknown unified comparison method")
        self.method, self.spec = method, SPECIFICATIONS[method]
        super().__init__(
            engine,
            self.spec.proposal,
            selection="h1" if self.spec.information_horizon else "passive",
            temperature=temperature,
            information_weight=information_weight,
            prior=prior,
        )
        self.gaussian_calibration, self.gaussian_law = (
            _calibration(gaussian_calibration)
            if self.spec.evidence == "sample_gaussian"
            else (None, None)
        )

    @torch.inference_mode()
    def decide(self, history, *, seed):
        if self.spec.horizon == 4:
            result = super().decide(history, seed=seed)
            result["information"] = result["information_h1"]
            return result
        if self._pending is not None:
            raise RuntimeError("Observe the preceding actual transition before replanning")
        public = public_history(history, self.engine.device)
        prior = predict_prior(self.belief)
        bank = generate_candidate_bank(
            self.engine,
            public,
            prior,
            method=self.spec.proposal,
            seed=seed,
            horizon=self.spec.horizon,
        )
        result = evaluate_candidates(
            self.engine,
            public,
            prior,
            bank["candidate_actions"],
            temperature=self.temperature,
            seed=seed,
            horizon=self.spec.horizon,
            information_horizon=self.spec.information_horizon,
        )
        score = result["expected_reward"].clone()
        if self.information_weight:
            score += self.information_weight * result["information"]
        _finite(score, "selection score")
        selected = int(score.argmax())
        action = bank["candidate_actions"][selected, 0].clone()
        moments = None
        if self.gaussian_law is not None:
            moments = self.engine.gaussian_moments(
                public,
                action[None],
                noise=_noise((1, 16, 2, 18), seed, "gaussian", self.engine.device),
            )
        self._pending = dict(
            prior=prior.clone(), history=public, action=action, seed=int(seed), moments=moments
        )
        result.update(bank)
        result.update(
            prior=prior,
            selected_index=selected,
            selected=selected,
            selected_action=action,
            selection_score=score,
            query_seed=int(seed),
            selected_force_scale=FORCE_SCALES[selected] if self.spec.proposal == "scaled" else 1.0,
            force_scale_applicable=self.spec.proposal == "scaled",
            selected_original_action_l2=float(
                bank["first_action_distance_from_original"][selected]
            ),
        )
        if moments is not None:
            result["executed_action_gaussian_moments"] = moments
        return result

    @torch.inference_mode()
    def observe(self, history, action, next_observation, reward, *, seed):
        if self.gaussian_law is None:
            return super().observe(history, action, next_observation, reward, seed=seed)
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
        outcome = self.engine.normalize_outcomes(public, next_observation, reward)
        moments = pending["moments"]
        base = moments["variance"] + moments["variance"].new_tensor(
            self.gaussian_calibration["extra_variance"]
        )
        log_evidence = self.gaussian_law.log_prob(outcome[:, None], moments["mean"], base)
        posterior = categorical_update(pending["prior"][None], log_evidence)[0]
        self.belief = posterior.clone()
        self._pending = None
        return dict(
            prior=pending["prior"],
            posterior=posterior,
            belief=posterior,
            log_evidence=log_evidence[0],
            log_likelihood=log_evidence[0],
            normalized_outcome=outcome[0],
        )

class UnifiedComparisonPolicy(OptimizedProposalPolicy):
    def __init__(
        self,
        method,
        engine,
        seed,
        *,
        gaussian_calibration=None,
        temperature=FROZEN_TEMPERATURE,
        information_weight=FROZEN_INFORMATION_WEIGHT,
    ):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("A nonnegative integer case seed is required")
        self.method, self.engine, self.seed = method, engine, int(seed)
        self.device, self.model = engine.device, engine.model
        self.controller = UnifiedComparisonController(
            engine,
            method,
            gaussian_calibration=gaussian_calibration,
            temperature=temperature,
            information_weight=information_weight,
        )
        self._step = 0

    def recipe(self):
        return method_recipe(
            self.method,
            gaussian_calibration=self.controller.gaussian_calibration,
            temperature=self.controller.temperature,
            information_weight=self.controller.information_weight,
            query_batch_size=self.engine.batch_size,
        )

    to_spec = recipe

class UnifiedOraclePolicy:
    """Privileged current-label H8 K1 sampler; no future schedule or ranking."""

    receives_truth = True
    records_beliefs = False
    history_length = 8
    method = "oracle"

    def __init__(self, engine, seed):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("A nonnegative integer case seed is required")
        self.engine, self.model, self.device = engine, engine.model, engine.device
        self.seed, self._step, self._pending = int(seed), 0, None

    def recipe(self):
        return dict(
            adapter="unified_oracle_fp32",
            method="oracle",
            horizon=8,
            execute_actions=1,
            proposal_draws=1,
            cost_queries=0,
            evidence_queries=0,
            information_queries=0,
            receives_truth=True,
            truth_scope="current condition only",
            history_length=8,
            sampling_steps=8,
            diffusion_temperature=1.0,
            dtype="float32",
            autocast=False,
            tf32=False,
            noise="sampling_seed + native_step; CPU keyed proposal stream",
        )

    to_spec = recipe

    @torch.inference_mode()
    def act(self, history, step, oracle_condition=None):
        if (
            isinstance(step, (bool, np.bool_))
            or not isinstance(step, Integral)
            or step != self._step
            or isinstance(oracle_condition, (bool, np.bool_))
            or not isinstance(oracle_condition, Integral)
            or not 0 <= oracle_condition < 8
        ):
            raise ValueError("Oracle requires a consecutive step and current integer truth")
        if self._pending is not None:
            raise RuntimeError("Observe the preceding actual transition before replanning")
        public = public_history(history, self.device)
        if len(public["mask"]) != 1:
            raise ValueError("Oracle receives one public history")
        sample = self.engine.predict(
            public,
            torch.tensor([oracle_condition], device=self.device),
            horizon=8,
            noise=_noise((1, 16, 18), self.seed + step, "proposal", self.device),
        )
        action = sample["actions"][0, 0].clone()
        if action.shape != (6,) or (action.abs() > 1).any():
            raise ValueError("Oracle proposal must be a bounded native action")
        self._pending = action
        return _numpy(action), dict(current_condition=int(oracle_condition), proposal_draws=1)

    def observe(self, action, next_observation, updated_history):
        if self._pending is None:
            raise RuntimeError("Observe requires a pending actual decision")
        public = public_history(updated_history, self.device)
        action, observation = _tensor(action, self.device), _tensor(next_observation, self.device)
        if (
            len(public["mask"]) != 1
            or action.shape != (6,)
            or observation.shape != (17,)
            or not torch.allclose(action, self._pending, atol=1e-6, rtol=0)
            or not torch.allclose(public["actions"][0, -1], action, atol=1e-6, rtol=0)
            or not torch.equal(public["observations"][0, -1], observation)
        ):
            raise ValueError("Oracle observation must match the executed public transition")
        self._pending = None
        self._step += 1
        return {}
