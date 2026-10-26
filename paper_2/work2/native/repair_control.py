from __future__ import annotations


import json

import math

from dataclasses import asdict

from numbers import Integral

import torch

from .control import BeliefController, ControlConfig, predictions_batch, proposals_batch

from .control import _evidence_batch as base_evidence_batch

from .inference import expected_task_cost, gaussian_log_prob, mutual_information, update_belief

PUBLIC_HISTORY_KEYS = {"observations", "actions", "rewards", "mask"}

POOLS = ("full", "generated_only")

class OriginalObservationLaw:
    """Exact original FP32 Gaussian operations, including their RNG order."""

    @staticmethod
    def to_spec():
        return {"family": "original_diagonal_gaussian", "version": 1}

    @staticmethod
    def log_prob(observed, mean, base_variance):
        return gaussian_log_prob(observed, mean, base_variance)

    @staticmethod
    def mutual_information(mean, base_variance, prior, draws=64, *, generator=None, noise=None):
        return mutual_information(
            mean, base_variance, prior, draws=draws, generator=generator, noise=noise
        )

def _law_spec(law):
    for name in ("log_prob", "mutual_information", "to_spec"):
        if not callable(getattr(law, name, None)):
            raise ValueError(f"Observation law must implement {name}")
    spec = law.to_spec()
    if not isinstance(spec, dict):
        raise ValueError("Observation-law specification must be a JSON object")
    # Require an actual independently serializable snapshot, not a mutable tensor.
    return json.loads(json.dumps(spec, allow_nan=False))

def _public_history(history):
    if set(history) != PUBLIC_HISTORY_KEYS:
        raise ValueError("Only public observation/action/reward history and mask are accepted")

def _truth(condition, privileged):
    if not privileged:
        if condition is not None:
            raise ValueError("Unprivileged controller must not receive evaluator truth")
    elif (
        isinstance(condition, bool) or not isinstance(condition, Integral) or not 0 <= condition < 8
    ):
        raise ValueError("Privileged controller requires an integer condition in [0, 7]")

def _validate_config(config):
    for name in (
        "horizon",
        "candidates",
        "task_samples",
        "evidence_samples",
        "sampling_steps",
        "information_draws",
        "chunk_size",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.candidates < 4 or config.candidates % 4:
        raise ValueError("Repair pools require a candidate count divisible by four")
    if config.information_draws < 2:
        raise ValueError("Information standard errors require at least two draws")
    for name in ("temperature", "hazard", "discount", "information_weight"):
        value = getattr(config, name)
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.temperature <= 0 or not 0 <= config.hazard <= 1 or not 0 <= config.discount <= 1:
        raise ValueError("Invalid temperature, hazard or discount")
    if config.information_weight < 0:
        raise ValueError("Information weight cannot be negative")

class RepairBeliefController(BeliefController):
    """Frozen-law full-belief controller, or explicitly privileged comparator.

    ``base_variance`` always means complete original covariance including CAL
    residual variance. A Student-t law receives it unchanged and applies its
    own location/scale parameters. No Student-t scale is treated as Gaussian
    variance. ``recipe()`` exposes the fixed law and selection rule for traces.
    """

    def __init__(
        self,
        model,
        config=None,
        *,
        mode="vfe_full",
        observation_law=None,
        pool="full",
        extra_variance=None,
        seed=13,
    ):
        config = config or ControlConfig()
        _validate_config(config)
        if mode not in ("vfe_full", "oracle") or pool not in POOLS:
            raise ValueError("Unsupported repair controller mode or candidate pool")
        self.observation_law = observation_law or OriginalObservationLaw()
        self.law_spec = _law_spec(self.observation_law)
        self.pool = pool
        self._repair_prior_route = False
        super().__init__(model, config, mode=mode, extra_variance=extra_variance, seed=seed)

    def recipe(self):
        return {
            "schema_version": 1,
            "controller": "repair_belief_controller",
            "mode": self.mode,
            "config": asdict(self.config),
            "observation_law": _law_spec(self.observation_law),
            "pool": self.pool,
            "generated_candidates": self.config.candidates * 3 // 4,
            "scored_candidates": self.config.candidates,
            "eligible_candidates": self.config.candidates
            if self.pool == "full"
            else self.config.candidates * 3 // 4,
            "pool_rule": "Generate and score the original full pool; restrict only final argmin",
            "covariance_input": "Complete original covariance, including frozen CAL residual",
            "parameters_frozen_at_deployment": True,
            "receives_truth": self.mode == "oracle",
        }

    def prior(self, history, true_condition=None):
        # Original evaluation dispatches act_batch directly, bypassing act().
        # Refuse that path rather than silently applying the wrong likelihood.
        if not self._repair_prior_route:
            raise RuntimeError("Repair controller requires repair_act_batch/dispatcher")
        return super().prior(history, true_condition)

    def base_evidence_laws(self, history, candidates):
        _public_history(history)
        return base_evidence_batch([self], [history], [candidates])[0]

    def evidence_laws(self, history, candidates):
        if not isinstance(self.observation_law, OriginalObservationLaw):
            raise RuntimeError("Use base_evidence_laws with this controller's observation law")
        return self.base_evidence_laws(history, candidates)

    @torch.no_grad()
    def act(self, history, true_condition=None):
        return repair_act_batch([self], [history], [true_condition])[0]

    @torch.no_grad()
    def observe(self, action, next_observation, updated_history=None):
        if self.cache is None:
            raise RuntimeError("No cached prediction for this executed action")
        if updated_history is not None:
            _public_history(updated_history)
        actual = torch.as_tensor(action, device=self.device)
        if actual.shape != self.cache["action"].shape or not torch.allclose(
            actual, self.cache["action"], atol=1e-6, rtol=0
        ):
            raise ValueError("Executed action differs from the action used for evidence")
        observed = torch.as_tensor(next_observation, device=self.device)
        if observed.shape != self.cache["observation"].shape or not torch.isfinite(observed).all():
            raise ValueError("Observation must have the expected finite public shape")
        feature = (
            observed - self.cache["observation"] - self.model.delta_mean
        ) / self.model.delta_std
        likelihood = self.observation_law.log_prob(
            feature, self.cache["mean"], self.cache["base_variance"]
        )
        self.belief = update_belief(self.cache["prior"], likelihood)
        self.cache = None
        return {"belief": self.belief.cpu().numpy(), "log_likelihood": likelihood.cpu().numpy()}

@torch.no_grad()
def repair_act_batch(controllers, histories, true_conditions=None):
    """Explicit law dispatch; original full-pool operations retain exact RNG order."""
    if len(controllers) != len(histories):
        raise ValueError("Each controller needs one matching public history")
    if true_conditions is None:
        true_conditions = [None] * len(controllers)
    if len(true_conditions) != len(controllers):
        raise ValueError("Each controller needs one truth-boundary entry")
    if not controllers:
        return []
    for c, h, truth in zip(controllers, histories, true_conditions, strict=True):
        if not isinstance(c, RepairBeliefController):
            raise ValueError("Use repair_act_batch only for repair controllers")
        _public_history(h)
        _truth(truth, c.mode == "oracle")
        if c._repair_prior_route or c.cache is not None:
            raise RuntimeError("Observe the previous transition before replanning")
    config = controllers[0].config
    if any(c.config.task_samples != config.task_samples for c in controllers):
        raise ValueError("Batched task predictions require a common particle count")
    priors = []
    for c, h, truth in zip(controllers, histories, true_conditions, strict=True):
        c._repair_prior_route = True
        try:
            priors.append(c.prior(h, truth))
        finally:
            c._repair_prior_route = False
    candidates = proposals_batch(controllers, histories, priors)
    hypotheses = [
        p.nonzero().flatten() if c.mode == "oracle" else None
        for c, p in zip(controllers, priors, strict=True)
    ]
    predictions = predictions_batch(
        controllers, histories, candidates, config.task_samples, config.horizon, hypotheses
    )
    active_mi = [i for i, c in enumerate(controllers) if c.config.information_weight > 0]
    laws = dict(
        zip(
            active_mi,
            base_evidence_batch(
                [controllers[i] for i in active_mi],
                [histories[i] for i in active_mi],
                [candidates[i] for i in active_mi],
            )
            if active_mi
            else [],
            strict=True,
        )
    )
    selected, details = [], []
    for i, (c, prior, prediction) in enumerate(zip(controllers, priors, predictions, strict=True)):
        weights = prior if hypotheses[i] is None else prior[hypotheses[i]]
        costs = expected_task_cost(
            prediction["rewards"], prediction["termination_probability"], weights, c.config.discount
        )
        information, error = torch.zeros_like(costs), torch.zeros_like(costs)
        if i in laws:
            information, error = c.observation_law.mutual_information(
                *laws[i], prior, draws=c.config.information_draws, generator=c.generator
            )
        objective = costs - c.config.information_weight * information
        if not torch.isfinite(objective).all() or not torch.isfinite(error).all():
            raise FloatingPointError("FATAL_CONTROLLER: nonfinite candidate score")
        eligible = len(objective) if c.pool == "full" else c.config.candidates * 3 // 4
        selected.append(int(objective[:eligible].argmin()))
        details.append(
            {
                "prior": prior.cpu().numpy(),
                "selected": selected[-1],
                "costs": costs.cpu().numpy(),
                "information": information.cpu().numpy(),
                "information_se": error.cpu().numpy(),
            }
        )
    remaining = [i for i in range(len(controllers)) if i not in laws]
    selected_laws = dict(
        zip(
            remaining,
            [
                (mean[0], variance[0])
                for mean, variance in base_evidence_batch(
                    [controllers[i] for i in remaining],
                    [histories[i] for i in remaining],
                    [candidates[i][selected[i] : selected[i] + 1] for i in remaining],
                )
            ]
            if remaining
            else [],
            strict=True,
        )
    )
    selected_laws.update(
        {i: (mean[selected[i]], variance[selected[i]]) for i, (mean, variance) in laws.items()}
    )
    output = []
    for i, c in enumerate(controllers):
        mean, variance = selected_laws[i]
        action = candidates[i][selected[i], 0]
        c.cache = {
            "prior": priors[i],
            "mean": mean,
            "base_variance": variance,
            "observation": histories[i]["observations"][0, -1].clone(),
            "action": action.clone(),
        }
        if isinstance(c.observation_law, OriginalObservationLaw):
            c.cache["variance"] = variance  # Original-law inspection/equivalence only.
        output.append((action.cpu().numpy(), details[i]))
    return output
