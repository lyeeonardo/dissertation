from __future__ import annotations


from dataclasses import asdict

from numbers import Integral

import numpy as np

import torch

from .control import _sample_requests

from .evaluation import decision_seed

from .inference import calibrated_moments, expected_task_cost

from .reference_policy_control import GENERATED_CANDIDATES, ReferenceSelectionConfig, _public_history, select_candidate

from .repair_control import RepairBeliefController

from .robust_observation_law import GaussianResidual

CONDITION_TAG = 4_410_101

PROPOSAL_TAG = 4_410_103

TASK_TAG = 4_410_107

EVIDENCE_TAG = 4_410_109

INFORMATION_TAG = 4_410_113

SELECTION_TAG = 4_410_127

METHODS = ("revised_vfe", "revised_efe")

def _generator(controller, tag):
    return torch.Generator(device=controller.device).manual_seed(controller.query_seed ^ tag)

def sample_condition_labels(prior, generator):
    """Sample the full categorical prior iid, retaining low-probability hypotheses."""
    if (
        prior.shape != (8,)
        or not torch.isfinite(prior).all()
        or (prior < 0).any()
        or not torch.isclose(prior.sum(), prior.new_tensor(1.0), atol=1e-6, rtol=0)
    ):
        raise ValueError("Reference mixture needs a normalized eight-condition belief")
    return torch.multinomial(prior, GENERATED_CANDIDATES, replacement=True, generator=generator)

class HiddenReferenceController(RepairBeliefController):
    """Exact repaired VFE update; reference generation/selection use the new dispatcher."""

    def __init__(self, model, config, *, selection, observation_law, extra_variance, seed=13):
        if not isinstance(observation_law, GaussianResidual):
            raise ValueError("The selected GaussianResidual evidence law is required")
        expected = dict(
            horizon=8,
            candidates=32,
            task_samples=8,
            evidence_samples=16,
            sampling_steps=8,
            temperature=1.0,
            hazard=0.06,
            discount=0.99,
            information_draws=64,
        )
        if any(getattr(config, key) != value for key, value in expected.items()):
            raise ValueError("Hidden reference control requires the frozen H8/belief/query recipe")
        if not isinstance(selection, ReferenceSelectionConfig):
            raise ValueError("An explicit frozen reference selection configuration is required")
        if model.config.history != 8 or not model.config.condition_input:
            raise ValueError("A conditional shared model with public history8 is required")
        super().__init__(
            model,
            config,
            mode="vfe_full",
            observation_law=observation_law,
            pool="generated_only",
            extra_variance=extra_variance,
            seed=seed,
        )
        self.selection = selection
        self.query_seed = None

    def act(self, history, true_condition=None):
        raise RuntimeError("Hidden reference control requires policy_preservation_actions")

    def recipe(self):
        return dict(
            schema_version=1,
            controller="hidden_reference_controller",
            config=asdict(self.config),
            observation_law=self.observation_law.to_spec(),
            selection=asdict(self.selection),
            selection_temperature=self.selection.temperature,
            generated_candidates=24,
            scored_candidates=24,
            eligible_candidates=24,
            reference="iid latent condition from full prior; conditional diffusion draw",
            scoring="All candidates evaluated under the same full eight-condition prior",
            information_target="Current physical condition; next executed-action observation",
            evidence="Same GaussianResidual law for H1 information and cached executed-action VFE",
            executed_prefix=1,
            history_length=8,
            receives_truth=False,
            proposal_density_reweighted=False,
            parameters_frozen_at_deployment=True,
            rng=dict(
                case_step="decision_seed(case_seed, step)",
                condition=CONDITION_TAG,
                proposal=PROPOSAL_TAG,
                task=TASK_TAG,
                evidence=EVIDENCE_TAG,
                information=INFORMATION_TAG,
                selection=SELECTION_TAG,
                stage_rule="Independent generators seeded by case-step seed XOR stage tag",
                evidence_rule="Full bank indexed noise; query only selected candidate at lambda0",
            ),
        )

@torch.no_grad()
def reference_proposals(controllers, histories, priors):
    requests, labels = [], []
    for controller, history, prior in zip(controllers, histories, priors, strict=True):
        condition = sample_condition_labels(prior, _generator(controller, CONDITION_TAG))
        context = {key: value.expand(24, *value.shape[1:]) for key, value in history.items()}
        noise = torch.randn(
            24,
            16,
            controller.model.token_width,
            device=controller.device,
            generator=_generator(controller, PROPOSAL_TAG),
        )
        requests.append((context, condition, None, noise))
        labels.append(condition)
    results = _sample_requests(controllers, requests, 8)
    banks = [result["actions"] for result in results]
    for bank in banks:
        if bank.shape != (24, 8, 6) or not torch.isfinite(bank).all() or (bank.abs() > 1).any():
            raise FloatingPointError("FATAL_CONTROLLER: invalid bounded reference bank")
    return banks, labels

@torch.no_grad()
def reference_predictions(
    controllers, histories, banks, samples, horizon, stream_tag, candidate_indices=None
):
    """Imposed-action predictions with common hypothesis draws and stable candidate indexing.

    A selected-only H1 request draws the same candidate-indexed noise as an all-bank
    request. Consequently MI work cannot change the executed-action evidence law.
    """
    if candidate_indices is None:
        candidate_indices = [None] * len(controllers)
    if not (len(controllers) == len(histories) == len(banks) == len(candidate_indices)):
        raise ValueError("Prediction batch lengths differ")
    if not controllers:
        return []
    requests, shapes = [], []
    for c, history, bank, indices in zip(
        controllers, histories, banks, candidate_indices, strict=True
    ):
        if bank.shape != (24, 8, 6) or not 1 <= horizon <= 8 or samples < 1:
            raise ValueError("Reference predictions require a complete 24-by-H8 bank")
        indices = (
            torch.arange(24, device=c.device)
            if indices is None
            else torch.as_tensor(indices, device=c.device)
        )
        if (
            indices.ndim != 1
            or indices.numel() == 0
            or indices.dtype != torch.long
            or ((indices < 0) | (indices >= 24)).any()
        ):
            raise ValueError("Invalid original candidate indices")
        k = len(indices)
        noise = torch.randn(
            24,
            samples,
            2 * horizon,
            c.model.token_width,
            device=c.device,
            generator=_generator(c, stream_tag),
        )[indices]
        noise = noise[:, None].expand(k, 8, samples, 2 * horizon, c.model.token_width)
        condition = torch.arange(8, device=c.device)[None, :, None].expand(k, 8, samples)
        actions = bank[indices, None, None, :horizon].expand(k, 8, samples, horizon, 6)
        context = {
            key: value.expand(k * 8 * samples, *value.shape[1:]) for key, value in history.items()
        }
        requests.append(
            (
                context,
                condition.reshape(-1),
                actions.reshape(-1, horizon, 6),
                noise.reshape(-1, 2 * horizon, c.model.token_width),
            )
        )
        shapes.append((k, 8, samples))
    result = _sample_requests(controllers, requests, horizon)
    return [
        {key: value.reshape(*shape, *value.shape[1:]) for key, value in prediction.items()}
        for shape, prediction in zip(shapes, result, strict=True)
    ]

def reference_evidence(controllers, histories, banks, candidate_indices=None):
    predictions = reference_predictions(
        controllers, histories, banks, 16, 1, EVIDENCE_TAG, candidate_indices
    )
    laws = []
    for c, history, prediction in zip(controllers, histories, predictions, strict=True):
        features = (
            prediction["observations"][..., 0, :]
            - history["observations"][0, -1]
            - c.model.delta_mean
        ) / c.model.delta_std
        laws.append(calibrated_moments(features, c.extra_variance))
    return laws

class HiddenReferencePolicy:
    """Generic runner adapter; accepts public history only and executes one native action."""

    receives_truth = False
    records_beliefs = True
    history_length = 8

    def __init__(
        self,
        method,
        model,
        config,
        seed,
        calibration,
        *,
        selection,
        observation_law,
        split="development",
    ):
        if method not in METHODS or split not in ("development", "test"):
            raise ValueError("Unknown hidden-reference method or split")
        if (method == "revised_vfe" and config.information_weight != 0) or (
            method == "revised_efe" and config.information_weight <= 0
        ):
            raise ValueError("VFE requires lambda0; EFE requires positive lambda")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("A nonnegative integer case sampling seed is required")
        self.method, self.model, self.config, self.seed = method, model, config, int(seed)
        self.selection, self.split = selection, split
        self.controller = HiddenReferenceController(
            model,
            config,
            selection=selection,
            observation_law=observation_law,
            extra_variance=calibration,
            seed=self.seed,
        )
        self.device = self.controller.device

    def recipe(self):
        return dict(
            schema_version=1,
            method=self.method,
            split=self.split,
            seed=self.seed,
            config=asdict(self.config),
            history_length=8,
            receives_truth=False,
            records_beliefs=True,
            controller_recipe=self.controller.recipe(),
        )

    to_spec = recipe

    def act(self, history, step, oracle_condition=None):
        return policy_preservation_actions([self], [history], [step], [oracle_condition])[0]

    def observe(self, action, next_observation, updated_history=None):
        return self.controller.observe(action, next_observation, updated_history)

@torch.no_grad()
def policy_preservation_actions(policies, histories, steps, oracle_conditions=None):
    if oracle_conditions is None:
        oracle_conditions = [None] * len(policies)
    if not (len(policies) == len(histories) == len(steps) == len(oracle_conditions)):
        raise ValueError("Decision batch lengths differ")
    if not policies:
        return []
    if len({id(p) for p in policies}) != len(policies):
        raise ValueError("Each batch entry needs a distinct policy")
    first = policies[0]
    priors, controllers = [], []
    for p, history, step, truth in zip(policies, histories, steps, oracle_conditions, strict=True):
        if not isinstance(p, HiddenReferencePolicy) or p.model is not first.model:
            raise ValueError("Hidden-reference batch requires one shared fitted model")
        query_config = {k: v for k, v in asdict(p.config).items() if k != "information_weight"}
        first_config = {k: v for k, v in asdict(first.config).items() if k != "information_weight"}
        if query_config != first_config:
            raise ValueError("Batched predictive query configurations differ")
        _public_history(history)
        if truth is not None:
            raise ValueError("Hidden-reference controller must not receive evaluator truth")
        if isinstance(step, (bool, np.bool_)) or not isinstance(step, Integral) or step < 0:
            raise ValueError("Decision step must be a nonnegative integer")
        c = p.controller
        if c.cache is not None or c._repair_prior_route:
            raise RuntimeError("Observe the previous executed action before replanning")
        c.query_seed = decision_seed(p.seed, step)
        c._repair_prior_route = True
        try:
            priors.append(c.prior(history))
        finally:
            c._repair_prior_route = False
        controllers.append(c)
    banks, labels = reference_proposals(controllers, histories, priors)
    predictions = reference_predictions(
        controllers, histories, banks, first.config.task_samples, first.config.horizon, TASK_TAG
    )
    active = [i for i, p in enumerate(policies) if p.config.information_weight > 0]
    laws = dict(
        zip(
            active,
            reference_evidence(
                [controllers[i] for i in active],
                [histories[i] for i in active],
                [banks[i] for i in active],
            )
            if active
            else [],
            strict=True,
        )
    )
    selected, details = [], []
    for i, (p, c, prior, prediction) in enumerate(
        zip(policies, controllers, priors, predictions, strict=True)
    ):
        costs = expected_task_cost(
            prediction["rewards"], prediction["termination_probability"], prior, p.config.discount
        )
        information, error = torch.zeros_like(costs), torch.zeros_like(costs)
        if i in laws:
            information, error = c.observation_law.mutual_information(
                *laws[i],
                prior,
                draws=p.config.information_draws,
                generator=_generator(c, INFORMATION_TAG),
            )
        objective = costs - p.config.information_weight * information
        if not all(torch.isfinite(x).all() for x in (costs, information, error, objective)):
            raise FloatingPointError("FATAL_CONTROLLER: nonfinite task/information score")
        chosen, weights = select_candidate(objective, p.selection, _generator(c, SELECTION_TAG))
        selected.append(chosen)
        details.append(
            dict(
                prior=prior.cpu().numpy(),
                proposal_conditions=labels[i].cpu().numpy(),
                selected=chosen,
                costs=costs.cpu().numpy(),
                information=information.cpu().numpy(),
                information_se=error.cpu().numpy(),
                objective=objective.cpu().numpy(),
                weights=weights.cpu().numpy(),
                ess=float(weights.square().sum().reciprocal()),
                candidate_actions=banks[i].cpu().numpy(),
                candidate_first_actions=banks[i][:, 0].cpu().numpy(),
                selection_temperature=p.selection.temperature,
                cost_scale=p.selection.cost_scale,
                query_seed=c.query_seed,
            )
        )
    passive = [i for i in range(len(policies)) if i not in laws]
    selected_laws = dict(
        zip(
            passive,
            [
                (mean[0], variance[0])
                for mean, variance in reference_evidence(
                    [controllers[i] for i in passive],
                    [histories[i] for i in passive],
                    [banks[i] for i in passive],
                    [[selected[i]] for i in passive],
                )
            ]
            if passive
            else [],
            strict=True,
        )
    )
    selected_laws.update(
        {i: (mean[selected[i]], variance[selected[i]]) for i, (mean, variance) in laws.items()}
    )
    outputs = []
    for i, c in enumerate(controllers):
        mean, variance = selected_laws[i]
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(variance).all()
            or (variance <= 0).any()
        ):
            raise FloatingPointError("FATAL_CONTROLLER: invalid cached evidence law")
        action = banks[i][selected[i], 0].clone()
        c.cache = dict(
            prior=priors[i].clone(),
            mean=mean,
            base_variance=variance,
            observation=histories[i]["observations"][0, -1].clone(),
            action=action,
        )
        outputs.append((action.cpu().numpy().copy(), details[i]))
    return outputs
