"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import torch

@dataclass(frozen=True)
class ControlConfig:
    horizon: int = 8
    candidates: int = 32
    task_samples: int = 8
    evidence_samples: int = 16
    sampling_steps: int = 8
    temperature: float = 1.0
    hazard: float = 0.01
    discount: float = 0.99
    information_weight: float = 0.0
    information_draws: int = 64
    chunk_size: int = 256

class PublicHistory:
    def __init__(self, observation, length=8):
        self.length = length
        self.observations = [np.asarray(observation, np.float32).copy()]
        self.actions, self.rewards = [], []

    def append(self, action, observation, reward):
        self.actions.append(np.asarray(action, np.float32).copy())
        self.rewards.append(float(reward))
        self.observations.append(np.asarray(observation, np.float32).copy())
        self.observations = self.observations[-self.length :]
        self.actions, self.rewards = (
            self.actions[-self.length + 1 :],
            self.rewards[-self.length + 1 :],
        )

    def tensors(self, device):
        n = len(self.observations)
        obs = np.zeros((self.length, 17), np.float32)
        actions, rewards = (
            np.zeros((self.length - 1, 6), np.float32),
            np.zeros(self.length - 1, np.float32),
        )
        mask = np.zeros(self.length, np.float32)
        obs[-n:], mask[-n:] = self.observations, 1
        if n > 1:
            actions[-n + 1 :], rewards[-n + 1 :] = self.actions, self.rewards
        return {
            key: torch.as_tensor(value[None], device=device)
            for key, value in dict(
                observations=obs, actions=actions, rewards=rewards, mask=mask
            ).items()
        }


from .inference import calibrated_moments, expected_task_cost, gaussian_log_prob, mutual_information, predict_belief, update_belief

def allocate_candidates(prior, count):
    probabilities = prior.detach().cpu().numpy().astype(np.float64)
    base = (prior >= 0.02).detach().cpu().numpy().astype(np.int64)
    if base.sum() > count:
        base[:] = 0
        base[np.argsort(-probabilities, kind="stable")[:count]] = 1
    remaining = count - base.sum()
    quota = probabilities * remaining / probabilities.sum()
    whole = np.floor(quota).astype(np.int64)
    base += whole
    order = np.argsort(-(quota - whole), kind="stable")
    base[order[: count - base.sum()]] += 1
    return np.repeat(np.arange(len(prior)), base)

class BeliefController:
    def __init__(
        self,
        model,
        config=None,
        mode="vfe_full",
        extra_variance=None,
        seed=13,
        classifier=None,
    ):
        if mode not in (
            "vfe_full",
            "vfe_map",
            "oracle",
            "fixed",
            "supervised_full",
            "supervised_map",
        ):
            raise ValueError(mode)
        self.model, self.config, self.mode = model, config or ControlConfig(), mode
        self.classifier = classifier
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.numpy_rng = np.random.default_rng(seed)
        self.extra_variance = torch.as_tensor(
            np.full(17, 1e-4) if extra_variance is None else extra_variance,
            device=self.device,
            dtype=torch.float32,
        )
        if not torch.isfinite(self.extra_variance).all() or (self.extra_variance <= 0).any():
            raise ValueError("Calibration variance must be positive and finite")
        self.belief = torch.full((8,), 1 / 8, device=self.device)
        self.cache = None

    def _sample(self, history, conditions, horizon, actions=None, noise=None):
        outputs = {}
        for start in range(0, len(conditions), self.config.chunk_size):
            end = start + self.config.chunk_size
            context = {key: value[start:end] for key, value in history.items()}
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                result = self.model.sample(
                    context,
                    conditions[start:end],
                    horizon,
                    fixed_actions=actions[start:end] if actions is not None else None,
                    noise=noise[start:end] if noise is not None else None,
                    generator=self.generator,
                    temperature=self.config.temperature,
                    steps=self.config.sampling_steps,
                )
            for key, value in result.items():
                outputs.setdefault(key, []).append(value.float())
        return {key: torch.cat(value) for key, value in outputs.items()}

    def proposals(self, history, prior):
        return proposals_batch([self], [history], [prior])[0]

    def predictions(self, history, candidates, samples, horizon, hypotheses=None):
        return predictions_batch([self], [history], [candidates], samples, horizon, [hypotheses])[0]

    def evidence_laws(self, history, candidates):
        predicted = self.predictions(history, candidates, self.config.evidence_samples, 1)
        current = history["observations"][0, -1]
        features = (
            predicted["observations"][..., 0, :] - current - self.model.delta_mean
        ) / self.model.delta_std
        return calibrated_moments(features, self.extra_variance)

    @torch.no_grad()
    def act(self, history, true_condition=None):
        return act_batch([self], [history], [true_condition])[0]

    def prior(self, history, true_condition=None):
        # true_condition is accepted only for the explicitly privileged reference arm.
        if self.cache is not None:
            raise RuntimeError(
                "The previous executed transition must be observed before replanning"
            )
        if self.mode == "oracle":
            if true_condition is None:
                raise ValueError("Oracle requires the next physical condition")
            prior = torch.nn.functional.one_hot(
                torch.tensor(true_condition, device=self.device), 8
            ).float()
        elif true_condition is not None:
            raise ValueError("Non-oracle controller must not receive evaluator truth")
        elif self.mode == "fixed":
            prior = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0], device=self.device)
        elif self.mode.startswith("supervised"):
            if self.classifier is None:
                raise ValueError("A trained history classifier is required")
            prior = predict_belief(self.classifier.probabilities(history)[0], self.config.hazard)
        else:
            prior = predict_belief(self.belief, self.config.hazard)
        return prior

    @torch.no_grad()
    def observe(self, action, next_observation, updated_history=None):
        if self.cache is None:
            raise RuntimeError("No cached prediction for this executed action")
        actual = torch.as_tensor(action, device=self.device)
        if not torch.allclose(actual, self.cache["action"], atol=1e-6, rtol=0):
            raise ValueError("Executed action differs from the action used for evidence")
        observed = torch.as_tensor(next_observation, device=self.device)
        feature = (
            observed - self.cache["observation"] - self.model.delta_mean
        ) / self.model.delta_std
        log_likelihood = gaussian_log_prob(feature, self.cache["mean"], self.cache["variance"])
        if self.mode.startswith("supervised"):
            if updated_history is None:
                raise ValueError("Supervised posterior requires the updated public history")
            self.belief = self.classifier.probabilities(updated_history)[0]
        else:
            self.belief = update_belief(self.cache["prior"], log_likelihood)
        self.cache = None
        return {"belief": self.belief.cpu().numpy(), "log_likelihood": log_likelihood.cpu().numpy()}

def _sample_requests(controllers, requests, horizon):
    """Combine deterministic requests; random draws remain owned by each case."""
    first = controllers[0]
    for controller in controllers:
        if controller.model is not first.model:
            raise ValueError("A GPU batch must share the same fitted model")
        for field in ("sampling_steps", "temperature", "chunk_size"):
            if getattr(controller.config, field) != getattr(first.config, field):
                raise ValueError(f"Batched queries differ in {field}")
    counts = [len(request[1]) for request in requests]
    history = {key: torch.cat([r[0][key] for r in requests]) for key in requests[0][0]}
    labels = torch.cat([r[1] for r in requests])
    actions = torch.cat([r[2] for r in requests]) if requests[0][2] is not None else None
    noise = torch.cat([r[3] for r in requests])
    result = first._sample(history, labels, horizon, actions, noise)
    split = {key: value.split(counts) for key, value in result.items()}
    return [{key: value[i] for key, value in split.items()} for i in range(len(requests))]

@torch.no_grad()
def proposals_batch(controllers, histories, priors):
    requests = []
    horizon = controllers[0].config.horizon
    for controller, history, prior in zip(controllers, histories, priors, strict=True):
        n = controller.config.candidates * 3 // 4
        if controller.config.horizon != horizon or n < 1:
            raise ValueError("Batched proposals require a common positive horizon/candidate count")
        labels = torch.as_tensor(allocate_candidates(prior, n), device=controller.device)
        context = {key: value.expand(n, *value.shape[1:]) for key, value in history.items()}
        noise = torch.randn(
            n,
            2 * horizon,
            controller.model.token_width,
            device=controller.device,
            generator=controller.generator,
        )
        requests.append((context, labels, None, noise))
    results = _sample_requests(controllers, requests, horizon)
    candidates = []
    for controller, result in zip(controllers, results, strict=True):
        proposed = result["actions"]
        count = controller.config.candidates - len(proposed)
        parents = torch.randint(
            len(proposed), (count,), device=controller.device, generator=controller.generator
        )
        noise = (
            torch.randn(count, horizon, 6, device=controller.device, generator=controller.generator)
            * 0.15
        )
        for step in range(1, horizon):
            noise[:, step] = 0.7 * noise[:, step - 1] + 0.3 * noise[:, step]
        candidates.append(torch.cat((proposed, (proposed[parents] + noise).clamp(-1, 1))))
    return candidates

@torch.no_grad()
def predictions_batch(controllers, histories, candidates, samples, horizon, hypotheses=None):
    requests, shapes = [], []
    if hypotheses is None:
        hypotheses = [None] * len(controllers)
    for controller, history, actions, labels in zip(
        controllers, histories, candidates, hypotheses, strict=True
    ):
        labels = torch.arange(8, device=controller.device) if labels is None else labels
        k, j = len(actions), len(labels)
        n = k * j * samples
        shapes.append((k, j, samples))
        condition = labels[None, :, None].expand(k, j, samples).reshape(n)
        imposed = (
            actions[:, None, None, :horizon]
            .expand(k, j, samples, horizon, 6)
            .reshape(n, horizon, 6)
        )
        # Different candidates get separate particles; hypotheses share draws.
        noise = torch.randn(
            k,
            samples,
            2 * horizon,
            controller.model.token_width,
            device=controller.device,
            generator=controller.generator,
        )
        noise = noise[:, None].expand(k, j, samples, 2 * horizon, controller.model.token_width)
        context = {key: value.expand(n, *value.shape[1:]) for key, value in history.items()}
        requests.append((context, condition, imposed, noise.reshape(n, 2 * horizon, -1)))
    results = _sample_requests(controllers, requests, horizon)
    return [
        {key: value.reshape(*shape, *value.shape[1:]) for key, value in result.items()}
        for shape, result in zip(shapes, results, strict=True)
    ]

def _evidence_batch(controllers, histories, candidates):
    samples = controllers[0].config.evidence_samples
    if any(c.config.evidence_samples != samples for c in controllers):
        raise ValueError("Batched evidence requires the same predictive sample count")
    predictions = predictions_batch(controllers, histories, candidates, samples, 1)
    laws = []
    for controller, history, prediction in zip(controllers, histories, predictions, strict=True):
        feature = (
            prediction["observations"][..., 0, :]
            - history["observations"][0, -1]
            - controller.model.delta_mean
        ) / controller.model.delta_std
        laws.append(calibrated_moments(feature, controller.extra_variance))
    return laws

@torch.no_grad()
def act_batch(controllers, histories, true_conditions=None):
    """One complete GPU decision stage for independent cases, with no truth sharing."""
    if not controllers:
        return []
    if true_conditions is None:
        true_conditions = [None] * len(controllers)
    priors = [
        c.prior(h, truth)
        for c, h, truth in zip(controllers, histories, true_conditions, strict=True)
    ]
    candidates = proposals_batch(controllers, histories, priors)
    config = controllers[0].config
    if any(c.config.task_samples != config.task_samples for c in controllers):
        raise ValueError("Batched task predictions require a common particle count")
    # Dropping exact zero weights is valid for privileged/fixed modes only.
    hypotheses = [
        p.nonzero().flatten() if c.mode in ("oracle", "fixed") else None
        for c, p in zip(controllers, priors, strict=True)
    ]
    predictions = predictions_batch(
        controllers, histories, candidates, config.task_samples, config.horizon, hypotheses
    )
    active_mi = [i for i, c in enumerate(controllers) if c.config.information_weight > 0]
    laws = dict(
        zip(
            active_mi,
            _evidence_batch(
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
    for i, (controller, prior, prediction) in enumerate(
        zip(controllers, priors, predictions, strict=True)
    ):
        weights = prior if hypotheses[i] is None else prior[hypotheses[i]]
        if controller.mode.endswith("_map"):
            weights = torch.nn.functional.one_hot(prior.argmax(), 8).float()
        costs = expected_task_cost(
            prediction["rewards"],
            prediction["termination_probability"],
            weights,
            controller.config.discount,
        )
        information, error = torch.zeros_like(costs), torch.zeros_like(costs)
        if i in laws:
            information, error = mutual_information(
                *laws[i],
                prior,
                draws=controller.config.information_draws,
                generator=controller.generator,
            )
        objective = costs - controller.config.information_weight * information
        if not torch.isfinite(objective).all():
            raise FloatingPointError("FATAL_CONTROLLER: nonfinite candidate score")
        selected.append(int(objective.argmin()))
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
    one_action_laws = (
        _evidence_batch(
            [controllers[i] for i in remaining],
            [histories[i] for i in remaining],
            [candidates[i][selected[i] : selected[i] + 1] for i in remaining],
        )
        if remaining
        else []
    )
    selected_laws = {
        i: (mean[0], var[0]) for i, (mean, var) in zip(remaining, one_action_laws, strict=True)
    }
    selected_laws.update(
        {i: (mean[selected[i]], var[selected[i]]) for i, (mean, var) in laws.items()}
    )
    actions = []
    for i, controller in enumerate(controllers):
        mean, variance = selected_laws[i]
        action = candidates[i][selected[i], 0]
        controller.cache = {
            "prior": priors[i],
            "mean": mean,
            "variance": variance,
            "observation": histories[i]["observations"][0, -1].clone(),
            "action": action.clone(),
        }
        actions.append((action.cpu().numpy(), details[i]))
    return actions
