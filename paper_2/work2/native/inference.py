from __future__ import annotations


import math

import torch

def transition_matrix(hazard=0.01, size=8, device="cpu"):
    if not 0 <= hazard <= 1 or size < 2:
        raise ValueError("Invalid total off-diagonal hazard")
    matrix = torch.full((size, size), hazard / (size - 1), device=device)
    matrix.fill_diagonal_(1 - hazard)
    return matrix

def predict_belief(belief, hazard=0.01):
    return transition_matrix(hazard, len(belief), belief.device) @ belief

def update_belief(prior, log_likelihood):
    if prior.ndim != 1 or log_likelihood.shape != prior.shape:
        raise ValueError("One likelihood is required for each categorical hypothesis")
    if not torch.isfinite(log_likelihood).all() or not torch.isfinite(prior).all():
        raise FloatingPointError("Nonfinite evidence or prior")
    if (prior < 0).any() or prior.sum() <= 0:
        raise ValueError("Invalid prior")
    logits = torch.where(prior > 0, prior.log(), -torch.inf) + log_likelihood
    return torch.softmax(logits.float(), dim=-1)

def gaussian_log_prob(value, mean, variance):
    variance = variance.float().clamp_min(1e-8)
    return -0.5 * (
        ((value.float() - mean.float()).square() / variance)
        + variance.log()
        + math.log(2 * math.pi)
    ).sum(-1)

def calibrated_moments(samples, extra_variance):
    """Sample axis is penultimate, feature axis is last; no policy density enters."""
    samples = samples.float()
    return samples.mean(-2), (samples.var(-2, unbiased=False) + extra_variance).clamp_min(1e-8)

def mutual_information(mean, variance, prior, draws=64, generator=None, noise=None):
    """Categorical MI of the very same diagonal Gaussian laws used for VFE.

    mean/variance: [candidates, hypotheses, features]. Returns raw MC estimates
    and standard errors; finite-sample negatives are neither hidden nor clipped.
    """
    candidates, hypotheses, dimensions = mean.shape
    if prior.shape != (hypotheses,):
        raise ValueError("MI needs the common hypothesis prior")
    if noise is None:
        noise = torch.randn(hypotheses, draws, dimensions, device=mean.device, generator=generator)
    observations = mean[:, :, None] + variance.sqrt()[:, :, None] * noise[None]
    all_log_prob = gaussian_log_prob(
        observations[:, :, :, None], mean[:, None, None], variance[:, None, None]
    )
    log_prior = torch.where(prior > 0, prior.log(), -torch.inf)
    mixture = torch.logsumexp(all_log_prob + log_prior, dim=-1)
    own = gaussian_log_prob(observations, mean[:, :, None], variance[:, :, None])
    ratios = own - mixture
    estimate = (ratios.mean(-1) * prior).sum(-1)
    error = ((ratios.var(-1, unbiased=True) * prior.square()).sum(-1) / draws).sqrt()
    return estimate, error

def expected_task_cost(rewards, termination_probability, prior, discount=0.99):
    """[candidate, hypothesis, particle, step] native rewards, zero after termination."""
    alive = torch.cat(
        (
            torch.ones_like(termination_probability[..., :1]),
            (1 - termination_probability[..., :-1]).clamp(0, 1),
        ),
        -1,
    ).cumprod(-1)
    gamma = discount ** torch.arange(rewards.shape[-1], device=rewards.device)
    values = (rewards.float() * alive * gamma).sum(-1).mean(-1)
    return -(values * prior).sum(-1)
