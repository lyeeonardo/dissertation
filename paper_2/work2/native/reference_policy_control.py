from __future__ import annotations


import math

from dataclasses import dataclass

import torch

GENERATED_CANDIDATES = 24

@dataclass(frozen=True)
class ReferenceSelectionConfig:
    selector: str
    cost_scale: float
    temperature_ratio: float | None = None

    def __post_init__(self):
        if self.selector not in ("uniform", "hard", "soft"):
            raise ValueError("Selector must be uniform, hard, or soft")
        if not math.isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("The frozen global DEV cost scale must be positive and finite")
        if self.selector == "soft":
            if self.temperature_ratio not in (0.5, 2.0):
                raise ValueError("Exp1 permits only soft temperature ratios 0.5 and 2")
            if (
                not math.isfinite(self.cost_scale * self.temperature_ratio)
                or self.cost_scale * self.temperature_ratio <= 0
            ):
                raise ValueError("Selection temperature must be positive and finite")
        elif self.temperature_ratio is not None:
            raise ValueError("Only soft selection has a finite temperature ratio")

    @property
    def temperature(self):
        if self.selector == "soft":
            return self.cost_scale * self.temperature_ratio
        return 0.0 if self.selector == "hard" else None

def selection_weights(costs, selection):
    """Weights on an already sampled, equally weighted oracle reference bank.

    No diffusion-density factor and no bank-dependent standardization enter.
    Uniform selection is represented exactly, avoiding an infinite JSON value.
    """
    if costs.shape != (GENERATED_CANDIDATES,) or not torch.isfinite(costs).all():
        raise ValueError("Selection needs 24 finite generated-candidate costs")
    costs = costs.double()
    if selection.selector == "uniform":
        return torch.full_like(costs, 1 / GENERATED_CANDIDATES)
    if selection.selector == "hard":
        return torch.nn.functional.one_hot(costs.argmin(), GENERATED_CANDIDATES).double()
    return torch.softmax(-(costs - costs.min()) / selection.temperature, dim=0)

def select_candidate(costs, selection, generator):
    weights = selection_weights(costs, selection)
    selected = (
        int(costs.argmin())
        if selection.selector == "hard"
        else int(torch.multinomial(weights, 1, generator=generator))
    )
    return selected, weights

def _public_history(history):
    shapes = {"observations": (1, 8, 17), "actions": (1, 7, 6), "rewards": (1, 7), "mask": (1, 8)}
    if set(history) != set(shapes):
        raise ValueError("Only public observation/action/reward/mask history is accepted")
    for key, shape in shapes.items():
        if history[key].shape != shape or not torch.isfinite(history[key]).all():
            raise ValueError(f"Invalid public history field: {key}")
