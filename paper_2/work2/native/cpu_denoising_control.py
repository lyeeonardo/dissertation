"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import torch

FORCE_SCALES = (0.75, 1.0, 1.25)

REAL_HAZARD = 0.06

_TAGS = dict(
    condition=511,
    proposal=521,
    prediction=523,
    termination=541,
    imagination=547,
    gaussian=557,
    actual=563,
)

def _rng(seed, stage, step=0):
    return torch.Generator(device="cpu").manual_seed(int(seed) ^ _TAGS[stage] ^ (step * 1009))
