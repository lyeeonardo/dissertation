"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import numpy as np

import torch

def decision_seed(case_seed, step):
    return int(
        np.random.SeedSequence([int(case_seed), int(step), 13]).generate_state(1, dtype=np.uint64)[
            0
        ]
    ) % (2**63 - 1)

def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


import os

def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
