"""Frozen 36-episode public-controller/oracle evaluation, one worker per job.

Primary scoring uses days 7--14 after uninterrupted physical conditioning on
days 0--7. Model checkpoint selection must precede these jobs. Workers never
restart or resume themselves; a saved checkpoint is retained for explicit use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import CollectionPolicy

class PublicPI:
    """The fixed 15-minute nominal PI behavior, using public DO5 and NO2 only."""
    def __init__(self, seed=17):
        self.policy = CollectionPolicy("nominal_pi", seed)
        self.step = 0

    def act(self, observation, previous_action=None):
        action = self.policy.action(self.step, observation)
        self.step += 1
        return action

def _synchronize(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)

def _save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **{key: np.asarray(value) for key, value in arrays.items()})
    temporary.replace(path)
