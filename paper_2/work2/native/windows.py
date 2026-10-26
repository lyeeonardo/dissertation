"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import json

from pathlib import Path

import h5py

import numpy as np

import torch

class WindowSampler:
    def __init__(self, output: Path, arm="R", split="train", seed=13, history=8, horizon=16):
        self.output, self.arm, self.split = output, arm, split
        self.history, self.horizon = history, horizon
        self.rng = np.random.default_rng(seed)
        from work2.paths import corpus_rows
        rows = corpus_rows(output, arm)
        self.rows = [row for row in rows if row["split"] == split]
        self.normalization = json.loads((output / "normalization.json").read_text())
        self.arrays = {}
        pieces = {
            key: []
            for key in (
                "observations",
                "actions",
                "rewards",
                "terminated",
                "eligible",
                "valid",
                "observation_valid",
                "condition",
            )
        }
        anchors, offset = [], 0
        handles = {}
        try:
            for row in self.rows:
                if row["path"] not in handles:
                    handles[row["path"]] = h5py.File(row["path"], "r")
                group = handles[row["path"]][row["episode"]]
                n = row["length"]
                left = history - 1
                total = left + n + 1 + horizon
                data = {
                    "observations": np.zeros((total, 17), np.float32),
                    "actions": np.zeros((total, 6), np.float32),
                    "rewards": np.zeros(total, np.float32),
                    "terminated": np.zeros(total, np.float32),
                    "eligible": np.zeros(total, bool),
                    "valid": np.zeros(total, bool),
                    "observation_valid": np.zeros(total, bool),
                    "condition": np.full(total, row["condition"], np.int64),
                }
                data["observations"][left : left + n + 1] = group["observations"][:]
                for key in ("actions", "rewards", "terminated"):
                    data[key][left : left + n] = group[key][:]
                data["eligible"][left : left + n] = group["proposal_eligible"][:]
                data["valid"][left : left + n] = True
                data["observation_valid"][left : left + n + 1] = True
                if "prefix_observations" in group:
                    if group["prefix_observations"].shape[0] < history:
                        raise ValueError("Requested history exceeds retained branch prefix")
                    data["observations"][:left] = group["prefix_observations"][-history:-1]
                    data["actions"][:left] = group["prefix_actions"][-left:]
                    data["rewards"][:left] = group["prefix_rewards"][-left:]
                    data["observation_valid"][:left] = group["prefix_mask"][-history:-1]
                    data["condition"][:left] = int(group.attrs["source_condition"])
                else:
                    data["condition"][:left] = -1  # No completed transition before reset.
                for key in pieces:
                    pieces[key].append(data[key])
                anchors.append(np.arange(offset + left, offset + left + n, dtype=np.int64))
                offset += total
        finally:
            for handle in handles.values():
                handle.close()
        if not anchors:
            raise ValueError(f"No episodes for {arm}/{split}")
        self.arrays = {key: np.concatenate(value) for key, value in pieces.items()}
        self.anchors = np.concatenate(anchors)
        self.pools, self.proposal_pools = [], []
        labels = self.arrays["condition"][self.anchors]
        eligible = self.arrays["eligible"][self.anchors]
        for condition in range(8):
            pool = self.anchors[labels == condition]
            proposal = self.anchors[(labels == condition) & eligible]
            if not len(pool) or not len(proposal):
                raise ValueError(
                    f"Missing prediction/proposal support in {arm}/{split}/condition {condition}"
                )
            self.pools.append(pool)
            self.proposal_pools.append(proposal)

    def history_at(self, anchors):
        obs = anchors[:, None] + np.arange(-self.history + 1, 1)
        previous = anchors[:, None] + np.arange(-self.history + 1, 0)
        return {
            "observations": self.arrays["observations"][obs],
            "actions": self.arrays["actions"][previous],
            "rewards": self.arrays["rewards"][previous],
            "mask": self.arrays["observation_valid"][obs].astype(np.float32),
        }

    def sample(self, batch_size=1024, augment=True):
        if batch_size % 32:
            raise ValueError(
                "Batch must be divisible by 32 for balanced conditions and 50/25/25 roles"
            )
        n = batch_size
        role = np.repeat(np.arange(3), [n // 2, n // 4, n // 4])
        condition = np.tile(np.arange(8), n // 8)
        anchors = np.empty(n, np.int64)
        for r in range(3):
            for c in range(8):
                mask = (role == r) & (condition == c)
                pool = self.proposal_pools[c] if r == 0 else self.pools[c]
                anchors[mask] = self.rng.choice(pool, int(mask.sum()))
        future = anchors[:, None] + np.arange(self.horizon)
        horizons = self.rng.choice([4, 8, 16], n)
        horizons[role == 1] = 1
        history = self.history_at(anchors)
        quality = (role == 0).astype(np.int64)
        if augment:
            short = self.rng.random(n) < 0.2
            lengths = self.rng.choice([1, 2, 4], n)
            history["mask"] *= ~short[:, None] | (
                np.arange(self.history)[None] >= self.history - lengths[:, None]
            )
            quality[self.rng.random(n) < 0.2] = 0
        result = {
            "history": history,
            "condition": condition,
            "role": role,
            "horizon": horizons,
            "quality": quality,
            "actions": self.arrays["actions"][future],
            "future_observations": self.arrays["observations"][future + 1],
            "rewards": self.arrays["rewards"][future],
            "terminated": self.arrays["terminated"][future],
            "valid": self.arrays["valid"][future],
            "proposal_eligible": self.arrays["eligible"][future],
        }
        permutation = self.rng.permutation(n)
        return {
            key: (
                {name: value[permutation] for name, value in values.items()}
                if isinstance(values, dict)
                else values[permutation]
            )
            for key, values in result.items()
        }

    def transition_batch(self, batch_size=1024):
        anchors = self.rng.choice(self.anchors, batch_size)
        return {
            "observations": self.arrays["observations"][anchors],
            "next_observations": self.arrays["observations"][anchors + 1],
            "actions": self.arrays["actions"][anchors],
            "rewards": self.arrays["rewards"][anchors],
            "terminated": self.arrays["terminated"][anchors],
            "history": self.history_at(anchors),
            # For supervised history, label the LAST completed law, never the future branch label.
            "history_label": self.arrays["condition"][anchors - 1],
            "condition": self.arrays["condition"][anchors],
        }

def to_device(batch, device="cuda"):
    return {
        key: (
            to_device(value, device)
            if isinstance(value, dict)
            else torch.as_tensor(value, device=device)
        )
        for key, value in batch.items()
    }

def slice_batch(batch, start, end):
    return {
        key: (slice_batch(value, start, end) if isinstance(value, dict) else value[start:end])
        for key, value in batch.items()
    }
