from __future__ import annotations

import argparse

from collections import Counter, defaultdict

import copy

from dataclasses import asdict

import hashlib

import importlib.util

import json

import math

import os

from pathlib import Path

import sys

import time

import traceback

def merge_samplers(left, right, seed):
    """Merge disjoint audited inventories; preserve native array and RNG operations."""
    import numpy as np
    cls = type(left)
    if type(right) is not cls or tuple(left.conditions) != tuple(range(4)) or tuple(right.conditions) != tuple(range(4, 8)):
        raise ValueError('Expected original1x and qualified3x native sampler inventories')
    if left.normalization != right.normalization:
        raise ValueError('Replay normalization differs')
    result = cls.__new__(cls)
    result.conditions = tuple(range(8)); result.history, result.horizon = 8, 16
    result.normalization = left.normalization; result.rows = left.rows + right.rows
    result.rng = np.random.default_rng(seed)
    offset = len(left.arrays['valid'])
    result.arrays = {key: np.concatenate([left.arrays[key], right.arrays[key]]) for key in left.arrays}
    result.anchors = np.concatenate([left.anchors, right.anchors + offset])
    result.pools = {**left.pools, **{c: values + offset for c, values in right.pools.items()}}
    result.proposal_pools = {**left.proposal_pools, **{c: values + offset for c, values in right.proposal_pools.items()}}
    return result

def development_bank(sampler):
    import numpy as np
    rng = np.random.default_rng(CONFIG['dev_seed']); banks = {}; roots = {}
    n = CONFIG['dev_batch_per_condition']
    for condition in range(8):
        roles = np.repeat(np.arange(3), [n//2, n//4, n//4])
        anchors = np.empty(n, np.int64)
        for role in range(3):
            mask = roles == role
            pool = sampler.proposal_pools[condition] if role == 0 else sampler.pools[condition]
            anchors[mask] = rng.choice(pool, int(mask.sum()))
        horizons = rng.choice([4, 8, 16], n); horizons[roles == 1] = 1
        banks[str(condition)] = sampler.batch_at(anchors, roles, horizons, (roles == 0).astype(np.int64))
        roots[str(condition)] = anchors
    return dict(batches=banks, anchors=roots, source_rows=sampler.rows, seed=CONFIG['dev_seed'],
                diagnostic_only=True, augmentation=False, fixed_noise=True)

class QueryCounter:
    def __init__(self, online, ema):
        self.stage = 'development'; self.counts = Counter(); self.handles = []
        for name, model in [('online', online), ('ema', ema)]:
            def count(_module, args, name=name):
                self.counts[f'{self.stage}:{name}:forward_calls'] += 1
                self.counts[f'{self.stage}:{name}:forward_examples'] += len(args[0])
            self.handles.append(model.register_forward_pre_hook(count))

    def result(self):
        return dict(forward_calls=sum(v for k, v in self.counts.items() if k.endswith(':forward_calls')),
                    forward_examples=sum(v for k, v in self.counts.items() if k.endswith(':forward_examples')),
                    by_stage=dict(self.counts), includes_attempted_forwards=True)

    def close(self):
        for handle in self.handles: handle.remove()

def evaluate_diagnostics(ema, banks):
    import torch
    from work2.native.windows import to_device
    result = {}
    ema.eval()
    with torch.no_grad():
        for condition, batch in banks['batches'].items():
            generator = torch.Generator(device='cuda').manual_seed(CONFIG['dev_seed'] + int(condition))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, components = ema.loss(to_device(batch), generator)
            result[condition] = {key: float(value) for key, value in dict(loss=loss, **components).items()}
    if any(not math.isfinite(v) or v < 0 for row in result.values() for v in row.values()):
        raise FloatingPointError('Nonfinite/negative diagnostic DEV loss')
    mean = lambda labels: {key: sum(result[str(c)][key] for c in labels)/len(labels) for key in result['0']}
    return dict(per_condition=result, one_x=mean(range(4)), three_x=mean(range(4, 8)),
                balanced=mean(range(8)), diagnostic_only=True, checkpoint_selected=False)

CONFIG = {'seed': 13, 'max_updates': 5000, 'checkpoints': [0, 500, 1000, 2500, 5000], 'batch_size': 256, 'microbatch_size': 256, 'learning_rate': 1e-05, 'ema_rate': 0.99, 'weight_decay': 0.01, 'dev_batch_per_condition': 256, 'dev_seed': 618130003, 'compilation': False}
