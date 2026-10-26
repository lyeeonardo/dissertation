from __future__ import annotations

import argparse

from collections import Counter, defaultdict

import copy

from dataclasses import asdict

import importlib.util

import json

import math

import os

from pathlib import Path

import sys

import time

import traceback



def inspect_data(index_path, conditions, split=None, expected_gravity=None):
    """Metadata check never opens excluded episode arrays or a neural model."""
    from work2.paths import corpus_rows
    rows = corpus_rows(Path(index_path).resolve().parents[1])
    parents, seen = defaultdict(set), set()
    for row in rows:
        key = (row['path'], row['episode'])
        if key in seen or row['split'] not in ('train', 'development', 'calibration', 'test'):
            raise ValueError('Duplicate episode or unknown split')
        seen.add(key); parents[str(row['parent_id'])].add(row['split'])
    if any(len(values) != 1 for values in parents.values()):
        raise ValueError('Parent leakage across data splits')
    chosen = [r for r in rows if r['condition'] in conditions and (split is None or r['split'] == split)]
    if not chosen or {r['condition'] for r in chosen} != set(conditions):
        raise ValueError('Missing requested condition/split support')
    if expected_gravity is not None and any(r.get('gravity_scale') != expected_gravity for r in chosen):
        raise ValueError('Physical gravity metadata differs from requested law')
    return chosen

class RestrictedWindowSampler:
    """Native window construction and sampling restricted to explicit conditions.

The native constructor requires all eight condition pools. This local copy
changes only the pool inventory; labels remain 4..7 for new3x, 0..3 for old1x.
Old TRAIN and all5x arrays are never opened. Branch prefixes in old DEV remain.
"""
    def __init__(self, output, conditions, split, seed):
        import h5py
        import numpy as np
        self.conditions = tuple(conditions)
        self.history, self.horizon = 8, 16
        self.rng = np.random.default_rng(seed)
        self.rows = inspect_data(Path(output) / 'splits/R_index.json', self.conditions, split)
        self.normalization = json.loads((Path(output) / 'normalization.json').read_text())
        pieces = {key: [] for key in ('observations', 'actions', 'rewards', 'terminated',
                  'eligible', 'valid', 'observation_valid', 'condition')}
        anchors, offset, handles = [], 0, {}
        try:
            for row in self.rows:
                if row['path'] not in handles:
                    handles[row['path']] = h5py.File(row['path'], 'r')
                group = handles[row['path']][row['episode']]
                n, left = int(row['length']), self.history - 1
                if (not group.attrs['complete'] or group.attrs['split'] != split
                        or str(group.attrs['parent_id']) != str(row['parent_id'])
                        or int(group.attrs['condition']) != row['condition']):
                    raise ValueError('Episode/index mismatch')
                expected_gravity = 3 if min(self.conditions) >= 4 else 1
                if float(group.attrs['gravity_scale']) != expected_gravity:
                    raise ValueError('Sampler encountered an incorrect actual physical law')
                total = left + n + 1 + self.horizon
                data = dict(observations=np.zeros((total, 17), np.float32), actions=np.zeros((total, 6), np.float32),
                            rewards=np.zeros(total, np.float32), terminated=np.zeros(total, np.float32),
                            eligible=np.zeros(total, bool), valid=np.zeros(total, bool),
                            observation_valid=np.zeros(total, bool), condition=np.full(total, row['condition'], np.int64))
                data['observations'][left:left+n+1] = group['observations'][:]
                for key in ('actions', 'rewards', 'terminated'):
                    data[key][left:left+n] = group[key][:]
                data['eligible'][left:left+n] = group['proposal_eligible'][:]
                data['valid'][left:left+n] = True
                data['observation_valid'][left:left+n+1] = True
                if 'prefix_observations' in group:
                    if group['prefix_observations'].shape[0] < self.history:
                        raise ValueError('Insufficient branch prefix')
                    data['observations'][:left] = group['prefix_observations'][-self.history:-1]
                    data['actions'][:left] = group['prefix_actions'][-left:]
                    data['rewards'][:left] = group['prefix_rewards'][-left:]
                    data['observation_valid'][:left] = group['prefix_mask'][-self.history:-1]
                    data['condition'][:left] = int(group.attrs['source_condition'])
                else:
                    data['condition'][:left] = -1
                for key in pieces:
                    pieces[key].append(data[key])
                anchors.append(np.arange(offset + left, offset + left + n, dtype=np.int64))
                offset += total
        finally:
            for handle in handles.values():
                handle.close()
        self.arrays = {key: np.concatenate(value) for key, value in pieces.items()}
        self.anchors = np.concatenate(anchors)
        self.pools, self.proposal_pools = {}, {}
        labels, eligible = self.arrays['condition'][self.anchors], self.arrays['eligible'][self.anchors]
        for condition in self.conditions:
            self.pools[condition] = self.anchors[labels == condition]
            self.proposal_pools[condition] = self.anchors[(labels == condition) & eligible]
            if not len(self.pools[condition]) or not len(self.proposal_pools[condition]):
                raise ValueError('Missing prediction/proposal support')

    def history_at(self, anchors):
        from work2.native.windows import WindowSampler
        return WindowSampler.history_at(self, anchors)

    def batch_at(self, anchors, roles, horizons, quality):
        import numpy as np
        future = anchors[:, None] + np.arange(self.horizon)
        return dict(history=self.history_at(anchors), condition=self.arrays['condition'][anchors],
                    role=roles, horizon=horizons, quality=quality,
                    actions=self.arrays['actions'][future], future_observations=self.arrays['observations'][future+1],
                    rewards=self.arrays['rewards'][future], terminated=self.arrays['terminated'][future],
                    valid=self.arrays['valid'][future], proposal_eligible=self.arrays['eligible'][future])

    def sample(self, batch_size=1024, augment=True):
        import numpy as np
        if batch_size % (4 * len(self.conditions)):
            raise ValueError('Batch must balance conditions and 50/25/25 roles')
        n = batch_size
        roles = np.repeat(np.arange(3), [n//2, n//4, n//4])
        conditions = np.tile(self.conditions, n//len(self.conditions))
        anchors = np.empty(n, np.int64)
        for role in range(3):
            for condition in self.conditions:
                mask = (roles == role) & (conditions == condition)
                pool = self.proposal_pools[condition] if role == 0 else self.pools[condition]
                anchors[mask] = self.rng.choice(pool, int(mask.sum()))
        horizons = self.rng.choice([4, 8, 16], n); horizons[roles == 1] = 1
        batch = self.batch_at(anchors, roles, horizons, (roles == 0).astype(np.int64))
        if augment:
            short = self.rng.random(n) < .2
            lengths = self.rng.choice([1, 2, 4], n)
            batch['history']['mask'] *= ~short[:, None] | (np.arange(self.history)[None] >= self.history-lengths[:, None])
            batch['quality'][self.rng.random(n) < .2] = 0
        order = self.rng.permutation(n)
        return {key: {name: value[order] for name, value in values.items()} if isinstance(values, dict)
                else values[order] for key, values in batch.items()}
