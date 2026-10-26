"""Relocate only two native belief means; retain their centered deviations.

This learned H1 location correction is not a new posterior likelihood. Public55
alone determines the target means. Nonnegative projection is recorded separately
because it can change the otherwise preserved covariance and target mean.
"""

from __future__ import annotations

import copy

import time

import numpy as np

import torch

from bsm1_offline_benchmark import stock_shared_evaluate as evaluator

from research_shared_model.defaults import controller_recipe as recipe

from .paired_forecast_control import DecisionLoggingController

SCHEMA = 'original300k_two_coordinate_weighted_mean_relocation_v1'

INDICES = np.array([59, 21], dtype=np.int64)

def relocate(values, weights, target):
    """Float64 weighted translation, then zero projection and execution cast.

Inputs are Kx2 current coordinates, normalized native weights and two physical
target means. No scale or random variable participates. Raw translation keeps
every pairwise deviation, including cross-covariance with untouched coordinates.
Returned diagnostics describe the particles after casting to their native dtype.
"""
    if values.ndim != 2 or values.shape[1] != 2 or weights.shape != values.shape[:1] or target.shape != (2,):
        raise ValueError('K two-coordinate particles, K weights and two target means required')
    x, w, mu = values.double(), weights.double(), target.double()
    if values.device.type == 'cpu':
        if not all(torch.isfinite(v).all() for v in (x, w, mu)) or (w < 0).any() or not torch.isclose(w.sum(), w.new_tensor(1.), rtol=0., atol=1e-12):
            raise ValueError('Finite particles/means and normalized nonnegative weights required')
    prior = (w[:, None]*x).sum(0)
    displacement = mu-prior
    raw = x+displacement
    projected = raw.clamp_min(0.).to(values.dtype)
    post = (w[:, None]*projected.double()).sum(0)
    details = dict(target_mean=mu, prior_mean=prior, post_mean=post,
        raw_shifted_mean=(w[:, None]*raw).sum(0), displacement=displacement,
        prior_sd=(w[:, None]*(x-prior).square()).sum(0).sqrt(),
        post_sd=(w[:, None]*(projected.double()-post).square()).sum(0).sqrt(),
        mean_target_residual=post-mu, projected_coordinates=(raw < 0).sum())
    return projected, raw, details

class SelectiveMeanController(DecisionLoggingController):
    def __init__(self, native, device):
        names = dict(particles='particle_count', refine='refinement')
        if native.model.checkpoint_updates != 300000 or any(getattr(native, names.get(k, k)) != v for k, v in recipe().items()):
            raise ValueError('Unchanged original300k native recipe required')
        super().__init__(native, device)
        self.boundaries = self.real_updates = self.initializer_calls = self.native_start_calls = 0
        self.provenance = []

    @torch.inference_mode()
    def act(self, public55, previous_action):
        started = time.perf_counter(); native = self.controller
        if native.pending is not None or native.remaining <= 0 or self.boundaries != self.real_updates:
            raise RuntimeError('Observe exactly once before the next mean relocation')
        public = np.asarray(public55, dtype=np.float32)
        if public.shape != (55,) or not np.isfinite(public).all():
            raise ValueError('Current finite public55 required')
        if not native.history_records:
            native.start(public, previous_action); self.native_start_calls += 1
        step = native.transition_count
        if step != self.boundaries or native.act_count != step or not np.array_equal(public, native.history_records[-1]['public']):
            raise RuntimeError('Current boundary must match the last native observation')
        if previous_action is not None and not np.allclose(np.asarray(previous_action, dtype=np.float32), native.history_records[-1]['previous_action'], atol=1e-7, rtol=0):
            raise ValueError('Previous action must be the actually executed command')
        before = native.particles.clone(); weights = native.weights.clone()
        first_two = step < 2
        rng_before = native.generator.get_state().cpu().numpy().copy() if first_two else None
        evaluator._synchronize(self.device); begin = time.perf_counter()
        initialized = native.model.initialize(native._history(), samples=0, generator=None)
        target = initialized['mean'][0, INDICES]
        evaluator._synchronize(self.device); initializer_seconds = time.perf_counter()-begin
        self.initializer_calls += 1
        projected, raw, details = relocate(before[:, INDICES], weights, target)
        native.particles[:, INDICES] = projected
        expected = before.clone(); expected[:, INDICES] = projected
        if first_two:
            rng_after = native.generator.get_state().cpu().numpy().copy()
            if not np.array_equal(rng_before, rng_after):
                raise AssertionError('Mean relocation consumed native RNG')
        scalars = {key: value.cpu().tolist() for key, value in details.items()}
        evaluator._synchronize(self.device); prefix_seconds = time.perf_counter()-started
        action = super().act(public55, previous_action)
        if not torch.equal(native.pending['particles'], expected) or not torch.equal(native.pending['weights'], weights):
            raise AssertionError('Native proposal root must retain the weighted two-coordinate relocation')
        if first_two:
            self.provenance.append(dict(step=step, particles_before=before.cpu().numpy(),
                particles_after=expected.cpu().numpy(), weights=weights.cpu().numpy(),
                target_mean=target.cpu().numpy(), raw_shifted_instruments=raw.cpu().numpy(),
                projected_instruments=projected.cpu().numpy(), rng_before=rng_before, rng_after=rng_after,
                public55=public.copy(), history=copy.deepcopy(native.history_records)))
        self.boundaries += 1
        self.last_diagnostics.update(scalars, mean_refresh_step=step,
            projected_denominator=16, initializer_seconds=initializer_seconds, mean_refresh_seconds=prefix_seconds,
            initializer_calls=self.initializer_calls, native_start_calls=self.native_start_calls,
            boundaries=self.boundaries, other157_coordinates_preserved=True, weights_preserved=True,
            native_sampling_streams_unchanged=True, fresh_sampling=False, initializer_scale_used=False,
            nitrate_calibration_used=False, acquired_readings=0, current_instrument_access=False,
            full_current_truth_access=False, future_truth_or_forcing_access=False)
        self.last_diagnostics['action_selection_seconds'] += prefix_seconds
        return action

    def observe(self, next_public55, executed_action, raw_cost):
        if self.boundaries != self.real_updates+1 or self.controller.pending is None:
            raise RuntimeError('Exactly one native observe per relocated boundary required')
        value = super().observe(next_public55, executed_action, raw_cost)
        self.real_updates += 1; self.last_diagnostics['real_updates'] = self.real_updates
        return value

    def runtime_state(self):
        return dict(super().runtime_state(), selective_mean=dict(schema=SCHEMA, boundaries=self.boundaries,
            real_updates=self.real_updates, initializer_calls=self.initializer_calls,
            native_start_calls=self.native_start_calls, provenance=copy.deepcopy(self.provenance)))

    def restore_runtime_state(self, state):
        extra = state.get('selective_mean', {})
        if extra.get('schema') != SCHEMA:
            raise ValueError('Identical selective-mean runtime contract required')
        super().restore_runtime_state(state)
        for key in ('boundaries', 'real_updates', 'initializer_calls', 'native_start_calls'):
            setattr(self, key, int(extra[key]))
        self.provenance = copy.deepcopy(extra['provenance'])
        if (self.boundaries != self.controller.act_count or self.real_updates != self.controller.transition_count
                or self.boundaries-self.real_updates != int(self.controller.pending is not None)
                or self.initializer_calls != self.boundaries or self.native_start_calls != int(self.boundaries > 0)):
            raise ValueError('Selective-mean/native runtime counters disagree')

def summarize_mean_refresh(rows, steps):
    if len(rows) != steps or [r['mean_refresh_step'] for r in rows] != list(range(steps)):
        raise ValueError('One mean-relocation record per actual boundary required')
    if any(r['real_updates'] != i+1 or r['initializer_calls'] != i+1 for i, r in enumerate(rows)):
        raise ValueError('Exactly one initializer and native observe per boundary required')
    projected = sum(r['projected_coordinates'] for r in rows)
    summary = dict(decisions=steps, real_updates=steps, initializer_calls=steps, native_start_calls=1,
        mean_refresh_calls=steps, projected_coordinates=projected, projected_denominator=16*steps,
        projection_rate=projected/(16*steps), modified_indices=INDICES.tolist(),
        other157_coordinates_preserved=True, weights_preserved=True, native_sampling_streams_unchanged=True,
        native_VFE_once=True, native_L1_unchanged=True, acquired_readings=0, current_instrument_access=False,
        fresh_sampling=False, initializer_scale_used=False, nitrate_calibration_used=False,
        full_current_truth_access=False, future_truth_or_forcing_access=False, terminal_state_refreshed=False,
        default_changed=False, maximum_absolute_mean_target_residual=float(np.abs([r['mean_target_residual'] for r in rows]).max()))
    for key in ('prior_mean', 'post_mean', 'target_mean', 'prior_sd', 'post_sd', 'displacement', 'mean_target_residual'):
        summary[key+'_mean'] = np.asarray([r[key] for r in rows]).mean(0).tolist()
    for key in ('initializer_seconds', 'mean_refresh_seconds'):
        values = np.array([r[key] for r in rows])
        summary[key] = dict(mean=float(values.mean()), median=float(np.median(values)), p95=float(np.quantile(values, .95)))
    return summary
