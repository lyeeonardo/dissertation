"""Isolated native H4 with separately supplied planning and H1 evidence engines.

No model loading, fitting, calibration, simulator, or current-condition input.
Only native denoising_errors is routed to the evidence backbone. Native action
proposal/refinement, reward scoring, hazard and categorical update are inherited.
The caller binds actual checkpoint and original-normalization identities.
"""
from __future__ import annotations

from numbers import Integral
from pathlib import Path
import math
import sys

import torch

from work2.native import optimized_proposal_control as native
from work2.native.unified_comparison_control import (
    UnifiedComparisonController, UnifiedComparisonPolicy, method_recipe,
)

VARIANTS = ('original_baseline', 'replay_native_evidence', 'original_evidence')
EXTRA_DETAIL_KEYS = ('proposal_conditions', 'selected_proposal_condition',
                     'evidence_source_code', 'evidence_temperature', 'evidence_query_seed')
EVIDENCE_SOURCE_CODES = {'original_baseline': 0, 'replay_native_evidence': 1, 'original_evidence': 0}
COUNT_KEYS = ('forward_calls', 'forward_examples')


def _snapshot(engine):
    value = engine.stats.snapshot()
    for key in COUNT_KEYS:
        if not isinstance(value[key], Integral) or value[key] < 0:
            raise ValueError('Native engine must expose nonnegative exact query counters')
    return value


class RoutedStats:
    """Native backbone counts are summed once per distinct supplied engine."""
    def __init__(self, router): self.router = router

    def snapshot(self):
        snapshots = [_snapshot(e) for e in self.router.unique_engines]
        return {key: sum(int(s[key]) for s in snapshots) for key in COUNT_KEYS}


class EvidenceRouter:
    """Single-device native engine surface with an explicit evidence-only route.

    Owns the supplied engines for close(). Use these dedicated engines only for
    this policy while accounting; direct external calls are not route-attributed.
    """
    def __init__(self, planning_engine, evidence_engine):
        if planning_engine.device != evidence_engine.device:
            raise ValueError('Planning and evidence engines must share the native device')
        if planning_engine.batch_size != evidence_engine.batch_size:
            raise ValueError('Planning and evidence query batch sizes must match')
        self.planning_engine, self.evidence_engine = planning_engine, evidence_engine
        self.unique_engines = tuple({id(e):e for e in (planning_engine,evidence_engine)}.values())
        self.device, self.model, self.batch_size = planning_engine.device, planning_engine.model, planning_engine.batch_size
        self.route_counts = {role:dict(calls=0,forward_calls=0,forward_examples=0) for role in ('planning','evidence')}
        self.stats, self.closed = RoutedStats(self), False
        for engine in self.unique_engines: _snapshot(engine)

    def _call(self, role, engine, name, *args, **kwargs):
        if self.closed: raise RuntimeError('Engine router is closed')
        before = _snapshot(engine); self.route_counts[role]['calls'] += 1
        try:
            return getattr(engine,name)(*args,**kwargs)
        finally:
            after = _snapshot(engine)
            for key in COUNT_KEYS:
                delta = int(after[key])-int(before[key])
                if delta < 0: raise RuntimeError('Native query counters decreased')
                self.route_counts[role][key] += delta

    def joint_sample(self, *args, **kwargs):
        return self._call('planning',self.planning_engine,'joint_sample',*args,**kwargs)

    def refine_joint(self, *args, **kwargs):
        return self._call('planning',self.planning_engine,'refine_joint',*args,**kwargs)

    def predict(self, *args, **kwargs):
        return self._call('planning',self.planning_engine,'predict',*args,**kwargs)

    def denoising_errors(self, *args, **kwargs):
        return self._call('evidence',self.evidence_engine,'denoising_errors',*args,**kwargs)

    def query_breakdown(self):
        return dict(unique_backbones=len(self.unique_engines),
                    routes={role:dict(value) for role,value in self.route_counts.items()},
                    engines=[dict(roles=[role for role,engine in [('planning',self.planning_engine),('evidence',self.evidence_engine)] if engine is item],
                                  counters=_snapshot(item)) for item in self.unique_engines],
                    semantics='Exact counter differences around routed calls; each native engine is counted once in aggregate, including failed calls. Dedicated engines only.')

    def close(self):
        if self.closed: return
        self.closed = True
        errors=[]
        for engine in self.unique_engines:
            try: engine.close()
            except BaseException as error: errors.append(error)
        if errors: raise errors[0]


def _configuration(variant, planning_engine, evidence_engine, planning_temperature, evidence_temperature):
    if variant not in VARIANTS: raise ValueError('Unknown isolated belief variant')
    if not math.isfinite(float(planning_temperature)) or planning_temperature <= 0:
        raise ValueError('Planning checkpoint requires its fixed positive CAL temperature')
    if variant == 'original_evidence':
        if evidence_engine is None or evidence_engine is planning_engine:
            raise ValueError('Split evidence requires a distinct explicitly supplied original engine')
        if evidence_temperature is None: raise ValueError('Split evidence requires its own fixed CAL temperature')
    else:
        if evidence_engine is not None and evidence_engine is not planning_engine:
            raise ValueError('Native evidence parity requires the exact planning engine object')
        evidence_engine = planning_engine
        if evidence_temperature is None: evidence_temperature = planning_temperature
        if evidence_temperature != planning_temperature:
            raise ValueError('Native evidence parity requires the native checkpoint temperature')
    if not math.isfinite(float(evidence_temperature)) or evidence_temperature <= 0:
        raise ValueError('Evidence temperature must be positive and finite')
    return evidence_engine,float(planning_temperature),float(evidence_temperature)


class BeliefRepairController(UnifiedComparisonController):
    def __init__(self, variant, planning_engine, *, planning_temperature,
                 evidence_engine=None, evidence_temperature=None, prior=None):
        evidence_engine,planning_temperature,evidence_temperature = _configuration(
            variant,planning_engine,evidence_engine,planning_temperature,evidence_temperature)
        router = EvidenceRouter(planning_engine,evidence_engine)
        # Reward-only H4 never uses temperature to alter a proposal or its score.
        # The native scalar is needed by observe; no information score is enabled.
        super().__init__(router,'refine_h4_reward',temperature=evidence_temperature,information_weight=0.,prior=prior)
        self.variant,self.planning_temperature = variant,planning_temperature

    def _provenance(self, seed):
        return dict(evidence_source_code=EVIDENCE_SOURCE_CODES[self.variant],
                    evidence_temperature=self.temperature,evidence_query_seed=int(seed))

    @torch.inference_mode()
    def decide(self, history, *, seed):
        result = super().decide(history,seed=seed)
        result['proposal_conditions'] = torch.full((3,),result['sampled_condition'],dtype=torch.long,device=self.engine.device)
        result['selected_proposal_condition'] = result['sampled_condition']
        result.update(self._provenance(seed))
        return result

    @torch.inference_mode()
    def observe(self, history, action, next_observation, reward, *, seed):
        # Entire native validation, actual-probe generation and Bayes update path.
        result = super().observe(history,action,next_observation,reward,seed=seed)
        result.update(self._provenance(seed))
        return result


class BeliefRepairPolicy(UnifiedComparisonPolicy):
    """Native act/observe and pending/history/truth guards; caller supplies engines."""
    receives_truth = False
    records_beliefs = True
    history_length = 8

    def __init__(self, variant, planning_engine, seed, *, planning_temperature,
                 evidence_engine=None,evidence_temperature=None):
        if isinstance(seed,bool) or not isinstance(seed,Integral) or seed < 0:
            raise ValueError('A nonnegative integer case seed is required')
        self.controller = BeliefRepairController(variant,planning_engine,planning_temperature=planning_temperature,
            evidence_engine=evidence_engine,evidence_temperature=evidence_temperature)
        self.variant,self.method,self.seed = variant,'refine_h4_reward',int(seed)
        self.engine,self.device,self.model = self.controller.engine,planning_engine.device,planning_engine.model
        self._step = 0

    def recipe(self):
        result = method_recipe('refine_h4_reward',temperature=self.controller.temperature,
                               information_weight=0.,query_batch_size=self.engine.batch_size)
        result.update(variant=self.variant,planning_backbone='original_62000' if self.variant=='original_baseline' else 'replay_05000',
            evidence_backbone='replay_05000' if self.variant=='replay_native_evidence' else 'original_62000',
            planning_temperature=self.controller.planning_temperature,
            planning_temperature_role='Native reward-only H4 score is independent of evidence temperature; all information scoring is disabled',
            evidence_source_codes=dict(original_62000=0,replay_05000=1),
            evidence_routing='Only denoising_errors routes to the supplied evidence engine; all proposal/refinement/prediction calls use planning engine',
            evidence_update='Unchanged native categorical update using actual selected action, next observation and public reward',
            evidence_noise='Unchanged CPU actual stream; sampling_seed + native_step; shape[1,4,2,18], four shared hypothesis probes',
            raw_energy_trace='denoising_errors[8]; log_likelihood=-raw_energy/evidence_temperature; no rank correction',
            evidence_provenance_trace=['evidence_source_code','evidence_temperature','evidence_query_seed'],
            authoritative_condition_trace='proposal_conditions[3]; all three native refinements share sampled_condition',
            distinct_backbones=len(self.engine.unique_engines),
            expected_query_cost=dict(planning_forward_calls=23,planning_forward_examples=235,
                evidence_forward_calls=1,evidence_forward_examples=32,total_forward_calls=24,total_forward_examples=267),
            query_cost_scope='Per native H4 step at query batch256; eight sampling steps plus final sampler pass; actual counters remain authoritative',
            checkpoint_identity_binding='Caller must bind both exact EMA checkpoints, shared normalization and existing CAL receipts; variant names are not identity verification')
        return result

    to_spec = recipe
