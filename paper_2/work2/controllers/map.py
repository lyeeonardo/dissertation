"""Isolated prior-MAP H4 proposals with the frozen evidence-package routes.

Only the shared proposal condition changes for the two MAP variants. Native H4
sampling/refinement, all-eight-condition scoring, hazard, actual evidence and
policy timing remain intact. No model loading, fitting, simulator or truth input.
"""
from __future__ import annotations

import hashlib
import importlib.util
from numbers import Integral
from pathlib import Path

import torch


from . import belief as evidence
native = evidence.native
EvidenceRouter = evidence.EvidenceRouter
UnifiedComparisonController = evidence.UnifiedComparisonController

MAP_VARIANTS = ('map_replay_evidence', 'map_original_evidence')
NATIVE_VARIANTS = evidence.VARIANTS
VARIANTS = NATIVE_VARIANTS + MAP_VARIANTS
SOURCE_VARIANTS = {
    **{variant: variant for variant in NATIVE_VARIANTS},
    'map_replay_evidence': 'replay_native_evidence',
    'map_original_evidence': 'original_evidence',
}
EVIDENCE_SOURCE_CODES = {
    variant: evidence.EVIDENCE_SOURCE_CODES[source]
    for variant, source in SOURCE_VARIANTS.items()
}
EXTRA_DETAIL_KEYS = evidence.EXTRA_DETAIL_KEYS
TIE_STREAM = 'h4_map_belief_repair_20261018:prior_map_exact_ties'


def _validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('A nonnegative integer decision seed is required')


def _tie_generator(seed):
    _validate_seed(seed)
    key = f'{TIE_STREAM}:{int(seed)}'.encode('ascii')
    keyed_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
    return torch.Generator(device='cpu').manual_seed(keyed_seed)


def ranked_conditions(prior, seed):
    """Descending pre-action prior, uniformly permuted only within exact ties.

    A uniform CPU permutation followed by stable sorting gives each of k exactly
    tied maxima probability 1/k of first place. No epsilon perturbation, global
    RNG use, label-order preference or current-condition input is involved.
    """
    probability = native._belief(prior, torch.device('cpu'))
    permutation = torch.randperm(8, generator=_tie_generator(seed))
    order = torch.argsort(probability[permutation], descending=True, stable=True)
    return permutation[order].to(device=prior.device)


class MapBeliefRepairController(evidence.BeliefRepairController):
    """Frozen evidence controller with a proposal-only MAP condition selector."""

    def __init__(self, variant, planning_engine, *, planning_temperature,
                 evidence_engine=None, evidence_temperature=None, prior=None):
        if variant not in VARIANTS:
            raise ValueError('Unknown isolated MAP belief variant')
        super().__init__(SOURCE_VARIANTS[variant], planning_engine,
                         planning_temperature=planning_temperature,
                         evidence_engine=evidence_engine,
                         evidence_temperature=evidence_temperature, prior=prior)
        self.variant = variant

    def _provenance(self, seed):
        return dict(evidence_source_code=EVIDENCE_SOURCE_CODES[self.variant],
                    evidence_temperature=self.temperature,
                    evidence_query_seed=int(seed))

    @torch.inference_mode()
    def decide(self, history, *, seed):
        _validate_seed(seed)
        if self.variant in NATIVE_VARIANTS:
            return super().decide(history, seed=seed)
        if self._pending is not None:
            raise RuntimeError('Observe the preceding actual transition before replanning')
        public = native.public_history(history, self.engine.device)
        if len(public['mask']) != 1:
            raise ValueError('A proposal bank uses exactly one public root')
        prior = native.predict_prior(self.belief)
        chosen = ranked_conditions(prior, seed)[0]
        proposal_selector = torch.zeros_like(prior)
        proposal_selector[chosen] = 1.
        # The native condition draw is forced by this one-hot input. Its keyed
        # stream is separate from proposal/refinement/prediction/actual noise.
        bank = native.generate_candidate_bank(
            self.engine, public, proposal_selector, method='refinement', seed=seed)
        bank['proposal_conditions'] = chosen.repeat(3)
        result = native.evaluate_candidates(
            self.engine, public, prior, bank['candidate_actions'],
            temperature=self.temperature, seed=seed, information_horizon=0)
        score = result['expected_reward'].clone()
        native._finite(score, 'selection score')
        selected = int(score.argmax())
        action = bank['candidate_actions'][selected, 0].clone()
        self._pending = dict(prior=prior.clone(), history=public, action=action,
                             seed=int(seed))
        result.update(bank)
        result.update(
            prior=prior, selected_index=selected, selected=selected,
            selected_action=action, selection_score=score, query_seed=int(seed),
            selected_force_scale=1., force_scale_applicable=False,
            selected_original_action_l2=float(
                bank['first_action_distance_from_original'][selected]),
            selected_proposal_condition=int(bank['proposal_conditions'][selected]),
            information=result['information_h1'],
        )
        result.update(self._provenance(seed))
        return result


class MapBeliefRepairPolicy(evidence.BeliefRepairPolicy):
    """Inherits native act/observe, truth isolation and consecutive-step guards."""

    def __init__(self, variant, planning_engine, seed, *, planning_temperature,
                 evidence_engine=None, evidence_temperature=None):
        _validate_seed(seed)
        self.controller = MapBeliefRepairController(
            variant, planning_engine, planning_temperature=planning_temperature,
            evidence_engine=evidence_engine, evidence_temperature=evidence_temperature)
        self.variant, self.method, self.seed = variant, 'refine_h4_reward', int(seed)
        self.engine = self.controller.engine
        self.device, self.model = planning_engine.device, planning_engine.model
        self._step = 0

    def recipe(self):
        result = super().recipe()
        source_variant = SOURCE_VARIANTS[self.variant]
        result.update(
            planning_backbone=('original_62000' if source_variant == 'original_baseline'
                               else 'replay_05000'),
            evidence_backbone=('replay_05000' if source_variant == 'replay_native_evidence'
                               else 'original_62000'),
            evidence_package_variant=source_variant,
            proposal_condition_strategy=(
                'One pre-action hazard-prior MAP condition, with seeded uniform exact ties; '
                'shared by the native original proposal and both refinements'
                if self.variant in MAP_VARIANTS else
                'Unchanged native categorical hazard-prior draw shared by all three plans'),
            proposal_selector_scope='Generation only; full eight-condition hazard prior retained for reward scoring and actual Bayes update',
            proposal_tie_noise=(TIE_STREAM if self.variant in MAP_VARIANTS else None),
            authoritative_condition_trace='proposal_conditions[3]; sampled_condition remains the shared generated label, including for MAP',
            preserved_native_noise='CPU proposal, prediction, termination, refinement and actual-evidence streams unchanged',
        )
        return result

    to_spec = recipe


# Drop-in names for the isolated worker's existing evidence-controller interface.
BeliefRepairController = MapBeliefRepairController
BeliefRepairPolicy = MapBeliefRepairPolicy
