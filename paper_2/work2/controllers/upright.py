"""Fixed upright preference over the frozen MAP replay controller's native bank.

No model loading, new queries, fitting, simulator or current/future truth input.
The only control change is selection among the existing three H4 action plans.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import torch


from . import map as frozen_map
native = frozen_map.native
EvidenceRouter = frozen_map.EvidenceRouter

BASE_VARIANT = 'map_replay_evidence'
CANDIDATE_VARIANT = 'upright_map_replay'
DIAGNOSTIC_VARIANT = 'upright_map_replay_zero'
UPRIGHT_VARIANTS = (CANDIDATE_VARIANT, DIAGNOSTIC_VARIANT)
VARIANTS = (BASE_VARIANT,) + UPRIGHT_VARIANTS
EVIDENCE_SOURCE_CODES = {variant: 1 for variant in VARIANTS}
FROZEN_REWARD_STD = 4.0567890408452785
REWARD_NORMALIZATION_SOURCE = 'checkpoint.normalization'
REWARD_NORMALIZATION_SHA256 = '3dde1d8e522811dfa10af55db5475e3aee7c472f44bc9084dc36a7724c690d87'
PITCH_INDEX = 1
DISCOUNT = 0.99
EXTRA_DETAIL_KEYS = frozen_map.EXTRA_DETAIL_KEYS + (
    'orientation_cost', 'orientation_penalty', 'predicted_observations',
    'kappa', 'orientation_reward_std', 'native_selected_index',
)


def orientation_cost(predicted_observations, prior, valid):
    """Unscaled periodic cost, with native float32 horizon-then-prior reduction.

    Every hypothesis retains its full pre-action prior weight. The same native
    validity mask and discount used for reward also apply to orientation.
    """
    if (predicted_observations.shape != (3, 8, 4, 17)
            or prior.shape != (8,) or valid.shape != (3, 8, 4)):
        raise ValueError('Orientation scoring requires native three-bank/eight-condition/H4 shapes')
    if (predicted_observations.dtype != torch.float32 or prior.dtype != torch.float32
            or valid.dtype != torch.bool):
        raise ValueError('Orientation scoring uses native float32 forecasts/prior and boolean validity')
    if not (predicted_observations.device == prior.device == valid.device):
        raise ValueError('Orientation tensors must share the native device')
    native._finite(predicted_observations, 'predicted public observations')
    native._finite(prior, 'orientation prior')
    discount = DISCOUNT ** torch.arange(4, device=prior.device)
    conditional_cost = ((1. - predicted_observations[..., PITCH_INDEX].cos())
                        * valid * discount).sum(-1)
    cost = (conditional_cost * prior[None]).sum(-1)
    native._finite(cost, 'orientation cost')
    return cost


class UprightMapController(frozen_map.MapBeliefRepairController):
    """Native MAP generation/evidence, with one fixed periodic score penalty.

    ``variant`` remains the frozen MAP route internally; ``upright_variant`` is
    the public candidate/diagnostic identity. This preserves inherited routing.
    """

    def __init__(self, variant, planning_engine, *, planning_temperature,
                 evidence_engine=None, evidence_temperature=None, prior=None):
        if variant not in UPRIGHT_VARIANTS:
            raise ValueError('Unknown upright MAP variant')
        super().__init__(BASE_VARIANT, planning_engine,
                         planning_temperature=planning_temperature,
                         evidence_engine=evidence_engine,
                         evidence_temperature=evidence_temperature, prior=prior)
        self.upright_variant = variant
        self.kappa = 1. if variant == CANDIDATE_VARIANT else 0.

    @torch.inference_mode()
    def decide(self, history, *, seed):
        # Exactly the frozen proposal, prediction and pending-state path. It
        # already exposes the physical decoded observations; no extra query.
        result = super().decide(history, seed=seed)
        cost = orientation_cost(result['observations'], result['prior'], result['valid'])
        penalty = cost * (FROZEN_REWARD_STD * self.kappa)
        # Explicit clone at kappa=0 preserves every native score bit, including
        # signed zero. The zero variant is diagnostic only, never a candidate.
        score = (result['expected_reward'].clone() if self.kappa == 0.
                 else result['expected_reward'] - penalty)
        native._finite(score, 'upright selection score')
        selected = int(score.argmax())  # Native deterministic first-index tie rule.
        action = result['candidate_actions'][selected, 0].clone()
        self._pending['action'] = action
        result.update(
            native_selected_index=result['selected_index'],
            selected_index=selected, selected=selected, selected_action=action,
            selection_score=score, selected_force_scale=1., force_scale_applicable=False,
            selected_original_action_l2=float(
                result['first_action_distance_from_original'][selected]),
            selected_proposal_condition=int(result['proposal_conditions'][selected]),
            orientation_cost=cost, orientation_penalty=penalty,
            predicted_observations=result['observations'],
            kappa=self.kappa, orientation_reward_std=FROZEN_REWARD_STD,
        )
        return result


class UprightMapPolicy(frozen_map.MapBeliefRepairPolicy):
    """Unchanged public act/observe guards and actual selected-action evidence."""

    def __init__(self, variant, planning_engine, seed, *, planning_temperature,
                 evidence_engine=None, evidence_temperature=None):
        frozen_map._validate_seed(seed)
        self.controller = UprightMapController(
            variant, planning_engine, planning_temperature=planning_temperature,
            evidence_engine=evidence_engine, evidence_temperature=evidence_temperature)
        self.variant, self.method, self.seed = BASE_VARIANT, 'refine_h4_reward', int(seed)
        self.engine = self.controller.engine
        self.device, self.model = planning_engine.device, planning_engine.model
        self._step = 0
        # Capture the fixed parent recipe while using its own recognized route
        # name, then expose the separate prospective variant to callers.
        self._native_recipe = super().recipe()
        self.variant = variant

    def recipe(self):
        result = copy.deepcopy(self._native_recipe)
        result.update(
            variant=self.variant, base_variant=BASE_VARIANT,
            diagnostic_only=self.variant == DIAGNOSTIC_VARIANT,
            kappa=self.controller.kappa, orientation_reward_std=FROZEN_REWARD_STD,
            orientation_scale_source=dict(path=str(REWARD_NORMALIZATION_SOURCE),
                sha256=REWARD_NORMALIZATION_SHA256, field='reward_std',
                provenance='Original frozen TRAIN normalization; no probe/DEV/TEST coefficient selection'),
            orientation_pitch_index=PITCH_INDEX,
            orientation_cost='sum_c prior[c] * sum_h valid[j,c,h] * 0.99**h * (1-cos(predicted_observations[j,c,h,1]))',
            selection_score='native expected_reward - kappa * original TRAIN reward_std * orientation_cost',
            score_tie_rule='Native torch.argmax first index; no new random stream',
            native_reward_unchanged=True, extra_model_queries=0,
            orientation_trace=list(EXTRA_DETAIL_KEYS[len(frozen_map.EXTRA_DETAIL_KEYS):]),
        )
        return result

    to_spec = recipe


BeliefRepairController = UprightMapController
BeliefRepairPolicy = UprightMapPolicy
