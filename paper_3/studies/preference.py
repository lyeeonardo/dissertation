"""Isolated oxygen-preference addition to the frozen recommended GenAIF.

This module loads no model at import time and never launches an experiment.
The two registered arms use weight 0 or .01.  The 2 mg/L target follows the
native nominal PI; .5 mg/L is a declared experimental preference width, not
sensor noise, a fitted uncertainty, or a regulatory interval.

The native act method runs unchanged, including imagined reward evidence.
Only its already-generated candidate bank is reranked, before any action is
sent to the environment.  The propagated state is switched to that same
candidate's existing forecast.  No extra model query or random draw is made.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import numpy as np
import torch



from bsm1_offline_benchmark.stock_joint_control import JointController
from research_shared_model import defaults
from research_shared_model.efe_exploration.selective_mean_control import SelectiveMeanController

SCHEMA = 'frozen_selective_mean_reward_oxygen_preference_v1'
DO5_INDEX = 59
REGISTERED_WEIGHTS = (0., .01)


@dataclass(frozen=True)
class Preference:
    weight: float = .01
    mean: float = 2.
    width: float = .5

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in (self.weight, self.mean, self.width))
                or self.weight < 0 or self.mean < 0 or self.width <= 0):
            raise ValueError('Finite nonnegative weight/target and positive preference width required')


def oxygen_penalties(do5, weights, discounts, *, mean=2., width=.5):
    """Return C discounted expectations; square each K,H sample before averaging."""
    Preference(weight=0., mean=mean, width=width)
    if do5.ndim != 3 or weights.shape != (do5.shape[1],) or discounts.shape != (do5.shape[2],):
        raise ValueError('Expected candidate x particle x horizon oxygen, particle weights, horizon discounts')
    values, w, d = do5.double(), weights.double(), discounts.double()
    if (not all(bool(torch.isfinite(x).all()) for x in (values, w, d))
            or bool((w < 0).any()) or bool((d < 0).any())
            or not bool(torch.isclose(w.sum(), w.new_tensor(1.), atol=1e-12, rtol=0))):
        raise ValueError('Finite predictions, normalized nonnegative weights and nonnegative discounts required')
    loss = .5 * ((values-float(mean))/float(width)).square()
    return (loss*w[None, :, None]*d[None, None, :]).sum((1, 2))


class PreferenceJointController(JointController):
    """Run native act first, then choose from its exact saved candidate bank."""

    def __init__(self, *args, preference=None, **kwargs):
        self.preference = preference if preference is not None else Preference()
        if not isinstance(self.preference, Preference):
            raise TypeError('preference must be a Preference configuration')
        self._selection_actions = self._selection_prediction = None
        super().__init__(*args, **kwargs)

    def _proposal_bank(self, history, horizon):
        actions, generating_indices = super()._proposal_bank(history, horizon)
        self._selection_actions = actions
        return actions, generating_indices

    def _forecast(self, history, actions):
        prediction = super()._forecast(history, actions)
        self._selection_prediction = prediction
        return prediction

    @torch.inference_mode()
    def act(self, public55, previous_action=None, current_truth=None):
        started = time.perf_counter()
        self._selection_actions = self._selection_prediction = None
        original_action = super().act(public55, previous_action, current_truth)
        actions, prediction = self._selection_actions, self._selection_prediction
        if actions is None or prediction is None or self.pending is None:
            raise RuntimeError('Native act must produce one candidate bank, forecast and pending transition')
        raw = self.last_diagnostics
        original_selected = int(raw['selected_candidate'])
        discounts = self.discount ** torch.arange(self.horizon, device=self.device, dtype=torch.float64)
        do5 = prediction['next_private'][..., DO5_INDEX]
        penalties = oxygen_penalties(do5, self.weights, discounts,
            mean=self.preference.mean, width=self.preference.width)
        native_efe = torch.as_tensor(raw['efe'], device=self.device, dtype=torch.float64)
        # The explicit zero branch preserves the native score/tie order exactly.
        combined = native_efe if self.preference.weight == 0 else native_efe+self.preference.weight*penalties
        selected = original_selected if self.preference.weight == 0 else int(combined.argmin())
        chosen = original_action
        if selected != original_selected:
            chosen = actions[selected, 0].cpu().numpy().astype(np.float64)
            self.pending['action'] = chosen.copy()
            self.pending['next_particles'] = prediction['next_private'][selected, :, 0].clone()
        if not np.array_equal(chosen, self.pending['action']):
            raise AssertionError('Selected action must be the action retained for the native real update')
        raw.update(
            native_selected_candidate=original_selected, selected_candidate=selected,
            native_efe=native_efe.cpu().tolist(), efe=combined.cpu().tolist(),
            preference_penalties=penalties.cpu().tolist(),
            preference_weight=float(self.preference.weight), preference_mean=float(self.preference.mean),
            preference_width=float(self.preference.width),
            preference_width_definition='Experimental tolerance in mg/L; not sensor noise or calibrated uncertainty',
            preference_state_index=DO5_INDEX,
            preference_definition='sum_h discount^h sum_k w_k * .5*((predicted_DO5[k,h]-mean)/width)^2',
            candidate_actions=actions.cpu().tolist(),
            candidate_do5_forecasts=do5.cpu().tolist(),
            candidate_reward_forecasts=prediction['rewards'].cpu().tolist(),
            forecast_discounts=discounts.cpu().tolist(), selected_action=chosen.tolist(),
            native_model_queries_unchanged=True, native_sampling_unchanged=True,
            measured_cost_retained=True, additional_sensor_readings=0,
            decision_seconds=time.perf_counter()-started,
        )
        return chosen

    def runtime_state(self):
        return dict(super().runtime_state(), preference_recipe=dict(schema=SCHEMA, **asdict(self.preference)))

    def restore_runtime_state(self, state):
        candidate = state.get('joint_state', state)
        if candidate.get('preference_recipe') != dict(schema=SCHEMA, **asdict(self.preference)):
            raise ValueError('Runtime must retain the same preference target, width and weight')
        super().restore_runtime_state(state)
        self._selection_actions = self._selection_prediction = None


AUDIT_KEYS = (
    'predicted_returns', 'information_l1', 'preference_penalties', 'native_efe', 'efe',
    'preference_weight', 'preference_mean', 'preference_width', 'preference_width_definition',
    'preference_state_index', 'preference_definition', 'candidate_actions', 'candidate_first_actions',
    'candidate_do5_forecasts', 'candidate_reward_forecasts', 'forecast_discounts', 'prior_weights',
    'native_selected_candidate', 'selected_candidate', 'selected_action', 'finite_preference_discount',
    'native_model_queries_unchanged', 'native_sampling_unchanged', 'measured_cost_retained',
    'additional_sensor_readings', 'proposal_generating_hypotheses',
)


class PreferenceRewardController(SelectiveMeanController):
    """Keep the final mean repair and expose auditable, correctly attributed scores."""

    def act(self, public55, previous_action=None):
        started = time.perf_counter()
        action = super().act(public55, previous_action)
        raw = self.controller.last_diagnostics
        self.last_diagnostics.update({key: copy.deepcopy(raw[key]) for key in AUDIT_KEYS})
        returns = np.asarray(raw['predicted_returns'], dtype=np.float64)
        info = np.asarray(raw['information_l1'], dtype=np.float64)
        penalties = np.asarray(raw['preference_penalties'], dtype=np.float64)
        scores = np.asarray(raw['efe'], dtype=np.float64)
        commands = np.asarray(raw['candidate_first_actions'], dtype=np.float64)
        selected, native = int(raw['selected_candidate']), int(raw['native_selected_candidate'])
        reward = int(returns.argmax())
        reward_preference = int((-returns+raw['preference_weight']*penalties).argmin())
        info_difference = float(np.max(np.abs(commands[selected]-commands[reward_preference])))
        preference_difference = float(np.max(np.abs(commands[selected]-commands[native])))
        self.last_diagnostics.update(
            reward_winner=reward, reward_preference_winner=reward_preference,
            information_changed_ranking=selected != reward_preference,
            information_changed_first_action=info_difference > 1e-7,
            maximum_first_action_difference=info_difference,
            preference_changed_ranking=selected != native,
            preference_changed_first_action=preference_difference > 1e-7,
            preference_maximum_first_action_difference=preference_difference,
            selected_minus_reward_winner_return=float(returns[selected]-returns[reward]),
            selected_information_advantage=float(info[selected]-info[reward]),
            lambda_information_advantage=float(raw['information_weight']*(info[selected]-info[reward])),
            weighted_preference_advantage=float(raw['preference_weight']*(penalties[reward]-penalties[selected])),
            selected_score_gain=float(scores[reward]-scores[selected]),
            information_score_gain=float(scores[reward_preference]-scores[selected]),
            preference_score_gain_vs_native_winner=float(scores[native]-scores[selected]),
            same_bank_scope=('Preference effect: combined versus native reward+information winner. '
                'Information effect: combined versus reward+preference winner. '
                'These are within-bank score/action diagnostics, not physical counterfactuals.'),
            action_selection_seconds=time.perf_counter()-started,
        )
        return action


def make_controller(weight=.01, device='cuda', remaining=1344):
    """Construct one frozen-model arm; the caller owns all execution/supervision."""
    if int(remaining) != remaining or remaining != 1344:
        raise ValueError('This fixed pilot retains the native 1344-decision full episode')
    preference = Preference(weight=float(weight))
    native = PreferenceJointController(defaults.CHECKPOINT, device=device,
        remaining=remaining, preference=preference, **defaults.controller_recipe())
    return PreferenceRewardController(native, device)


def descriptor(weight=.01):
    preference = Preference(weight=float(weight))
    return dict(schema=SCHEMA, checkpoint=str(defaults.CHECKPOINT),
        controller=dict(defaults.controller_recipe(), remaining=1344), preference=asdict(preference),
        preference_scope='Soft forecast DO5 tracking added to native reward and L1; public55 inference',
        target_source='bsm1_offline_benchmark/dataset_upgrade.py:PublicExpertPolicy',
        preference_width_source='Declared pilot choice, not an existing native tolerance or sensor noise',
        registered_weights=list(REGISTERED_WEIGHTS), new_fits=0, new_calibration=0,
        measured_cost_retained=True, additional_sensor_readings=0, default_changed=False)
