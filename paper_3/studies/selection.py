"""Bounded final-controller selection ablation; upstream sources stay unchanged."""
from pathlib import Path
import sys

import time
import numpy as np
import torch
from bsm1_offline_benchmark.stock_joint_control import JointController, entropy
from research_shared_model import defaults
from research_shared_model.efe_exploration.selective_mean_control import SelectiveMeanController

class SelectionController(JointController):
    def __init__(self, forecast_ranking, device='cuda'):
        self.forecast_ranking=bool(forecast_ranking)
        self.forecast_calls=0
        super().__init__(defaults.CHECKPOINT,device=device,remaining=1344,**defaults.controller_recipe())

    def _forecast(self, history, actions):
        self.forecast_calls += 1
        assert tuple(actions.shape[:2]) == (3,30), 'Unexpected forecast/fallback'
        return super()._forecast(history,actions)

    def runtime_state(self):
        return dict(super().runtime_state(), selection_ablation=dict(
            forecast_ranking=self.forecast_ranking,forecast_calls=self.forecast_calls))

    def restore_runtime_state(self, state):
        raise RuntimeError('This bounded ablation does not permit resume')

    @torch.inference_mode()
    def act(self, public55, previous_action=None, current_truth=None):
        if self.pending is not None:
            raise RuntimeError("Observe the previous executed transition before selecting again")
        if self.remaining <= 0:
            raise RuntimeError("Known evaluation episode has ended")
        started = time.perf_counter()
        history = self._root(public55, previous_action, current_truth)
        # The plant is continuing; the reporting record's end is not a physical
        # terminal. Stock Han and this controller retain H30 at every decision.
        horizon = self.horizon
        actions, generating_indices = self._proposal_bank(history, horizon)
        prediction = self._forecast(history, actions)
        discounts = self.discount ** torch.arange(horizon, device=self.device, dtype=torch.float64)
        rewards = (prediction["rewards"].to(torch.float64) * discounts).sum(-1)
        expected_return = (rewards * self.weights[None]).sum(-1)
        information, info_details = self._information(history, actions, prediction)
        efe = -expected_return - self.information_weight*information
        native_selected = int(efe.argmin())
        selected = native_selected if self.forecast_ranking else 0
        chosen = actions[selected, 0].cpu().numpy().astype(np.float64)
        self.pending = dict(particles=self.particles.clone(), weights=self.weights.clone(),
            history={key: value.clone() for key, value in history.items()}, action=chosen.copy(),
            next_particles=prediction["next_private"][selected, :, 0].clone())
        self.act_count += 1
        self.last_diagnostics = dict(mode=self.mode, act_count=self.act_count,
            transition_count=self.transition_count, current_truth_access=self.mode == "oracle",
            history_requested=self.history_length, history_available=int(history["valid"].sum()),
            horizon=horizon, remaining=self.remaining, passes=self.passes, candidates=self.candidates,
            continuous_hypotheses=len(self.weights), refinement=self.refinement,
            belief_source=("current_oracle_truth" if self.mode == "oracle" else
                "history_initializer_each_decision" if self.mode == "history" else
                "propagated_VFE" if self.belief_update else "propagated_prior"),
            refinement_midpoint=self.passes//2 if self.refinement else None,
            refinement_time=float(self.model.config.maximum_time/2) if self.refinement else None,
            partial_updates=self.passes-self.passes//2 if self.refinement else None,
            proposal_denoiser_evaluations=self.passes+2*(self.passes-self.passes//2) if self.refinement else self.candidates*self.passes,
            forecast_denoiser_evaluations=self.candidates*len(self.weights)*self.passes,
            proposal_generating_hypotheses=generating_indices, original_candidate=0,
            proposal_batch_sizes=[1, 2 if self.refinement else self.candidates-1],
            candidate_first_actions=actions[:, 0].cpu().tolist(),
            predicted_returns=expected_return.cpu().tolist(), information_l1=information.cpu().tolist(),
            efe=efe.cpu().tolist(), selected_candidate=selected,
            reward_action_contrast=float(expected_return.max()-expected_return.min()),
            information_action_contrast=float(information.max()-information.min()),
            prior_weights=self.weights.cpu().tolist(), belief_entropy=float(entropy(self.weights)),
            belief_ess=float(1/self.weights.square().sum()), tau=self.tau,
            information_weight=self.information_weight, finite_preference_discount=self.discount,
            decision_seconds=time.perf_counter()-started,
            uncertainty_definition="finite continuous-state hypotheses; no categorical hazard",
            information_definition="one-step weighted entropy reduction from common prior; negative values retained")
        if info_details is not None:
            self.last_diagnostics["information_details"] = {key: value.cpu().tolist() for key, value in info_details.items()}
        self.last_diagnostics.update(forecast_ranking=self.forecast_ranking,
            native_forecast_winner=native_selected,
            cached_action_matches_selected=bool(np.array_equal(chosen, self.pending['action'])),
            cached_states_match_selected=bool(torch.equal(self.pending['next_particles'], prediction['next_private'][selected, :, 0])),
            full_horizon_forecasts_preserved=True)
        return chosen

class AuditedMeanController(SelectiveMeanController):
    def act(self, public55, previous_action):
        action=super().act(public55,previous_action)
        raw=self.controller.last_diagnostics
        commands=np.asarray(raw['candidate_first_actions'])
        selected=int(raw['selected_candidate']); native=int(raw['native_forecast_winner'])
        reward=int(np.asarray(raw['predicted_returns']).argmax())
        shadow_difference=float(np.max(np.abs(commands[native]-commands[reward])))
        actual_difference=float(np.max(np.abs(commands[selected]-commands[reward])))
        on=self.controller.forecast_ranking
        for name in ('selected_information_advantage','lambda_information_advantage','selected_score_gain','maximum_first_action_difference'):
            if name in self.last_diagnostics:
                self.last_diagnostics['actual_selection_vs_reward_'+name]=self.last_diagnostics.pop(name)
        # OFF selection is forced by the experiment, never an information effect.
        self.last_diagnostics.update(forecast_ranking=on,native_forecast_winner=native,
            selected_candidate=selected,original_candidate=0,
            information_changed_ranking=bool(on and native!=reward),
            information_changed_first_action=bool(on and shadow_difference>1e-7),
            shadow_information_changed_ranking=bool(native!=reward),
            shadow_information_changed_first_action=bool(shadow_difference>1e-7),
            selection_vs_reward_first_action_difference=actual_difference,
            forced_original_differs_from_forecast=bool(not on and np.max(np.abs(commands[0]-commands[native]))>1e-7),
            candidate_first_actions=commands.tolist(),predicted_returns=raw['predicted_returns'],
            information_l1=raw['information_l1'],efe=raw['efe'],
            prior_weights=raw['prior_weights'],
            cached_action_matches_selected=raw['cached_action_matches_selected'],
            cached_states_match_selected=raw['cached_states_match_selected'],
            full_horizon_forecasts_preserved=True,
            forecast_denoiser_evaluations=raw['forecast_denoiser_evaluations'],
            proposal_generating_hypotheses=raw['proposal_generating_hypotheses'])
        assert np.array_equal(action,commands[selected])
        assert raw['cached_action_matches_selected'] and raw['cached_states_match_selected']
        assert selected==(native if on else 0)
        return action

    def observe(self,next_public55,executed_action,raw_cost):
        native=self.controller
        assert native.pending is not None
        assert np.allclose(np.asarray(executed_action,dtype=np.float32),native.pending['action'],atol=1e-7,rtol=0)
        forecast_calls=native.forecast_calls
        result=super().observe(next_public55,executed_action,raw_cost)
        assert native.forecast_calls==forecast_calls==native.transition_count
        self.last_diagnostics.update(executed_action_matches_cache=True,
            native_transition_count=native.transition_count,
            forecast_calls=native.forecast_calls,no_observe_forecast_fallback=True,
            posterior_weights=result['posterior_weights'])
        return result


def make_controller(arm,device='cuda'):
    if arm not in ('on','off'):raise ValueError(arm)
    return AuditedMeanController(SelectionController(arm=='on',device),device)
