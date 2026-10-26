"""Record per-decision diagnostics around the native GenAIF controller."""

from __future__ import annotations

import copy

import time

import numpy as np

from bsm1_offline_benchmark import stock_shared_evaluate as evaluator

class DecisionLoggingController:
    """Delegate native behavior, retaining only compact per-transition diagnostics."""
    def __init__(self, controller, device):
        self.controller = controller; self.device = device; self.last_diagnostics = {}

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def act(self, public55, previous_action):
        started = time.perf_counter()
        action = self.controller.act(public55, previous_action)
        evaluator._synchronize(self.device)
        elapsed = time.perf_counter()-started
        raw = self.controller.last_diagnostics
        returns = np.asarray(raw['predicted_returns']); information = np.asarray(raw['information_l1'])
        commands = np.asarray(raw['candidate_first_actions'])
        selected = int(raw['selected_candidate']); reward_winner = int(returns.argmax())
        difference = float(np.max(np.abs(commands[selected]-commands[reward_winner])))
        delta_r = float(returns[selected]-returns[reward_winner])
        delta_i = float(information[selected]-information[reward_winner])
        weighted_i = float(raw['information_weight'])*delta_i
        self.last_diagnostics = dict(step=int(raw['transition_count']), selected_candidate=selected,
            reward_winner=reward_winner, information_changed_ranking=selected != reward_winner,
            information_changed_first_action=difference > 1e-7, maximum_first_action_difference=difference,
            selected_minus_reward_winner_return=delta_r, selected_information_advantage=delta_i,
            lambda_information_advantage=weighted_i, selected_score_gain=delta_r+weighted_i,
            reward_action_contrast=float(raw['reward_action_contrast']),
            information_action_contrast=float(raw['information_action_contrast']),
            prior_ess=float(raw['belief_ess']), prior_entropy=float(raw['belief_entropy']),
            information_weight=float(raw['information_weight']), tau=float(raw['tau']),
            action_selection_seconds=elapsed)
        return action

    def observe(self, next_public55, executed_action, raw_cost):
        started = time.perf_counter()
        result = self.controller.observe(next_public55, executed_action, raw_cost)
        evaluator._synchronize(self.device)
        elapsed = time.perf_counter()-started
        self.last_diagnostics.update(inference_update_seconds=elapsed,
            total_controller_seconds=self.last_diagnostics['action_selection_seconds']+elapsed,
            raw_cost=float(raw_cost), normalized_reward=float(result['normalized_reward']),
            vfe_applied=bool(result['applied_update']),
            vfe_gain=float(result['prior_vfe']-result['applied_vfe']),
            vfe_identity_residual=float(result['optimum_identity_residual']),
            posterior_ess=float(result['posterior_ess']), posterior_entropy=float(result['posterior_entropy']))
        return result

    def runtime_state(self):
        return dict(joint_state=self.controller.runtime_state(), compact_last_diagnostics=copy.deepcopy(self.last_diagnostics))

    def restore_runtime_state(self, state):
        self.controller.restore_runtime_state(state['joint_state'])
        self.last_diagnostics = copy.deepcopy(state['compact_last_diagnostics'])
