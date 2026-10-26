"""Frozen v2 comparison: five controllers, three explicitly distinct cohorts.

Preparation does not launch experiments. The enclosing supervisor is responsible
for fresh processes, scientific-progress monitoring and stopping without retry.
"""

from __future__ import annotations

from .dataset_upgrade import PublicExpertPolicy

class PublicExpert:
    """PI-compatible runtime state, including feedforward and controller memory."""
    def __init__(self, specification, seed=17):
        self.policy = PublicExpertPolicy(specification, seed)
        self.step = 0

    def act(self, observation, previous_action=None):
        action = self.policy.action(self.step, observation)
        self.step += 1
        return action
