from __future__ import annotations


import numpy as np

def trace_histories(trace, anchors, length=8):
    """History ending at observation t contains actions/rewards strictly before t."""
    anchors = np.asarray(anchors, np.int64)
    indices = anchors[:, None] + np.arange(-length + 1, 1)[None]
    mask = indices >= 0
    observations = trace["observations"][indices.clip(0)] * mask[..., None]
    action_indices = indices[:, :-1]
    action_mask = action_indices >= 0
    actions = trace["actions"][action_indices.clip(0)] * action_mask[..., None]
    rewards = trace["rewards"][action_indices.clip(0)] * action_mask
    return dict(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rewards.astype(np.float32),
        mask=mask.astype(np.float32),
    )
