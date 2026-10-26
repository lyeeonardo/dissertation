from __future__ import annotations


import numpy as np

from .belief_diagnostics import belief_metrics

def recovery_measure(speed, target_speed, window=150):
    """Ten-step smoothing, 20-step hold; failures remain censored and capped."""
    if not np.isfinite(target_speed) or target_speed <= 0.1:
        return {
            "reference_valid": False,
            "recovered": False,
            "delay": None,
            "capped_delay": window,
            "sustained": False,
        }
    speed = np.asarray(speed, np.float64)[:window]
    padded = np.pad(speed, (0, max(0, window - len(speed))), constant_values=0)
    smoothed = np.convolve(padded, np.ones(10) / 10, mode="valid")
    above = smoothed >= 0.9 * target_speed
    # Index is the observation when the required hold has actually been confirmed.
    for start in range(max(0, len(above) - 19)):
        if above[start : start + 20].all():
            confirmation = start + 10 + 20 - 1
            return {
                "reference_valid": True,
                "recovered": True,
                "delay": confirmation,
                "validated_onset_delay": start + 10,
                "capped_delay": confirmation,
                "sustained": bool(above[start:].all()),
            }
    return {
        "reference_valid": True,
        "recovered": False,
        "delay": None,
        "capped_delay": window,
        "sustained": False,
    }

def episode_metrics(case, rewards, speeds, inverted, reference_speeds=None, beliefs=None):
    cap = case["steps"]
    reward = np.pad(np.asarray(rewards), (0, max(0, cap - len(rewards))))[:cap]
    speed = np.pad(np.asarray(speeds), (0, max(0, cap - len(speeds))))[:cap]
    events = []
    for event in case["events"]:
        start = event["step"]
        row = {
            **event,
            "reward_150": float(reward[start : start + 150].sum()),
            "mean_speed_150": float(speed[start : start + 150].mean()),
        }
        if reference_speeds is not None:
            row["recovery"] = recovery_measure(
                speed[start : start + 150], reference_speeds[str(event["condition"])]
            )
        events.append(row)
    result = {
        "case_id": case["case_id"],
        "family": case["family"],
        "initial_condition": case["initial_condition"],
        "return": float(reward.sum()),
        "native_steps": len(rewards),
        "mean_speed": float(speed.mean()),
        "inverted_fraction": float(np.mean(inverted)) if len(inverted) else 0,
        "event_mean_reward_150": float(np.mean([e["reward_150"] for e in events]))
        if events
        else None,
        "events": events,
    }
    if beliefs is not None:
        posterior = np.asarray(beliefs, np.float64)[1 : len(rewards) + 1]
        labels = np.full(len(rewards), case["initial_condition"], np.int64)
        for event in case["events"]:
            labels[event["step"] :] = event["condition"]
        result["belief"] = belief_metrics(posterior, labels, case["events"])
    return result
