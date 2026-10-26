"""Native metric reduction for the retained revised information pair."""
import numpy as np
from .reduced_benchmark_metrics import _vector,recovery,timing_metrics
FAMILIES=('stationary','single_factor','repeated_factor')
def require(condition,message):
    if not condition: raise ValueError(message)

def episode_metrics(case, arrays, reference_speeds, native_dt=0.05):
    require(native_dt == 0.05 and case["steps"] == 1000, "fixed HalfCheetah time/cap")
    require(
        set(reference_speeds) == {str(i) for i in range(8)}
        and all(np.isfinite(v) and v > 0.1 for v in reference_speeds.values()),
        "eight valid frozen reference speeds",
    )
    rewards = _vector(arrays, "rewards").astype(np.float64)
    n = len(rewards)
    require(0 < n <= 1000, "completed native length")
    speeds = _vector(arrays, "speed", n)
    terminated, truncated = (_vector(arrays, key, n) for key in ("terminated", "truncated"))
    require(np.isin(terminated, (0, 1)).all() and np.isin(truncated, (0, 1)).all(), "done flags")
    done = terminated.astype(bool) | truncated.astype(bool)
    require(
        not done[:-1].any() and (n == 1000 or terminated[-1]),
        "incomplete, early-truncated or post-terminal native trace",
    )
    for key in ("observations", "actions", "beliefs", "conditions", "log_likelihood", "priors"):
        if key in arrays:
            values = np.asarray(arrays[key])
            require(
                len(values) == n + int(key in ("observations", "beliefs"))
                and np.isfinite(values).all(),
                f"invalid native {key}",
            )
    if "conditions" in arrays:
        expected = np.full(n, case["initial_condition"])
        for event in case["events"]:
            expected[event["step"] :] = event["condition"]
        require(np.array_equal(expected, arrays["conditions"]), "hidden schedule differs")
    require(
        case["family"] in FAMILIES
        and len(case["events"])
        == {"stationary": 0, "single_factor": 1, "repeated_factor": 3}[case["family"]],
        "family/event count differs",
    )
    events = []
    for event in case["events"]:
        start = event["step"]
        speed = speeds[start : start + 150]
        events.append(
            dict(
                **event,
                reached=start < n,
                observed_steps=len(speed),
                reward_150=float(rewards[start : start + 150].sum()),
                recovery=recovery(speed, reference_speeds[str(event["condition"])], native_dt),
            )
        )

    def average(values):
        return float(np.mean(values)) if events else None

    return dict(
        **{
            k: case[k]
            for k in (
                "case_id",
                "initial_condition",
                "family",
                "factor_mask",
                "reset_seed",
                "sampling_seed",
            )
        },
        events=events,
        event_reward_150=average([e["reward_150"] for e in events]),
        sustained_recovery_rate=average([e["recovery"]["sustained"] for e in events]),
        recovery_time_steps=average([e["recovery"]["capped_delay_steps"] for e in events]),
        recovery_time_seconds=average([e["recovery"]["capped_delay_seconds"] for e in events]),
        nonrecovery_rate=average([not e["recovery"]["recovered"] for e in events]),
        native_steps=n,
        native_seconds=n * native_dt,
        native_termination=int(terminated[-1]),
        native_truncation=int(truncated[-1]),
        unreached_switches=sum(not e["reached"] for e in events),
        timing=timing_metrics(arrays, n, native_dt),
        **{"return": float(rewards.sum())},
    )
