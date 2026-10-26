"""Scientific implementation used by the Work 2 studies."""
from __future__ import annotations

import numpy as np

def require(condition, message):
    if not condition:
        raise ValueError("Reduced benchmark: " + message)

def _vector(arrays, key, n=None, *, nonnegative=False):
    value = np.asarray(arrays[key])
    require(value.ndim == 1 and (n is None or len(value) == n), f"invalid {key} native shape")
    require(np.isfinite(value).all(), f"nonfinite {key}")
    require(not nonnegative or (value >= 0).all(), f"negative {key}")
    return value

def recovery(speed, reference, native_dt=0.05):
    """Same observed confirmation convention as protocol.recovery_measure, without padding speed."""
    require(np.isfinite(reference) and reference > 0.1, "invalid frozen reference speed")
    speed = np.asarray(speed, np.float64)
    require(speed.ndim == 1 and len(speed) <= 150 and np.isfinite(speed).all(), "invalid speed")
    delay, sustained = None, False
    if len(speed) >= 29:
        above = np.convolve(speed, np.ones(10) / 10, mode="valid") >= 0.9 * reference
        for start in range(len(above) - 19):
            if above[start : start + 20].all():
                delay = start + 29
                sustained = len(speed) == 150 and bool(above[start:].all())
                break
    capped = 150 if delay is None else delay
    return dict(
        reference_valid=True,
        recovered=delay is not None,
        sustained=sustained,
        delay_steps=delay,
        capped_delay_steps=capped,
        capped_delay_seconds=capped * native_dt,
        observed_steps=len(speed),
    )

def timing_metrics(arrays, n, native_dt):
    planning = _vector(arrays, "decision_s", n, nonnegative=True).astype(np.float64)
    update = _vector(arrays, "update_s", n, nonnegative=True).astype(np.float64)
    simulator = _vector(arrays, "simulator_s", n, nonnegative=True).astype(np.float64)
    batch = np.asarray(arrays.get("batch_size", np.ones(n)), np.float64)
    require(batch.shape == (n,) and np.isfinite(batch).all(), "invalid batch sizes")
    require((batch >= 1).all() and np.equal(batch, np.floor(batch)).all(), "invalid batch sizes")
    masks = {}
    for key, default in (
        ("planning_mask", np.ones(n, bool)),
        ("update_mask", np.full(n, "log_likelihood" in arrays, bool)),
    ):
        mask = np.asarray(arrays.get(key, default))
        require(mask.shape == (n,) and np.isin(mask, (0, 1)).all(), f"invalid {key}")
        masks[key] = mask.astype(bool)
    total = planning + update
    planning_calls = int(masks["planning_mask"].sum())
    return dict(
        controller_wall_s=float(total.sum()),
        planning_wall_s=float(planning[masks["planning_mask"]].sum()),
        observation_update_wall_s=float(update.sum()),
        simulator_wall_s=float(simulator.sum()),
        amortized_controller_s=float((total / batch).sum()),
        per_action_mean_s=float(total.mean()),
        per_action_median_s=float(np.median(total)),
        per_action_p95_s=float(np.quantile(total, 0.95)),
        planning_calls=planning_calls,
        planning_calls_per_action=planning_calls / n,
        planning_calls_per_simulated_second=planning_calls / (n * native_dt),
        vfe_update_calls=int(masks["update_mask"].sum()),
        maximum_batch_size=int(batch.max()),
        minimum_batch_size=int(batch.min()),
        scope="single-controller response"
        if (batch == 1).all()
        else "batch response; not B=1 latency",
    )
