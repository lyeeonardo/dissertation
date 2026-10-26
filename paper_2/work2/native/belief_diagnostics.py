from __future__ import annotations


import numpy as np

def factor_probabilities(probabilities):
    probabilities = np.asarray(probabilities, np.float64)
    states = np.arange(8)
    return {str(bit): probabilities[:, (states & bit) != 0].sum(-1).clip(0, 1) for bit in (4, 2, 1)}

def belief_metrics(probabilities, labels, events=(), window=150, stationary_burnin=150):
    probabilities, labels = np.asarray(probabilities, np.float64), np.asarray(labels, np.int64)
    if probabilities.shape != (len(labels), 8) or not len(labels):
        raise ValueError("Each observed transition requires eight posterior probabilities")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(-1), 1, atol=1e-5, rtol=0)
        or np.any((labels < 0) | (labels > 7))
    ):
        raise ValueError("Belief diagnostics require normalized probabilities and valid labels")
    prediction = probabilities.argmax(-1)
    true_probability = probabilities[np.arange(len(labels)), labels].clip(1e-12, 1)
    changes = prediction[1:] != prediction[:-1]
    unchanged = labels[1:] == labels[:-1]
    stationary = not events and bool(np.all(labels == labels[0]))
    after_burnin = np.arange(1, len(labels)) >= stationary_burnin
    event_metrics = []
    for event in events:
        start = event["step"]
        correct = prediction[start : start + window] == labels[start : start + window]
        indices = np.flatnonzero(correct)
        held = [i for i in range(max(0, len(correct) - 19)) if correct[i : i + 20].all()]
        first = int(indices[0] + 1) if len(indices) else None
        full_window = len(correct) == window
        observed_sustained = bool(correct[held[0] :].all()) if held else False
        event_metrics.append(
            {
                "step": start,
                "observed_transitions": len(correct),
                "complete_observation_window": full_window,
                "first_correct_delay": first,
                "capped_first_correct_delay": first if first is not None else window,
                "sustained_20_onset_delay": held[0] + 1 if held else None,
                "sustained_20_confirmation_delay": held[0] + 20 if held else None,
                "correct_fraction": float(correct.mean()) if len(correct) else None,
                "correct_through_observed_remainder": observed_sustained,
                "correct_through_remainder": full_window and observed_sustained,
            }
        )
    factor_accuracy, factor_nll, factor_brier = {}, {}, {}
    for bit, marginal in factor_probabilities(probabilities).items():
        truth = ((labels & int(bit)) != 0).astype(float)
        factor_accuracy[bit] = float(((marginal > 0.5) == truth).mean())
        p_true = np.where(truth, marginal, 1 - marginal).clip(1e-12, 1)
        factor_nll[bit] = float(-np.log(p_true).mean())
        factor_brier[bit] = float(np.square(marginal - truth).mean())
    confusion = np.zeros((8, 8), np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    return {
        "nll": float(-np.log(true_probability).mean()),
        "brier": float(np.square(probabilities - np.eye(8)[labels]).sum(-1).mean()),
        "accuracy": float((prediction == labels).mean()),
        "factor_accuracy": factor_accuracy,
        "factor_nll": factor_nll,
        "factor_brier": factor_brier,
        "factor_scope": "Binary marginals; factor Brier is scalar (p(bit=1)-truth)^2",
        "joint_map_factor_accuracy": {
            str(bit): float(((prediction & bit) == (labels & bit)).mean()) for bit in (4, 2, 1)
        },
        "joint_confusion_counts": confusion.tolist(),
        "confidently_wrong_fraction": float(
            ((prediction != labels) & (probabilities.max(-1) >= 0.9)).mean()
        ),
        "map_changes_per_1000_unchanged_label_transitions": float(1000 * changes[unchanged].mean())
        if unchanged.any()
        else 0.0,
        "stationary_map_changes_per_1000": float(1000 * changes[after_burnin].mean())
        if stationary and after_burnin.any()
        else None,
        "stationary_burnin_transitions": stationary_burnin,
        "stationary_scored_transitions": int(after_burnin.sum()) if stationary else 0,
        "stationary_change_scope": (
            "Instability after burn-in; continued identification can contribute"
        ),
        "quality_scope": "Observed transitions only; no fabricated posterior after native death",
        "events": event_metrics,
    }
