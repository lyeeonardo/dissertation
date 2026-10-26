from __future__ import annotations


import numpy as np

def paired_case_contrast(rows, first, second, metric, draws=5000, seed=13):
    """Resample full matched cases within initial-condition/family strata."""
    by_method = {
        method: {row["case_id"]: row for row in rows if row["method"] == method}
        for method in (first, second)
    }
    if set(by_method[first]) != set(by_method[second]):
        raise ValueError(
            "Paired inference needs equal complete case sets; do not silently intersect"
        )
    groups, case_count = {}, 0
    for case_id, a in by_method[first].items():
        b = by_method[second][case_id]
        if (a["initial_condition"], a["family"]) != (b["initial_condition"], b["family"]):
            raise ValueError("Paired case definitions differ")
        if a[metric] is None and b[metric] is None:
            continue
        if a[metric] is None or b[metric] is None:
            raise ValueError("Metric availability differs within a paired case")
        key = (a["initial_condition"], a["family"])
        groups.setdefault(key, []).append(float(a[metric]) - float(b[metric]))
        case_count += 1
    if not groups:
        raise ValueError(f"No paired cases support {metric}")
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(draws)
    estimates = []
    for key in sorted(groups):
        differences = np.asarray(groups[key])
        estimates.append(differences.mean())
        bootstrap += differences[
            rng.integers(len(differences), size=(draws, len(differences)))
        ].mean(1) / len(groups)
    return {
        "first": first,
        "second": second,
        "metric": metric,
        "estimate": float(np.mean(estimates)),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "cases": case_count,
        "strata": len(groups),
        "draws": draws,
        "unit": "complete evaluation case, paired within condition/family",
        "scope": "Conditional on the single trained fit and fixed benchmark regimes",
    }
