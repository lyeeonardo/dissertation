#!/usr/bin/env python3
"""Export numerical plot inputs from completed work2.run directories.

Read actual saved traces, not cached metrics or historical absolute trace paths.
No model, environment, or training code is imported.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def export(runs, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; existing files are preserved")
    cases_file = (
        Path(__file__).resolve().parents[2] / "paper_2/work2/configs/cases.json"
    )
    design_cases = json.loads(cases_file.read_text())
    adaptation, summary, series, seen = [], [], [], set()
    for run in runs:
        run = Path(run).resolve()
        records = json.loads((run / "results.json").read_text())
        design_path = run / "design.json"
        design = json.loads(design_path.read_text()) if design_path.exists() else {}
        if not isinstance(records, list):
            raise ValueError("Expected the results.json list written by work2.run")
        for record in records:
            if record.get("state") != "completed":
                raise ValueError(
                    "Figure exports require completed records; unfinished runs are not discarded"
                )
            study, method, case = record["study"], record["method"], record["case"]
            if study not in ("adaptation", "gravity_sweep"):
                continue
            expected = {row["case_id"]: row for row in design_cases[study]}
            if case != expected.get(case["case_id"]):
                raise ValueError(
                    f"Case differs from the retained study design: {study}/{case['case_id']}"
                )
            arm = record.get("arm", design.get("arm", "zero"))
            key = study, arm, method, case["case_id"]
            if key in seen:
                raise ValueError(f"Repeated study/arm/method/case: {key}")
            seen.add(key)
            trace = run / method / f"case_{case['case_id']:04d}" / "trace.npz"
            with np.load(trace, allow_pickle=False) as arrays:
                rewards = np.asarray(arrays["rewards"], dtype=float)
                n = len(rewards)
                if (
                    rewards.shape != (n,)
                    or n != case["steps"]
                    or not np.isfinite(rewards).all()
                ):
                    raise ValueError(f"Expected a complete finite episode: {trace}")
                gravity = np.asarray(arrays["gravity_scale"], dtype=float)
                if gravity.shape != (n,) or not np.allclose(
                    gravity, gravity[0], rtol=0, atol=1e-12
                ):
                    raise ValueError("These plots require stationary physical gravity")
                total = float(rewards.sum())
                if study == "adaptation":
                    adaptation.append(
                        dict(
                            method=method,
                            arm=arm,
                            case_key=case["case_key"],
                            return_value=total,
                            actual_gravity=float(gravity[0]),
                            bank_high=case.get("high_gravity", 3),
                            native_steps=n,
                            assigned_steps=case["steps"],
                            reset_seed=case["reset_seed"],
                            sampling_seed=case["sampling_seed"],
                        )
                    )
                else:
                    if method != "genaif" or arm != "zero":
                        raise ValueError(
                            "The fixed-bank gravity sweep uses the unadapted GenAIF controller"
                        )
                    if not np.isclose(
                        gravity[0], case["actual_gravity"], rtol=0, atol=1e-12
                    ):
                        raise ValueError(
                            "Observed gravity differs from the assigned sweep level"
                        )
                    summary.append(
                        dict(actual_gravity=float(gravity[0]), return_value=total)
                    )
                    if float(gravity[0]) not in (2.0, 5.0):
                        continue
                    beliefs = np.asarray(arrays["beliefs"], dtype=float)
                    if beliefs.shape != (n + 1, 8):
                        raise ValueError(
                            "Expected the initial belief and one posterior per transition"
                        )
                    friction = np.asarray(arrays["sliding_friction_scale"], dtype=float)
                    joint_range = np.asarray(arrays["joint_range_scale"], dtype=float)
                    if (
                        friction.shape != (n,)
                        or joint_range.shape != (n,)
                        or not np.allclose(friction, 1.0)
                        or not np.allclose(joint_range, 1.0)
                    ):
                        raise ValueError(
                            "The gravity sweep holds friction and joint range at one"
                        )
                    for index in range(n):
                        series.append(
                            dict(
                                actual_gravity=float(gravity[index]),
                                transition_index=index,
                                action_time_s=index * 0.05,
                                outcome_time_s=(index + 1) * 0.05,
                                reward=float(rewards[index]),
                                actual_friction=float(friction[index]),
                                actual_joint_range=float(joint_range[index]),
                                **{
                                    f"posterior_{condition:03b}": float(
                                        beliefs[index + 1, condition]
                                    )
                                    for condition in range(8)
                                },
                            )
                        )
    if not adaptation and not summary:
        raise ValueError("No adaptation or gravity_sweep results were supplied")
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "adaptation.csv": adaptation,
        "gravity_summary.csv": sorted(summary, key=lambda row: row["actual_gravity"]),
        "gravity_series.csv": series,
    }
    for name, rows in tables.items():
        if not rows:
            continue
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return {name: len(rows) for name, rows in tables.items() if rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="work2.run output directory; repeat for the three adaptation arms",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
