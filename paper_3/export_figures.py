"""Export numerical figure inputs from completed Work 3 evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from work3_paths import ROOT

METHODS = {
    'genaif': 'genaif_selective_mean',
    'history': 'public_history_h1',
    'oracle': 'oracle_selected',
    'stock_han': 'stock_han',
    'stock_yang': 'stock_yang',
    'expert_pi': 'expert_pi',
    'public_pi': 'public_pi',
}
CASES = tuple(f'stock_test_p{i:03d}' for i in range(6))


def episode(root, study, case, method):
    folder = root / study / f'{case}__{method}'
    result = json.loads((folder / 'result.json').read_text())
    if (result.get('status') != 'complete' or result.get('implementation_smoke')
            or result.get('completed_decisions') != 1344):
        raise ValueError(f'Complete fourteen-day episode required: {folder}')
    if (result['study'], result['case_id'], result['method']) != (study, case, method):
        raise ValueError(f'Run identity does not match its directory: {folder}')
    with np.load(folder / 'arrays.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    if arrays['costs'].shape != (1344,) or not np.isfinite(arrays['costs']).all():
        raise ValueError(f'Invalid full-episode costs: {folder}')
    np.testing.assert_allclose(arrays['time_end_days'], np.arange(1, 1345) / 96,
                               rtol=0, atol=1e-10)
    return folder, result, arrays


def benchmark(root):
    rows = []
    for case in CASES:
        for method, label in METHODS.items():
            folder, result, arrays = episode(root, 'benchmark', case, method)
            components = arrays['components']
            if components.shape != (1344, 6) or not np.isfinite(components).all():
                raise ValueError(f'Invalid integrated accounting components: {folder}')
            eqi = float(components[:, 3].sum() / 14)
            oci = float(components[:, [0, 1, 2, 4, 5]].sum() / 14)
            cost = float(arrays['costs'].mean())
            np.testing.assert_allclose(cost, eqi + .3 * oci, rtol=1e-10, atol=1e-8)
            rows.append(dict(case_id=case, method=label, cost_per_day=cost,
                             eqi=eqi, oci=oci, weighted_oci=.3 * oci,
                             integrated_cost=float(arrays['costs'].sum() / 96),
                             decision_mean_seconds=float(arrays['decision_seconds'].mean())))
    return rows


def forecast(root):
    pairs, traces = [], []
    for case in CASES:
        _, on_result, on = episode(root, 'forecast-selection', case, 'scoring_on')
        _, off_result, off = episode(root, 'forecast-selection', case, 'scoring_off')
        if on_result['weather'] != off_result['weather']:
            raise ValueError(f'Paired weather mismatch: {case}')
        on_total, off_total = np.cumsum(on['costs']) / 96, np.cumsum(off['costs']) / 96
        if not (off_total > 0).all():
            raise ValueError(f'Positive reference cumulative costs required: {case}')
        effect = 100 * (on_total / off_total - 1)
        pairs.append(dict(case_id=case, weather=on_result['weather'],
                          on_cost=float(on['costs'].mean()), off_cost=float(off['costs'].mean()),
                          on_vs_off_percent=float(100 * (on['costs'].mean() / off['costs'].mean() - 1))))
        for day, on_value, off_value, percent in zip(on['time_end_days'], on_total, off_total, effect):
            traces.append(dict(case_id=case, day=float(day), on_integrated_cost=float(on_value),
                               off_integrated_cost=float(off_value), on_vs_off_cumulative_percent=float(percent)))
    return pairs, traces


def belief(root):
    folder, _, arrays = episode(root, 'development', 'validation_p002', 'genaif')
    with np.load(folder / 'belief_forecast.npz', allow_pickle=False) as saved:
        if 'initial_particles' not in saved:
            raise ValueError('The run lacks the saved initial pre-correction particles; use the current evaluation logger.')
        indices = [59, 21]
        corrected = np.take(saved['particles'][:9], indices, axis=-1)
        selected = np.take(saved['forecast'][:9], indices, axis=-1)
        updated = np.take(saved['updated_particles'][:9], indices, axis=-1)
        initial = np.take(saved['initial_particles'], indices, axis=-1)
        prior = saved['weights'][:9].copy()
        posterior = saved['updated_weights'][:9].copy()
    incoming = np.concatenate((initial[None], updated[:8]), axis=0)
    incoming_weights = np.concatenate((prior[:1], posterior[:8]), axis=0)
    # Hidden coordinates are unchanged by clamping observed public channels.
    particles = np.stack((incoming, corrected, selected, updated, updated), axis=1)
    weights = np.stack((incoming_weights, prior, prior, posterior, posterior), axis=1)
    if particles.shape != (9, 5, 8, 2) or weights.shape != (9, 5, 8):
        raise ValueError('Nine boundaries and eight particles are required for the startup figure')
    np.testing.assert_array_equal(incoming_weights, prior)
    np.testing.assert_allclose(weights.sum(-1), 1, rtol=0, atol=1e-12)
    return dict(particles=particles.tolist(), weights=weights.tolist(),
                truth=np.take(arrays['privileged'][:9], indices, axis=-1).tolist())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('benchmark', 'forecast', 'belief', 'all'))
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory for numerical figure inputs')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a fresh output directory; existing files are preserved')
    exports = {}
    if args.kind in ('benchmark', 'all'):
        exports['benchmark.csv'] = benchmark(args.root)
    if args.kind in ('forecast', 'all'):
        pairs, traces = forecast(args.root)
        exports['forecast/paired.csv'], exports['forecast/cumulative_cost.csv'] = pairs, traces
    if args.kind in ('belief', 'all'):
        exports['belief_startup.json'] = belief(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in exports.items():
        path = args.output / name
        if path.suffix == '.csv':
            write_csv(path, values)
        else:
            with path.open('x') as stream:
                json.dump(values, stream, indent=2, allow_nan=False)
                stream.write('\n')
        print(path)


if __name__ == '__main__':
    main()
