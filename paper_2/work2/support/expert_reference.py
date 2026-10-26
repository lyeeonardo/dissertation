"""Compute stationary expert TRAIN references from a generated source corpus."""
import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from work2.paths import corpus_rows


def summarize(root, refreshed):
    rows = [row for row in corpus_rows(root) if row['condition'] < 4]
    rows += [row for row in corpus_rows(refreshed) if row['condition'] >= 4]
    if (Path(root)/'normalization.json').read_bytes() != (Path(refreshed)/'normalization.json').read_bytes():
        raise ValueError('Base and refreshed source normalization differ')
    selected = [row for row in rows if row['split'] == 'train' and row['kind'] == 'expert']
    result = []
    for condition in range(8):
        group = [row for row in selected if row['condition'] == condition]
        if len(group) != 56 or len({str(row['parent_id']) for row in group}) != 56:
            raise ValueError('Each condition requires 56 distinct expert TRAIN parents')
        returns, steps, eligible = [], 0, 0
        for row in group:
            with h5py.File(row['path'], 'r') as handle:
                episode = handle[row['episode']]
                rewards = np.asarray(episode['rewards'], dtype=np.float64)
                if rewards.shape != (1000,) or not np.isfinite(rewards).all():
                    raise ValueError('Expert TRAIN episodes must contain 1000 finite rewards')
                if (episode.attrs['split'] != 'train' or int(episode.attrs['condition']) != condition
                        or str(episode.attrs['parent_id']) != str(row['parent_id'])):
                    raise ValueError('Corpus index and episode identities differ')
                returns.append(float(rewards.sum()))
                steps += len(rewards)
                eligible += int(np.asarray(episode['proposal_eligible'], dtype=bool).sum())
        result.append(dict(condition=f'{condition:03b}', expert_episodes=len(group), steps=steps,
                           mean_return=float(np.mean(returns)), minimum_return=min(returns),
                           maximum_return=max(returns), mean_step_reward=float(np.sum(returns)/steps),
                           proposal_eligible_steps=eligible))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--refreshed-data', type=Path, required=True,
                        help='Qualified 3x specialist corpus; base data supply only conditions 0-3')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.data_root, args.refreshed_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved eight expert TRAIN references to {args.output}')


if __name__ == '__main__':
    main()
