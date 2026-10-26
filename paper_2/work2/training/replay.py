"""The selected 5,000-update shared replay fit, with explicit input/output paths."""
from pathlib import Path
from dataclasses import asdict
import argparse
import copy
import json
import torch
from work2.io import save, identity
from .replay_sampler import RestrictedWindowSampler
from .replay_core import CONFIG, merge_samplers, development_bank, evaluate_diagnostics
from work2.native.common import seed_all, require_cuda, atomic_torch_save, rng_state
from work2.native.train import load_model, train_step, update_ema
from work2.native.windows import to_device


def execute(base_checkpoint, original_data, refreshed_data, output, regenerated=False):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    seed_all(CONFIG['seed']); require_cuda()
    def sampler(split, seed):
        left = RestrictedWindowSampler(Path(original_data), range(4), split, seed)
        right = RestrictedWindowSampler(Path(refreshed_data), range(4, 8), split, seed)
        return merge_samplers(left, right, seed)
    train = sampler('train', 13)
    dev = sampler('development', CONFIG['dev_seed'])
    if len(train.anchors) != 537600 or len(dev.anchors) != 76800:
        raise ValueError('The original 537600 TRAIN / 76800 DEV replay inventory is required')
    if {str(r['parent_id']) for r in train.rows} & {str(r['parent_id']) for r in dev.rows}:
        raise ValueError('TRAIN/DEV parent overlap')
    banks = development_bank(dev)
    atomic_torch_save(output / 'development_banks.pt', banks)
    model, base = load_model(Path(base_checkpoint), device='cuda', ema=True)
    if base['seed'] != 13 or base['arm'] != 'R' or not model.config.condition_input:
        raise ValueError('A seed-13 R conditional EMA is required')
    if not regenerated and base['updates'] != 62000:
        raise ValueError('The selected original 62000-update EMA is required; use --regenerated-checkpoints for a new selected fit')
    if base['normalization'] != train.normalization or train.normalization != dev.normalization:
        raise ValueError('The original normalization must agree across both datasets and the checkpoint')
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01, fused=True)
    diagnostics = []
    base_id = identity(base_checkpoint)
    save(output / 'inputs.json', dict(config=CONFIG, base_checkpoint=base_id, regenerated_checkpoints=regenerated,
         original_data=str(Path(original_data).resolve()), refreshed_data=str(Path(refreshed_data).resolve())))
    def checkpoint(update):
        values = evaluate_diagnostics(ema, banks)
        diagnostics.append(dict(updates=update, **values))
        atomic_torch_save(output / 'checkpoints' / f'step_{update:07d}.pt', dict(
            model_config=asdict(model.config), normalization=base['normalization'], model=model.state_dict(),
            ema=ema.state_dict(), optimizer=optimizer.state_dict(), seed=13, arm='R', method='shared',
            updates=update, base_updates=base['updates'], base_checkpoint=base_id, regenerated_checkpoints=regenerated,
            rng=rng_state(), sampler_rng=copy.deepcopy(train.rng.bit_generator.state),
            diagnostic_development=copy.deepcopy(diagnostics), ema_rate=.99, physical_transitions=0))
        save(output / 'development.json', diagnostics)
    checkpoint(0)
    for update in range(1, 5001):
        batch = train.sample(256, augment=True)
        metrics = train_step(model, optimizer, to_device(batch), 256, ema=None)
        update_ema(ema, model, rate=.99)
        if not all(torch.isfinite(v).all() for v in metrics.values()):
            raise FloatingPointError('Nonfinite native loss')
        if update in CONFIG['checkpoints']: checkpoint(update)
    save(output / 'result.json', dict(state='completed', updates=5000, selected_dissertation_checkpoint='checkpoints/step_0005000.pt',
         selection='Fixed 5000-update choice; DEV losses remain diagnostic, no new checkpoint search',
         regenerated_checkpoints=regenerated, physical_transitions=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-checkpoint', type=Path, required=True)
    p.add_argument('--original-data', type=Path, required=True)
    p.add_argument('--refreshed-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regenerated-checkpoints', action='store_true', help='Use the selected newly trained base fit; retain the fixed 5000 replay updates')
    a = p.parse_args(); execute(a.base_checkpoint, a.original_data, a.refreshed_data, a.output, a.regenerated_checkpoints)

if __name__ == '__main__': main()
