"""Recompute final checkpoint evidence temperature from its fixed 256 CAL transitions."""
import argparse
from pathlib import Path
from work2.io import save, read, identity, bind_checkpoint
from .inputs import references


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--regenerated-checkpoints',action='store_true',help='Explicitly label new fits instead of requiring exact archived weights')
    a = p.parse_args()
    checkpoint_identity=bind_checkpoint(a.checkpoint,'final',a.regenerated_checkpoints)
    refs = references('final', 'calibration', a.data_root)
    a.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from work2.native.common import seed_all
    from work2.native.train import load_model
    from work2.native.unified_comparison_control import UnifiedComparisonEngine
    from work2.native.cpu_denoising_evidence import fit_temperature
    from .calibration_core import query_calibration
    seed_all(13, threads=4)
    model, state = load_model(a.checkpoint, device=a.device, ema=True)
    if state['normalization'] != read(a.data_root / 'normalization.json'):
        raise ValueError('Checkpoint and original TRAIN normalization differ')
    engine = UnifiedComparisonEngine(model, batch_size=256)
    save(a.output / 'inputs.json', dict(checkpoint=checkpoint_identity, references=refs,
         probes=[.2,.4,.6,.8], seed_base=398000000, seed_stride=10000, device=a.device))
    try:
        def commit(start, arrays, batch):
            np.savez_compressed(a.output / f'batch_{start:04d}.npz', **arrays)
        arrays = query_calibration(engine, refs, commit)
        fitted = fit_temperature(torch.from_numpy(arrays['errors']).float(), torch.from_numpy(arrays['labels']).long(), split='CAL')
        save(a.output / 'result.json', dict(state='completed', calibration=fitted,
             archived_temperature=.007242573030413831, query_counts=engine.stats.snapshot(),
             optimizer_updates=0, physical_transitions=0, dev_test_queries=0,
             comparison_scope='A regenerated fit or different numerical environment can change temperature'))
    finally: engine.close()

if __name__ == '__main__': main()
