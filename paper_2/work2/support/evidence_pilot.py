"""Earlier CPU evidence-module comparison: 256 CAL and 128 DEV, four denoising probes versus 16 samples."""
import argparse
from pathlib import Path
from work2.io import save, read, identity, bind_checkpoint
from .inputs import references


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='Earlier original 100000-update shared model; not final controller weights')
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regenerated-checkpoints',action='store_true',help='Explicitly label new fits instead of requiring exact archived weights')
    a = p.parse_args()
    checkpoint_identity=bind_checkpoint(a.checkpoint,'earlier',a.regenerated_checkpoints)
    refs = {split: references('pilot', split, a.data_root) for split in ('calibration','development')}
    a.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from work2.native.common import seed_all
    from work2.native.train import load_model
    from work2.native.cpu_denoising_evidence import CPUQueryEngine,fit_temperature,fit_gaussian_calibration,categorical_update,entropy
    from work2.native.cpu_denoising_data import _cache_group, _evidence
    from work2.native.belief_diagnostics import belief_metrics
    seed_all(13, threads=4)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model, state = load_model(a.checkpoint, device='cpu', ema=True)
    if state['updates'] != 100000 or state['arm'] != 'R' or state['seed'] != 13:
        raise ValueError('This published pilot used the original 100000-update R checkpoint')
    if state['normalization'] != read(a.data_root / 'normalization.json'):
        raise ValueError('Original TRAIN normalization differs')
    save(a.output/'inputs.json', dict(checkpoint=checkpoint_identity, references=refs, threads=4,
         scope='Earlier CPU evidence module only; not final-controller latency'))
    engine = CPUQueryEngine(model, batch_size=32)
    try:
        cal, cal_time = _cache_group(a.output, 'calibration', refs['calibration'], engine)
        labels = torch.tensor(cal['denoising']['labels'], dtype=torch.int64)
        calibration = dict(denoising=fit_temperature(torch.tensor(cal['denoising']['errors']), labels),
             gaussian=fit_gaussian_calibration({k:torch.tensor(cal['gaussian'][k]) for k in ('mean','variance')},torch.tensor(cal['gaussian']['observed']),labels))
        save(a.output/'calibration.json', calibration)
        dev, dev_time = _cache_group(a.output, 'development', refs['development'], engine)
        likelihood, scoring_time = _evidence(dev, calibration)
        uniform = torch.full((128,8), .125)
        metrics, arrays = {}, {}
        for method in ('denoising','gaussian'):
            posterior = categorical_update(uniform, likelihood[method])
            metrics[method] = belief_metrics(posterior.numpy(), dev[method]['labels'])
            metrics[method]['mean_entropy'] = float(entropy(posterior).mean())
            dev_time[method]['evidence_transform_seconds'] = scoring_time[method]
            dev_time[method]['complete_seconds'] = dev_time[method]['seconds'] + scoring_time[method]
            arrays[method+'_posterior'] = posterior.numpy()
        np.savez_compressed(a.output/'development.npz', **arrays, labels=dev['denoising']['labels'])
        save(a.output/'result.json', dict(state='completed', development=metrics, timing=dict(calibration=cal_time,development=dev_time),
             calibration=calibration, physical_transitions=0, optimizer_updates=0,
             timing_scope='Measured on the current CPU; excludes collection and model training'))
    finally: engine.close()

if __name__ == '__main__': main()
