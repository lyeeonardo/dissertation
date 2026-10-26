"""Train the original seed13 R shared model through100000 updates for the retained earlier evidence/IG studies."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from work2.io import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eager',action='store_true',help='Disable original torch.compile(default); labels the result as a changed numerical execution')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from work2.native.common import seed_all,require_cuda,atomic_torch_save,rng_state
    from work2.native.model import ModelConfig,SharedDiffusion
    from work2.native.windows import WindowSampler,to_device
    from work2.native.train import train_step,learning_rate,development_loss
    seed_all(13);require_cuda()
    train=WindowSampler(a.data_root,'R','train',seed=13)
    dev=WindowSampler(a.data_root,'R','development',seed=130003)
    if len(train.anchors)!=560000: raise ValueError('Original R560000-transition TRAIN corpus required')
    config=ModelConfig(condition_input=True)
    model=SharedDiffusion(config,train.normalization).cuda()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,fused=True)
    if not a.eager: model.forward=torch.compile(model.forward,mode='default',fullgraph=True)
    save(a.output/'recipe.json',dict(model_config=asdict(config),seed=13,arm='R',updates=100000,
        original_schedule_target=300000,effective_batch=1024,microbatch=512,ema=.999,
        forward_compilation='eager' if a.eager else 'default',selection='Fixed earlier dissertation checkpoint, no new selection'))
    development=[]
    for update in range(1,100001):
        for group in optimizer.param_groups: group['lr']=learning_rate(update-1)
        metrics=train_step(model,optimizer,to_device(train.sample(1024)),512,ema)
        if not all(torch.isfinite(v).all() for v in metrics.values()): raise FloatingPointError('Nonfinite native training metric')
        if update%10000==0:
            development.append(dict(updates=update,**development_loss(ema,dev)))
            save(a.output/'development.json',development)
        if update%25000==0:
            atomic_torch_save(a.output/'checkpoints'/f'step_{update:07d}.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                model_config=asdict(config),normalization=train.normalization,arm='R',seed=13,updates=update,
                effective_batch=1024,rng=rng_state(),sampler_rng=train.rng.bit_generator.state,development=development))
    save(a.output/'result.json',dict(state='completed',selected='checkpoints/step_0100000.pt',updates=100000,physical_transitions=0))

if __name__ == '__main__': main()
