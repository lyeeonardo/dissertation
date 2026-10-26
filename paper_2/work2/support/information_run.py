"""Execute the two retained historical information-gain pairs with their fixed earlier model and protocols."""
import argparse
from pathlib import Path
from work2.io import read,save,identity,bind_checkpoint

CONFIG=Path(__file__).resolve().parents[1]/'configs'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=('earlier','revised'),required=True)
    p.add_argument('--checkpoint',type=Path,help='Earlier original shared_R step0100000; different from final checkpoint')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--list',action='store_true')
    p.add_argument('--regenerated-checkpoints',action='store_true',help='Explicitly label new fits instead of requiring exact archived weights')
    a=p.parse_args(); design=read(CONFIG/'information_protocols.json')[a.study]
    if a.list: print(__import__('json').dumps(design,indent=2));return
    if a.checkpoint is None or a.output is None: p.error('--checkpoint and --output required for execution')
    checkpoint_identity=bind_checkpoint(a.checkpoint,'earlier',a.regenerated_checkpoints)
    import numpy as np
    import torch
    from work2.native.common import seed_all
    from work2.native.train import load_model
    from work2.native.control import ControlConfig,BeliefController
    from work2.native.evaluation import decision_seed
    from work2.native.policy_preservation_control import HiddenReferencePolicy
    from work2.native.reference_policy_control import ReferenceSelectionConfig
    from work2.native.robust_observation_law import GaussianResidual
    from work2.native.policy_preservation_metrics import episode_metrics as revised_metrics
    from . import information_earlier_core,information_revised_core
    seed_all(13,threads=4)
    model,state=load_model(a.checkpoint,device=a.device,ema=True)
    if state['updates']!=100000 or state['arm']!='R' or state['seed']!=13:
        raise ValueError('Both included earlier comparisons used the original seed13 shared_R100000 EMA')
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'design.json',dict(study=a.study,**design,checkpoint=checkpoint_identity,device=a.device,
         scope='Earlier information-seeking configurations, not the final Chapter4 controller; batch8 throughput'))

    class EarlierPolicy:
        def __init__(self,method,recipe,case):
            self.method,self.model,self.seed=method,model,case['sampling_seed']
            self.config=ControlConfig(**recipe['config']);self.device=next(model.parameters()).device
            self.history_length=8
            self.controller=BeliefController(model,self.config,mode='vfe_full',
                extra_variance=recipe['calibration']['extra_variance'],seed=self.seed)
        @torch.no_grad()
        def act(self,history,step,oracle_condition=None):
            if oracle_condition is not None: raise ValueError('Hidden truth cannot enter either paired controller')
            self.controller.generator.manual_seed(decision_seed(self.seed,step))
            return self.controller.act(history)
        def observe(self,action,observation,history):
            return self.controller.observe(action,observation,history)

    rows=[]; changed=decisions=0
    for method,recipe in design['recipes'].items():
        directory=a.output/method;directory.mkdir()
        for start in range(0,len(design['cases']),8):
            cases=design['cases'][start:start+8]
            if a.study=='earlier': policies=[EarlierPolicy(method,recipe,c) for c in cases]
            else:
                selected=recipe['selection']
                policies=[HiddenReferencePolicy(method,model,ControlConfig(**recipe['config']),c['sampling_seed'],
                    recipe['calibration']['extra_variance'],selection=ReferenceSelectionConfig('soft',selected['cost_scale'],selected['temperature_multiplier']),
                    observation_law=GaussianResidual.from_spec(recipe['law_spec']),split='test') for c in cases]
            def complete(case,arrays,metrics=None):
                nonlocal changed,decisions
                stem=directory/f"case_{case['case_id']:04d}"
                np.savez_compressed(stem.with_suffix('.npz'),**arrays)
                if metrics is None: metrics=revised_metrics(case,arrays,design['reference_speeds'])
                event=float(np.mean([np.asarray(arrays['rewards'])[e['step']:e['step']+150].sum() for e in case['events']])) if case['events'] else None
                row=dict(method=method,**{k:case[k] for k in ('case_id','initial_condition','family')},
                     **{'return':float(np.asarray(arrays['rewards']).sum()),'event_reward_150':event},
                     metrics=metrics,trace=str(stem.with_suffix('.npz')))
                if a.study=='earlier' and method=='efe_full':
                    changed+=int((arrays['selected']!=arrays['costs'].argmin(-1)).sum());decisions+=len(arrays['rewards'])
                rows.append(row);save(stem.with_suffix('.json'),row);save(a.output/'case_metrics.json',rows)
            try:
                if a.study=='earlier': information_earlier_core.run_episodes(policies,cases,reference_speeds=design['reference_speeds'],completed_callback=complete)
                else: information_revised_core.run_episodes(policies,cases,recipe,method,completed_callback=complete)
            except BaseException as error:
                for cid,arrays in getattr(error,'paper2_partial_traces',{}).items():
                    np.savez_compressed(directory/f'failed_case_{cid:04d}.npz',**arrays)
                save(a.output/'failure.json',dict(error=repr(error),automatically_retried=False));raise
        print(f'Completed {method}: {len(design["cases"])} cases',flush=True)
    from .information import revised_contrasts
    from work2.native.analysis import paired_case_contrast
    contrasts=([paired_case_contrast(rows,'efe_full','vfe_full','event_reward_150')] if a.study=='earlier' else revised_contrasts(rows))
    save(a.output/'result.json',dict(state='completed',contrasts=contrasts,cases=rows,
        choice_count=dict(changed=changed,decisions=decisions,scope='Own realized pools, not a common-state causal comparison') if a.study=='earlier' else None))

if __name__ == '__main__': main()
