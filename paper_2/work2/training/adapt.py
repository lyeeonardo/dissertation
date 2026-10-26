"""Bounded native-objective adaptation. Default CLI is metadata only.

DEV selection averages baseline-normalized native components, using the fixed
additive ratio (loss+1e-8)/(baseline+1e-8) to retain valid zero losses. For each component,
weights are .5 across supported target behavior groups, .25 old1, .25 new3.
Shared/History components are action/state/reward/termination; DADP components
are forward+inverse and policy. Random target actions are never demonstrations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from work2.io import read, save, identity, verify
from work2.paths import runtime_manifest
from work2 import physical_model as physical
HERE = None
INPUT_FILE = None
POOL_FILE = None
DESIGN_FILE = Path(__file__).resolve().parents[1] / "configs/adaptation.json"
from .adaptation_sampler import Windows, arm_rows, GROUPS


def require(ok,message):
    if not ok: raise ValueError(message)


def verify_any(records):
    for r in records:
        actual=identity(r['path'])
        require(actual['sha256']==r['sha256'] and actual['size_bytes']==r.get('size_bytes',r.get('bytes')),f'Changed source {r["path"]}')


def design(): return read(DESIGN_FILE)


def load_base(plan,method):
    import torch
    pass # All model source is bundled in this package.
    record=plan['models'][method];verify_any([record['identity']]);path=Path(record['checkpoint'])
    if method=='dadp':
        from work2.native.dadp_training import load_policy
        policy,state=load_policy(path,device='cuda')
        rep,planner=policy.representation,policy.planner
        # Start exactly at the deployed EMA, not the older unaveraged training fit.
        planner.model.load_state_dict(state['ema']);planner.model_ema.load_state_dict(state['ema'])
        planner.ema_rate=design()['ema_rate']
        for k,v in planner.model.state_dict().items(): require(torch.equal(v,planner.model_ema.state_dict()[k]),'DADP zero clone mismatch')
        for k,v in rep.state_dict().items(): require(torch.equal(v,state['representation'][k]),'DADP zero representation mismatch')
        rep.requires_grad_(True)
        return dict(rep=rep,planner=planner),state
    from work2.native.train import load_model
    native,state=load_model(path,device='cuda',ema=True)
    require(native.config.condition_input==(method=='shared'),'Native condition-input contract differs')
    if method=='shared':
        pass # physical_model is a local archive module.
        model=physical.PhysicalSharedDiffusion.from_native(native)
        for k,v in native.state_dict().items():require(torch.equal(v,model.state_dict()[k]),'Physical adapter weights changed')
    else:model=native
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    for k,v in model.state_dict().items():require(torch.equal(v,ema.state_dict()[k]),'Zero EMA clone differs')
    return dict(model=model,ema=ema),state




class Queries:
    def __init__(self,method,loaded):
        self.counts=Counter();self.stage='development';self.handles=[]
        if method=='dadp':
            modules={'policy_online':loaded['planner'].model['diffusion'],'policy_ema':loaded['planner'].model_ema['diffusion']}
            d=loaded['rep'].dynamics
            modules.update({name:getattr(d,name) for name in ('seq_encoder','inverse_head','forward_head','reward_head','embedding_forward_head','policy_head','state_head','factor_head')})
        else: modules={'diffusion_online':loaded['model'],'diffusion_ema':loaded['ema']}
        for name,network in modules.items():
            def count(_module,args,name=name):
                self.counts[f'{self.stage}:{name}:calls']+=1;self.counts[f'{self.stage}:{name}:examples']+=len(args[0])
            self.handles.append(network.register_forward_pre_hook(count))
    def result(self):
        return dict(forward_calls=sum(v for k,v in self.counts.items() if k.endswith(':calls')),
                    forward_examples=sum(v for k,v in self.counts.items() if k.endswith(':examples')),
                    by_stage_module=dict(self.counts),units='Actual denoiser forwards; DADP additionally disjoint sequence-encoder and dynamics-head forwards')


def make_banks(sampler,method,bounds):
    # Local RNG plus scoped fixed Torch corruption below makes banks identical across arms.
    sampler.rng=__import__('numpy').random.default_rng(617410013)
    banks={};support={}
    for group in GROUPS:
        if method=='dadp':
            banks[group]={'representation':sampler.representation(256,group=group)}
            supported=bool(sampler.count(group,'policy'));support[group]={'forward_inverse':True,'policy':supported}
            if supported:banks[group]['policy']=sampler.policy(256,group=group,value_bounds=bounds)
        else:
            banks[group]=sampler.diffusion(256,group=group,augment=False)
            support[group]={k:True for k in ('action','state','reward','termination')}
            support[group]['action']=bool(sampler.count(group,'proposal'))
    return banks,support


def evaluate(method,loaded,banks,support):
    import torch
    from work2.native.windows import to_device
    values={}
    if method=='dadp':
        from work2.native.dadp_training import stage_loss
        rep,planner=loaded['rep'],loaded['planner'];rep.eval();planner.eval();online=planner.model
        planner.model=planner.model_ema
        try:
            for i,(group,bank) in enumerate(banks.items()):
                with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(617420013+i)
                    _,parts=stage_loss('representation',rep,rep,to_device(bank['representation']))
                    row={'forward_inverse':float(parts['forward']+parts['inverse']),'policy':None}
                    if support[group]['policy']:
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            loss,_=stage_loss('policy',planner,rep,to_device(bank['policy']))
                        row['policy']=float(loss)
                    values[group]=row
        finally:planner.model=online
    else:
        loaded['ema'].eval()
        for i,(group,bank) in enumerate(banks.items()):
            generator=torch.Generator(device='cuda').manual_seed(617420013+i)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                _,parts=loaded['ema'].loss(to_device(bank),generator)
            values[group]={k:float(v) if support[group][k] else None for k,v in parts.items()}
    require(all(x is None or math.isfinite(x) and x>=0 for row in values.values() for x in row.values()),'Invalid DEV component')
    return values


def selection_score(values,baseline):
    components=list(baseline['old1']);scores={};weights={}
    for component in components:
        targets=[g for g in ('expert','random','weak') if baseline[g][component] is not None]
        require(targets,'No target support for monitored component')
        weights[component]={'old1':.25,'new3':.25,**{g:.5/len(targets) for g in targets}}
        ratios={}
        for group,w in weights[component].items():
            base=baseline[group][component];value=values[group][component]
            require(base is not None and math.isfinite(base) and base>=0 and value is not None and math.isfinite(value) and value>=0,'Invalid baseline normalization')
            ratios[group]=(value+1e-8)/(base+1e-8)
        scores[component]=sum(weights[component][g]*ratios[g] for g in ratios)
    return sum(scores.values())/len(scores),dict(component_scores=scores,group_weights=weights,
        epsilon=1e-8,formula='mean_components[.5*mean_supported_target_kinds((loss+epsilon)/(update0_loss+epsilon))+.25*old1_ratio+.25*new3_ratio]',
        zero_baseline_note='A zero baseline component is valid; positive increases are measured relative to fixed epsilon. Epsilon was declared before any optimization.')


def payload(method,loaded,base,normalization,counts,plan,arm,history):
    common=dict(arm='R',seed=13,method=method,updates=counts['total'],normalization=normalization,
                base_checkpoint=plan['models'][method]['identity'],base_updates=base['updates'],
                optimizer_updates=dict(counts),development=history,adaptation_arm=arm,
                physical_transitions=0,ema_rate=.99)
    if method=='dadp':
        return dict(common,stage='policy',model=loaded['planner'].model.state_dict(),ema=loaded['planner'].model_ema.state_dict(),
                    representation=loaded['rep'].state_dict(),value_bounds=base['value_bounds'])
    return dict(common,model_config=asdict(loaded['model'].config),model=loaded['model'].state_dict(),ema=loaded['ema'].state_dict())


def execute(method, arm_id, inputs, pools, output):
    global HERE, INPUT_FILE, POOL_FILE
    HERE = Path(output).resolve(); INPUT_FILE = Path(inputs).resolve(); POOL_FILE = Path(pools).resolve()
    require(method in ("shared", "history", "dadp") and arm_id in ("replay_only", "expert_1000"), "Only dissertation methods and arms are included")
    plan=runtime_manifest(INPUT_FILE);config=design();arm=next(x for x in config['arms'] if x['id']==arm_id)
    out=HERE/'fits'/method/arm_id;out.mkdir(parents=True,exist_ok=True)
    # Exclusive start is durable even for a technical failure; no retries/resumes.
    fd=os.open(out/'started.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid(),time=time.time(),method=method,arm=arm_id),stream)
    counts={'total':0,'representation':0,'policy':0,'diffusion':0};started=time.monotonic();queries=None;selected=None
    optimization_times={'representation':0.,'policy':0.,'diffusion':0.}
    try:
        import numpy as np
        import torch
        pass # All model source is bundled in this package.
        from work2.native.common import seed_all,atomic_torch_save
        from work2.native.windows import to_device
        from work2.native.train import train_step,update_ema
        seed_all(config['seed']);pools=runtime_manifest(POOL_FILE);require(pools['state']=='completed','Collection incomplete')
        records=pools['files']+[pools['normalization']]
        verify_any(records);normalization=read(pools['normalization']['path'])
        train=Windows(arm_rows(pools,arm,'train'),normalization,13)
        dev=Windows(arm_rows(pools,arm,'development'),normalization,617410013)
        def ancestry(sampler):return {(e['row']['path'],str(e['row']['parent_id'])) for e in sampler.episodes}
        require(not ancestry(train)&ancestry(dev),'TRAIN/DEV parent overlap')
        tr_roots={e['row'].get('root_id') for e in train.episodes if e['group'] in ('expert','random','weak')}
        dv_roots={e['row'].get('root_id') for e in dev.episodes if e['group'] in ('expert','random','weak')}
        require(None not in tr_roots|dv_roots and not tr_roots&dv_roots,'Target reset ancestry leakage')
        loaded,base=load_base(plan,method);require(base['normalization']==normalization,'Normalization changed')
        bounds=base.get('value_bounds')
        if bounds is not None:train.value_bounds=list(bounds);dev.value_bounds=list(bounds)
        banks,support=make_banks(dev,method,bounds)
        atomic_torch_save(out/'development_banks.pt',dict(banks=banks,support=support,sampler=dev.metadata()))
        frozen=dict(method=method,arm=arm,design=config,pools=identity(POOL_FILE),inputs=identity(INPUT_FILE),
                    sources=records,base=plan['models'][method]['identity'],normalization=pools['normalization'],
                    train=train.metadata(),dev=dev.metadata(),banks=identity(out/'development_banks.pt'),
                    value_bounds=bounds,selection='baseline-normalized component ratios, target/source equal, strict minimum0/100/250',
                    target_policy_fallback='Unsupported target proposal/policy draws use half old1 and half new3 expert replay',
                    ema_initialization='online and EMA both cloned from deployed baseline EMA; rate .99',
                    source_only_control=arm_id=='replay_only')
        save(out/'inputs.json',frozen)
        queries=Queries(method,loaded);history=[];baseline=None;best=math.inf
        if method=='dadp':
            from work2.native.dadp_training import optimization_step
            rep=loaded['rep'];planner=loaded['planner']
            rep_optimizer=torch.optim.Adam(rep.parameters(),lr=config['learning_rate'])
            policy_optimizer=torch.optim.AdamW(planner.model.parameters(),lr=config['learning_rate'],weight_decay=1e-5)
        else:optimizer=torch.optim.AdamW(loaded['model'].parameters(),lr=config['learning_rate'],weight_decay=.01,fused=True)
        def checkpoint():
            nonlocal baseline,best,selected
            queries.stage='development';values=evaluate(method,loaded,banks,support)
            if baseline is None:baseline=copy.deepcopy(values)
            # Retain every observed component before numerical selection can fail.
            save(out/'development_observations'/f'step_{counts["total"]:07d}.json',
                 dict(updates=counts['total'],optimizer_updates=dict(counts),values=values,baseline=baseline,
                      support=support,selection_epsilon=1e-8,model_queries=queries.result()))
            score,explanation=selection_score(values,baseline);require(math.isfinite(score),'Nonfinite selection')
            row=dict(updates=counts['total'],optimizer_updates=dict(counts),components=values,score=score,selection=explanation,is_best=score<best)
            history.append(row)
            state=payload(method,loaded,base,normalization,counts,plan,arm_id,history)
            state['optimizer']={'representation':rep_optimizer.state_dict(),'policy':policy_optimizer.state_dict()} if method=='dadp' else optimizer.state_dict()
            path=out/'checkpoints'/f'step_{counts["total"]:07d}.pt';atomic_torch_save(path,state)
            if score<best:
                best=score; temp=out/'best.pt.tmp';os.link(path,temp);temp.replace(out/'best.pt')
                selected=dict(state='selected',checkpoint=str(out/'best.pt'),checkpoint_identity=identity(out/'best.pt'),updates=counts['total'],
                              optimizer_updates=dict(counts),selected_score=score,selected_metric='fixed_dev_baseline_normalized_component_mean',
                              base_checkpoint=plan['models'][method]['identity'],base_updates=base['updates'],ema=True,method=method,arm=arm_id)
                save(out/'selected.json',selected)
            save(out/'development.json',history)
            save(out/'progress.json',dict(state='running',method=method,arm=arm_id,updates=counts['total'],optimizer_updates=counts,selection=selected,training_exposure=train.metadata(),model_queries=queries.result()))
        checkpoint();target=arm_id!='replay_only'
        if method=='dadp':
            for block in config['dadp_schedule']:
                for stage,number in [('representation',block['representation_updates']),('policy',block['policy_updates'])]:
                    rep.requires_grad_(stage=='representation');rep.train(stage=='representation')
                    if stage=='policy':planner.train()
                    torch.cuda.synchronize();optimization_started=time.monotonic()
                    for _ in range(number):
                        queries.stage='training'
                        batch=train.representation(256,target=target) if stage=='representation' else train.policy(256,target=target,value_bounds=bounds)
                        model=rep if stage=='representation' else planner;opt=rep_optimizer if stage=='representation' else policy_optimizer
                        optimization_step(stage,model,rep,to_device(batch),opt)
                        counts['total']+=1;counts[stage]+=1
                    torch.cuda.synchronize();optimization_times[stage]+=time.monotonic()-optimization_started
                require(counts['total']==block['checkpoint'],'DADP schedule mismatch');checkpoint()
        else:
            torch.cuda.synchronize();optimization_started=time.monotonic()
            for step in range(1,config['max_optimizer_updates']+1):
                queries.stage='training';batch=train.diffusion(256,target=target)
                metrics=train_step(loaded['model'],optimizer,to_device(batch),256,ema=None)
                update_ema(loaded['ema'],loaded['model'],rate=config['ema_rate'])
                counts['total']+=1;counts['diffusion']+=1
                require(all(torch.isfinite(v).all() for v in metrics.values()),'Nonfinite native loss')
                if step in config['checkpoints']:
                    torch.cuda.synchronize();optimization_times['diffusion']+=time.monotonic()-optimization_started
                    checkpoint()
                    torch.cuda.synchronize();optimization_started=time.monotonic()
        require(counts['total']==250,'Wrong total update budget')
        verify_any(records);verify(plan.get('source_records', []));require(identity(POOL_FILE)==frozen['pools'],'Pool mutation')
        selected['state']='completed';save(out/'selected.json',selected)
        final=out/'checkpoints/step_0000250.pt';os.link(final,out/'final.pt')
        result=dict(state='completed',method=method,arm=arm_id,updates=counts['total'],optimizer_updates=counts,
                    selected_update=selected['updates'],selected_checkpoint=selected['checkpoint'],selected_checkpoint_identity=selected['checkpoint_identity'],
                    selected_score=selected['selected_score'],selected_metric=selected['selected_metric'],base_checkpoint=plan['models'][method]['identity'],
                    inputs=identity(out/'inputs.json'),selected=identity(out/'selected.json'),development=identity(out/'development.json'),
                    training_exposure=train.metadata(),dev_support=support,model_queries=queries.result(),physical_transitions=0,
                    optimization_wall_seconds=sum(optimization_times.values()),optimization_seconds_by_stage=optimization_times,
                    elapsed_seconds=time.monotonic()-started,stop_reason='declared_250_update_budget_not_convergence',automatic_retry=False)
        save(out/'result.json',result);save(out/'progress.json',result);return result
    except BaseException as error:
        save(out/'result.json',dict(state='failed',method=method,arm=arm_id,updates=counts['total'],optimizer_updates=counts,error=repr(error),
                                    traceback=traceback.format_exc(),model_queries=queries.result() if queries else {'forward_calls':0,'forward_examples':0},
                                    physical_transitions=0,elapsed_seconds=time.monotonic()-started,optimization_wall_seconds=sum(optimization_times.values()),
                                    optimization_seconds_by_stage=optimization_times,timing_note='Only completed optimization intervals counted on failure',selected=selected,automatic_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser(description="Final dissertation 250-update expert/replay adaptation")
    parser.add_argument("--method", choices=("shared", "history", "dadp"), required=True)
    parser.add_argument("--arm", choices=("replay_only", "expert_1000"), required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.method, args.arm, args.inputs, args.pools, args.output)

if __name__ == "__main__": main()
