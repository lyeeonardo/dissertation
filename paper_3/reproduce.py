"""Reproduce only the experiments retained in Dissertation Chapter 5 and Appendix C."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from work3_paths import ROOT, INPUTS, input_path

STUDIES = ('benchmark','correction','forecast-selection','development','preference','settling','oxygen-transfer','sensor-offset')

def read(path):
    return json.loads(Path(path).read_text())

def cases(cohort):
    return read(ROOT / 'configs' / f'{cohort}_cases.json')

def jobs(study):
    test, dev = cases('test'), cases('development')
    specs = {
        'benchmark': [(c,m) for c in test for m in ('genaif','history','oracle','public_pi','expert_pi','stock_han','stock_yang')],
        'correction': [(c,m) for c in test for m in ('genaif','uncorrected')],
        'forecast-selection': [(c,m) for c in test for m in ('scoring_on','scoring_off')],
        'development': [(c,'genaif') for c in dev],
        'preference': [(c,m) for c in dev for m in ('reward_reference','oxygen_preference')],
        'settling': [(dev[0],m) for m in ('nominal','settling_6h','settling_24h','settling_permanent')],
        'oxygen-transfer': [(c,m) for c in dev for m in ('nominal','transfer_07')],
        'sensor-offset': [(dev[0],m) for m in ('nominal','sensor_plus2')],
    }
    return [dict(study=study,case=c,method=m,job_id=f'{c["case_id"]}__{m}') for c,m in specs[study]]

def configure(device, *, legacy_precision=False):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    if device.startswith('cuda'): torch.cuda.manual_seed_all(17)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=legacy_precision

def make_controller(method,device):
    from bsm1_offline_benchmark.stock_joint_control import JointController
    from bsm1_offline_benchmark.evaluate import PublicPI
    from bsm1_offline_benchmark.evaluate_v2 import PublicExpert
    from research_shared_model import defaults, recommended
    from research_shared_model.efe_exploration.selective_mean_control import SelectiveMeanController
    recipe=defaults.controller_recipe()
    if method=='public_pi':return PublicPI(17)
    if method=='expert_pi':return PublicExpert(read(ROOT/'configs/expert_pi.json'),17)
    if method in ('stock_han','stock_yang'):
        from importlib import import_module
        return import_module('bsm1_offline_benchmark.'+method).load_controller(input_path('models/'+method+'.pt'),device=device)
    if method in ('history','oracle','uncorrected'):
        if method!='uncorrected':recipe.update(mode=method,belief_update=False,refine=False,information_weight=0.)
        return JointController(defaults.CHECKPOINT,device,remaining=1344,**recipe)
    if method in ('scoring_on','scoring_off'):
        from studies.selection import make_controller as make
        return make('on' if method=='scoring_on' else 'off',device)
    if method in ('reward_reference','oxygen_preference'):
        from studies.preference import make_controller as make
        return make(weight=0. if method=='reward_reference' else .01,device=device)
    return recommended.make_controller(device)

class RecordedController:
    """Passive trace capture at the same boundaries used by the dissertation."""
    def __init__(self,base,sensor_offset=False):
        self.base=base;self.sensor_offset=sensor_offset;self.rows=[];self.monitor_rows=[];self.last_diagnostics={}
    def __getattr__(self,name):return getattr(self.base,name)
    @property
    def native(self):return getattr(self.base,'controller',self.base)
    def packet(self,public,boundary):
        values=np.asarray(public).copy()
        if self.sensor_offset and 672<=boundary<768:values[38]+=2.
        return values
    def act(self,public,previous_action,current_truth=None):
        native=self.native
        if not hasattr(native,'model'):return self.base.act(public,previous_action)
        step=int(native.transition_count); public=self.packet(public,step)
        if current_truth is None:action=self.base.act(public,previous_action)
        else:action=self.base.act(public,previous_action,current_truth=current_truth)
        pending=native.pending
        def array(v):return v.detach().cpu().numpy().copy()
        self.rows.append(dict(particles=array(pending['particles']),weights=array(pending['weights']),forecast=array(pending['next_particles']),received_public55=public.copy()))
        self.last_diagnostics=copy.deepcopy(native.last_diagnostics)
        self.last_diagnostics.update(copy.deepcopy(getattr(self.base,'last_diagnostics',{})))
        return action
    def observe(self,public,action,cost):
        native=self.native
        value=self.base.observe(self.packet(public,int(native.transition_count)+1),action,cost)
        self.rows[-1].update(updated_particles=native.particles.detach().cpu().numpy().copy(),updated_weights=native.weights.detach().cpu().numpy().copy())
        self.last_diagnostics.update(copy.deepcopy(getattr(self.base,'last_diagnostics',{})))
        if hasattr(self.base,'last_monitor_result'):self.monitor_rows.append(copy.deepcopy(self.base.last_monitor_result))
        return value
    def runtime_state(self):return self.base.runtime_state()
    def save(self,folder):
        if not self.rows:return
        values={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        values['train_state_scale']=self.native.model.state_scale.detach().cpu().numpy().copy()
        correction_records=getattr(self.base,'provenance',None)
        if correction_records:
            # The controller already records this pre-correction startup state.
            # Preserve it for plotting without replaying a model or changing RNG.
            values['initial_particles']=np.asarray(correction_records[0]['particles_before']).copy()
        np.savez_compressed(folder/'belief_forecast.npz',**values)
        if self.monitor_rows:(folder/'monitor.json').write_text(json.dumps(self.monitor_rows)+'\n')

def make_environment(case,method,steps):
    from bsm1_offline_benchmark.paper_shared_dataset import PaperPublic57Env
    from bsm1_offline_benchmark.dataset import forcing_for_parent
    from studies.oxygen_transfer import TransferEfficiencyEnv
    parent=case['parent']
    kwargs=dict(forcing=forcing_for_parent(parent),seed=parent['sensor_seed'],episode_days=14.)
    if method=='transfer_07':
        class TransferShift(TransferEfficiencyEnv):
            def step(self,action):
                self.eta=.7 if self.decisions>=672 else 1.
                return super().step(action)
        env=TransferShift(case['weather'],noisy_do=False,eta=1.,**kwargs)
    else:env=PaperPublic57Env(case['weather'],**kwargs)
    if method.startswith('settling_'):
        duration={'settling_6h':24,'settling_24h':96,'settling_permanent':672}[method]
        baseline=env.plant.settler.sedpar.copy();native_step=env.step
        def step(action):
            parameters=baseline.copy()
            if 672<=env.decisions<672+duration:parameters[:2]*=.2
            env.plant.settler.sedpar=parameters
            return native_step(action)
        env.step=step
    env.episode_steps=steps
    return env

def run_job(item,output,device='cuda',steps=1344,allow_different_inputs=False):
    from bsm1_offline_benchmark.stock_shared_evaluate import execute
    from studies.monitor import PassiveMismatchController
    folder=output/item['study']/item['job_id']
    if folder.exists():raise FileExistsError(f'Existing run preserved: {folder}')
    input_checks=check_job_inputs(item,allow_different_inputs)
    configure(device,legacy_precision=item['method'] in ('history','oracle','uncorrected','stock_han','stock_yang'));started=time.perf_counter()
    env=make_environment(item['case'],item['method'],steps)
    controller=make_controller(item['method'],device)
    is_model=hasattr(controller,'model') and item['method'] not in ('stock_han','stock_yang')
    if item['study'] in ('settling','oxygen-transfer','sensor-offset'):
        controller=PassiveMismatchController(controller)
    recorder=RecordedController(controller,sensor_offset=item['method']=='sensor_plus2') if is_model else None
    active=recorder if recorder else controller
    job=dict(job_id=item['job_id'],case_id=item['case']['case_id'],weather=item['case']['weather'],method='oracle' if item['method']=='oracle' else item['method'],study=item['study'],seed=17,implementation_smoke=steps!=1344,input_checks=input_checks,archived_inputs=all(v['sha256_matches'] for v in input_checks))
    result=execute(env,active,job,folder,device,setup_seconds=time.perf_counter()-started)
    if recorder:recorder.save(folder)
    # Runtime checkpoints are restart aids, not part of the public code archive.
    return result

def check_job_inputs(item,allow_different=False):
    method=item['method'];weather={'storm':'strm'}.get(item['case']['weather'],item['case']['weather'])
    required={f'forcing/Inf_{weather}_2006.txt'}
    if method not in ('public_pi','expert_pi'):
        model=method if method in ('stock_han','stock_yang') else 'joint_diffusion'
        required.add(f'models/{model}.pt')
    rows=[]
    for expected in read(ROOT/'configs/inputs.json'):
        if expected['path'] not in required:continue
        path=input_path(expected['path'])
        if not path.is_file():raise FileNotFoundError(f"Missing reproduction input {path}; see configs/inputs.json")
        actual=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        row=dict(path=expected['path'],sha256=actual,sha256_matches=actual==expected['sha256']);rows.append(row)
        if not row['sha256_matches'] and not allow_different:
            raise ValueError(f"Input hash differs from the final dissertation: {path}. Use --allow-different-inputs for a separately labelled new fit.")
    return rows

def verify_inputs(group='evaluation'):
    wanted=read(ROOT/'configs/inputs.json');rows=[]
    for item in wanted:
        if group=='evaluation' and item['path'].startswith('training/'):continue
        path=input_path(item['path']);exists=path.is_file()
        match=exists and hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==item['sha256']
        rows.append(dict(path=str(path),exists=exists,sha256_matches=match))
    print(json.dumps(rows,indent=2))
    return all(r['sha256_matches'] for r in rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list');p=sub.add_parser('verify-inputs');p.add_argument('--group',choices=('evaluation','all'),default='evaluation')
    p=sub.add_parser('run');p.add_argument('study',choices=STUDIES);p.add_argument('--job');p.add_argument('--device',default='cuda');p.add_argument('--output',type=Path,default=ROOT/'outputs');p.add_argument('--steps',type=int,default=1344);p.add_argument('--allow-different-inputs',action='store_true',help='Explicitly evaluate new fitted inputs; record differing hashes')
    args=parser.parse_args()
    if args.command=='list':
        print(json.dumps({s:[j['job_id'] for j in jobs(s)] for s in STUDIES},indent=2));return
    if args.command=='verify-inputs':raise SystemExit(0 if verify_inputs(args.group) else 1)
    if args.steps not in (1,2,1344):parser.error('Use full1344 decisions or explicit1/2-decision implementation smoke')
    chosen=jobs(args.study)
    if args.job:chosen=[j for j in chosen if j['job_id']==args.job]
    if not chosen:parser.error('Unknown job; see list')
    for item in chosen:
        print(item['job_id'],flush=True);run_job(item,args.output.resolve(),args.device,args.steps,args.allow_different_inputs)
if __name__=='__main__':main()
