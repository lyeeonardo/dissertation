"""Native800000-transition1×/3× corpus and frozen normalization."""
from contextlib import ExitStack
import fcntl,json,os,time,traceback
import numpy as np
from .common import save as write_json
HERE = ORIGINAL = None
SPLITS=('train','development','calibration','test')

def finalize(h5py):
    rows, ancestry = [], {}
    sums = {'observation':np.zeros(17), 'delta':np.zeros(17), 'reward':np.zeros(1)}
    squares = {k:np.zeros_like(v) for k,v in sums.items()}; counts = {k:0 for k in sums}
    required = dict(expert=(80,1000), recovery=(20,600), paired=(1000,8))
    paired_roots = {}; eligible_total=0
    for condition in range(8):
        path = HERE/'data/HalfCheetah-v5'/f'cell_{condition}.h5'
        with h5py.File(path,'r') as f:
            for kind,(groups,length) in required.items():
                assert len(f[kind]) == groups
                for key in sorted(f[kind]):
                    g=f[kind][key]; n=int(g.attrs['length']); parent=int(g.attrs['parent_id']); split=g.attrs['split']
                    assert g.attrs['complete'] and n==length and int(g.attrs['condition'])==condition and split in SPLITS
                    assert float(g.attrs['gravity_scale']) == (1. if condition<4 else 3.)
                    if parent in ancestry: assert ancestry[parent]==split
                    ancestry[parent]=split
                    obs,act,reward=g['observations'][:],g['actions'][:],g['rewards'][:]
                    assert obs.shape==(n+1,17) and act.shape==(n,6) and reward.shape==(n,)
                    assert all(np.isfinite(v).all() for v in (obs,act,reward)) and np.max(abs(act))<=1.000001
                    assert not g['terminated'][:].any()
                    elig=g['proposal_eligible'][:]; assert elig.shape==(n,)
                    if kind=='paired':
                        assert not elig.any()
                        identity=(int(g.attrs['anchor_id']),int(g.attrs['chunk_id']))
                        shared=(obs[0],act,g['prefix_observations'][:],g['prefix_actions'][:],g['prefix_rewards'][:],g['prefix_mask'][:],parent,split)
                        if identity in paired_roots:
                            prior=paired_roots[identity]
                            for before,after in zip(prior[:6],shared[:6]):np.testing.assert_array_equal(before,after)
                            assert prior[6:]==shared[6:]
                        else:paired_roots[identity]=shared
                    else:
                        expected=(~g['intervention'][:]) & (~g['inverted'][:]) & (g['speed'][:]>=.5)
                        np.testing.assert_array_equal(elig,expected)
                    eligible_total+=int(elig.sum())
                    rows.append(dict(path=str(path),episode=g.name,length=n,condition=condition,parent_id=parent,
                                     split=split,kind=kind,proposal_eligible=int(elig.sum())))
                    if kind=='expert' and split=='train':
                        values={'observation':obs[:-1], 'delta':np.diff(obs,axis=0), 'reward':reward[:,None]}
                        for field,value in values.items():
                            value=value.astype(np.float64); sums[field]+=value.sum(0); squares[field]+=(value**2).sum(0);counts[field]+=n
    assert len(rows)==8800 and len(ancestry)==800 and len(paired_roots)==1000
    split_totals={s:sum(r['length'] for r in rows if r['split']==s) for s in SPLITS}
    assert split_totals==dict(train=560000,development=80000,calibration=80000,test=80000)
    assert sum(r['length'] for r in rows)==800000
    assert all(sum(r['length'] for r in rows if r['condition']==c)==100000 for c in range(8))
    assert set(counts.values())=={448000}
    normalization={}
    for key in sums:
        mean=sums[key]/counts[key]; std=np.sqrt(np.maximum(squares[key]/counts[key]-mean**2,1e-6))
        normalization[key+'_mean']=mean.item() if key=='reward' else mean.tolist()
        normalization[key+'_std']=std.item() if key=='reward' else std.tolist()
    write_json(HERE/'splits/R_index.json',dict(episodes=rows))
    write_json(HERE/'normalization.json',normalization)
    result=dict(state='complete',env_id='HalfCheetah-v5',seed=13,gravity_levels=[1.,3.],
                arms=dict(R=dict(transitions=800000,splits=split_totals,conditions={str(c):100000 for c in range(8)},
                                 proposal_eligible=eligible_total)),
                normalization_source='Stationary expert TRAIN parents; first80 percell; new combined1x/3x laws',
                normalization_transitions=448000,ancestry_groups=800,episode_groups=8800,
                paired_same_roots_actions_prefixes_and_split=True,
                old_high_gravity_transitions_relabelled=0,train_dev_cal_test_parent_disjoint=True)
    write_json(HERE/'data_summary.json',result)
    return result


def execute(plan):
    with (HERE/'training.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (HERE/'data').exists() or (HERE/'data_collection_started.json').exists():
            raise RuntimeError('Existing collection attempt is retained; no automatic retry or overwrite')
        with (HERE/'data_collection_started.json').open('x') as f:
            json.dump(dict(pid=os.getpid(),started_at=time.time(),plan=plan),f,indent=2)
        write_json(HERE/'data_collection_declaration.json',plan)
        # Bundled imports.
        import h5py
        import torch
        from stable_baselines3 import SAC
        from work2.environment import Gravity3Env
        from work2.collection import native_data as native
        torch.set_num_threads(2);torch.manual_seed(13)
        directory=HERE/'data/HalfCheetah-v5';directory.mkdir(parents=True)
        counters=dict(reused_trunks=0,new_trunks=0,new_paired=0,auxiliary_action_chunk_steps=0)
        phase='copy';env=None;started=time.monotonic()

        class RecordedEnv(Gravity3Env):
            def __init__(self,*args,**kwargs):
                self.physical_calls=0;self.records=[];self.count_key=None;super().__init__(*args,**kwargs)
            def clear_records(self):self.records=[]
            def step(self,action):
                obs=self.unwrapped._get_obs().astype(np.float32)
                self.physical_calls+=1
                result=super().step(action)
                nxt,reward,terminated,truncated,info=result
                if self.count_key is None:raise RuntimeError('Physical collection phase must be declared')
                counters[self.count_key]+=1
                self.records.append(dict(observations=obs,next_observations=nxt.copy(),actions=np.array(action,np.float32),
                                         rewards=reward,terminated=terminated,truncated=truncated,
                                         conditions=self.condition.index,gravity_scale=self.condition.gravity,
                                         gravity_vector=self.unwrapped.model.opt.gravity.copy(),
                                         simulation_time=float(self.unwrapped.data.time)))
                return result
            def recorded_arrays(self):
                if not self.records:return {}
                return {k:np.asarray([r[k] for r in self.records]) for k in self.records[0]}

        def progress(**extra):
            write_json(HERE/'data_progress.json',dict(state='collecting',phase=phase,counts=counters,**extra))
        def stamp(g,c,origin):
            g.attrs['gravity_scale']=1. if c<4 else 3.;g.attrs['sliding_friction_scale']=1. if not(c&2) else .5
            g.attrs['joint_range_scale']=1. if not(c&1) else .8;g.attrs['gravity3_origin']=origin
        try:
            copies=[]
            for c in range(4):
                source=ORIGINAL/'data/HalfCheetah-v5'/f'cell_{c}.h5';target=directory/f'cell_{c}.h5'
                with h5py.File(source,'r') as old,h5py.File(target,'x') as new:
                    for kind,count,length in [('expert',80,1000),('recovery',20,600)]:
                        new.require_group(kind)
                        for i in range(count):
                            key=f'{i:04d}';g=old[f'{kind}/{key}']
                            assert g.attrs['complete'] and int(g.attrs['condition'])==c and int(g.attrs['source_condition'])==c
                            assert int(g.attrs['length'])==length and g.attrs['split']==native.split_for_episode(i)
                            old.copy(g,new[kind],name=key);stamp(new[f'{kind}/{key}'],c,'reused_actual1x_trunk')
                            counters['reused_trunks']+=length
                    copies.append(dict(source=str(source),target=str(target),expert_groups=80,recovery_groups=20,
                                       transitions=92000,source_gravity_scale=1.,paired_groups_copied=0))
                progress(condition=c)
            write_json(HERE/'reused_trunk_sources.json',dict(copies=copies,reused_transitions=counters['reused_trunks'],
                       original_recovery_behavior_preserved=plan['reused_recovery_behavior']))
            models={r['condition']:SAC.load(r['resolved_policy'],device='cpu',custom_objects={'seed':None}) for r in plan['expert_sources']}
            for model in models.values():model.policy.eval().requires_grad_(False)
            phase='high_gravity_trunks'
            for c in range(4,8):
                env=RecordedEnv(c);env.count_key='new_trunks'
                with h5py.File(directory/f'cell_{c}.h5','x') as f:
                    for kind,count,length in [('expert',80,1000),('recovery',20,600)]:
                        for i in range(count):
                            env.clear_records();before=env.physical_calls
                            arrays,metadata=native.collect_rollout(env,models,c,i,kind)
                            actual=env.physical_calls-before
                            assert actual==len(arrays['actions'])
                            assert all(r['gravity_scale']==3. for r in env.records)
                            env.audit_condition()
                            name=f'{kind}/{i:04d}';native.write_episode(f,name,arrays,metadata);stamp(f[name],c,'new_actual3x_trunk');f.flush()
                            if actual!=length:raise ValueError('Early native termination retained; no automatic top-up')
                            progress(condition=c,kind=kind,episode=i)
                env.close();env=None
            phase='paired_branches'
            anchors=native.anchor_table(HERE)
            env=RecordedEnv();env.reset(seed=13)
            with ExitStack() as stack:
                cells=[stack.enter_context(h5py.File(directory/f'cell_{i}.h5','a')) for i in range(8)]
                aux=stack.enter_context(h5py.File(HERE/'data/auxiliary_action_chunks.h5','x'))
                for anchor in anchors:
                    source=cells[anchor['cell']][anchor['episode']];index=anchor['step']
                    history=native.public_history(source,index,length=33)
                    state=dict(condition=anchor['cell'],elapsed=index,integration=source['integration'][index],
                               rng=json.loads(source.attrs['rng']))
                    env.clear_records();env.count_key='auxiliary_action_chunk_steps';before=env.physical_calls
                    chunks=native.branch_action_chunks(models[anchor['cell']],models[anchor['cell']^7],env,state,
                                                       history['observations'][-1],240000+anchor['anchor_id'])
                    actual=env.physical_calls-before
                    native.write_episode(aux,f"anchor_{anchor['anchor_id']:04d}",env.recorded_arrays(),
                                         dict(anchor_id=anchor['anchor_id'],source_condition=anchor['cell'],
                                              parent_id=anchor['parent_id'],split=anchor['split'],length=actual,
                                              dataset_examples=False,purpose='Two8stepbehaviorchunks; otherchunks algebraic'))
                    if actual!=16:raise ValueError('Early native auxiliary termination retained')
                    for chunk_id,chunk in enumerate(chunks):
                        for target in range(8):
                            env.clear_records();env.count_key='new_paired';obs=env.restore(state,target);env.audit_condition()
                            observations=[obs];integrations=[env.snapshot()['integration']]
                            rewards=[];terminals=[];truncateds=[]
                            for action in chunk:
                                obs,reward,terminal,truncated,_=env.step(action)
                                observations.append(obs);integrations.append(env.snapshot()['integration'])
                                rewards.append(reward);terminals.append(terminal);truncateds.append(truncated)
                                if terminal or truncated:break
                            n=len(rewards)
                            arrays=dict(observations=np.asarray(observations,np.float32),actions=chunk[:n],rewards=np.asarray(rewards,np.float32),
                                        terminated=np.asarray(terminals,bool),truncated=np.asarray(truncateds,bool),proposal_eligible=np.zeros(n,bool),
                                        integration=np.asarray(integrations),**{'prefix_'+k:v for k,v in history.items()})
                            metadata=dict(condition=target,source_condition=anchor['cell'],kind='paired',parent_id=anchor['parent_id'],
                                          split=anchor['split'],anchor_id=anchor['anchor_id'],chunk_id=chunk_id,length=n,initial_elapsed=index,
                                          rng=state['rng'],collection_cutoff=not(terminals[-1] or truncateds[-1]))
                            name=f"paired/{anchor['anchor_id']:04d}_{chunk_id}"
                            native.write_episode(cells[target],name,arrays,metadata);stamp(cells[target][name],target,'new_paired_target_law');cells[target].flush()
                            if n!=8:raise ValueError('Early native paired termination retained; no automatic top-up')
                    progress(anchor=anchor['anchor_id'],total_anchors=len(anchors))
            env.close();env=None
            phase='finalization';finalize(h5py)
            assert counters==dict(reused_trunks=368000,new_trunks=368000,new_paired=64000,auxiliary_action_chunk_steps=4000)
            result=dict(state='completed',counts=counters,dataset_transitions=800000,
                        new_dataset_transitions=432000,new_collection_physical_transitions=436000,
                        preceding_transfer_physical_transitions=12000,total_new_physical_transitions_with_transfer=448000,
                        training_updates=0,expert_fits=0,wall_seconds=time.monotonic()-started,
                        summary=str(HERE/'data_summary.json'),original_projects_modified=False)
            write_json(HERE/'data_collection_result.json',result);write_json(HERE/'data_progress.json',result)
            print(json.dumps(result,indent=2));return result
        except BaseException as error:
            if env is not None and env.records:
                np.savez_compressed(HERE/'data_collection_failed_prefix.npz',**env.recorded_arrays())
            write_json(HERE/'data_collection_failed.json',dict(state='failed',phase=phase,counts=counters,error=repr(error),
                        traceback=traceback.format_exc(),observed_calls_in_current_environment=None if env is None else env.physical_calls,
                        automatic_retry=False,automatic_expert_adaptation=False))
            raise
        finally:
            if env is not None:env.close()
