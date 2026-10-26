"""Native prerequisite check for original teachers transferred to3×."""
import json,os,time,traceback
import numpy as np
from .common import save as atomic_json
HERE = None
BANK = []
CONDITIONS=(4,5,6,7)
SEEDS=(391000000,391000001,391000002)
def expert_sources(conditions):
    return [r for r in BANK if r['condition'] in conditions]

def declaration(device):
    bank = expert_sources(tuple(range(8)))
    return dict(study='gravity3x_frozen_expert_transfer', env_id='HalfCheetah-v5',
                conditions=list(CONDITIONS), reset_seeds=list(SEEDS), assigned_steps_per_case=1000,
                native_dt=.05, assigned_cases=12, maximum_physical_transitions=12000,
                inference_device=device, expert_sources=[r for r in bank if r['condition']>=4],
                policies={str(r['condition']): r['resolved_policy'] for r in bank},
                training_updates=0, deterministic_actions=True,
                qualification=dict(each_of_three_episodes_required=True, minimum_mean_speed_m_s=.5,
                                   observed_steps=1000, inverted_steps=0, early_native_termination_allowed=False),
                scope='Limited data-collection readiness on three fixed seeds per cell; not benchmark proof or a new expert fit.',
                failures='Retain all observed prefixes, adverse outcomes and native flags; no automatic rerun or adaptation.',
                labels='Truth retained only by evaluator; source actors receive native 17-dimensional observations.')

def save_arrays(path, arrays):
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(path)

def run_case(model, condition, seed, case_id, output):
    from work2.environment import Gravity3Env
    env = Gravity3Env(condition, max_episode_steps=1000)
    observation, _ = env.reset(seed=seed)
    initial_law = env.audit_condition()
    lists = {k: [] for k in ('actions', 'rewards', 'speed', 'inverted', 'terminated', 'truncated',
                             'gravity_vector', 'simulation_time')}
    observations = [observation.copy()]
    started = time.monotonic()
    stem = output/f'case_{case_id:04d}'

    def arrays():
        n = len(lists['rewards'])
        return dict(observations=np.asarray(observations, np.float32).reshape(n+1,17),
                    actions=np.asarray(lists['actions'], np.float32).reshape(n,6),
                    rewards=np.asarray(lists['rewards'], np.float64), speed=np.asarray(lists['speed'], np.float64),
                    inverted=np.asarray(lists['inverted'], bool), terminated=np.asarray(lists['terminated'], bool),
                    truncated=np.asarray(lists['truncated'], bool),
                    gravity_vector=np.asarray(lists['gravity_vector'], np.float64).reshape(n,3),
                    simulation_time=np.asarray(lists['simulation_time'], np.float64),
                    conditions=np.full(n,condition,np.int64), gravity_scale=np.full(n,3.,np.float64))
    try:
        for step in range(1000):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            diagnostics = env.diagnostics()
            gravity = env.unwrapped.model.opt.gravity.copy()
            np.testing.assert_allclose(gravity, env.nominal['gravity']*3., rtol=0, atol=1e-12)
            if not np.isclose(env.unwrapped.data.time, (step+1)*.05, rtol=0, atol=1e-8):
                raise AssertionError('Native simulation clock differs')
            observations.append(observation.copy())
            lists['actions'].append(np.asarray(action).copy())
            lists['rewards'].append(float(reward)); lists['speed'].append(float(info['x_velocity']))
            lists['inverted'].append(bool(diagnostics['inverted']))
            lists['terminated'].append(bool(terminated)); lists['truncated'].append(bool(truncated))
            lists['gravity_vector'].append(gravity); lists['simulation_time'].append(float(env.unwrapped.data.time))
            if terminated or truncated:
                break
        saved = arrays(); n = len(saved['rewards'])
        if not n or not np.isfinite(saved['observations']).all() or not np.isfinite(saved['rewards']).all():
            raise AssertionError('No valid finite episode')
        if not (saved['terminated'][-1] or saved['truncated'][-1]):
            raise AssertionError('Native TimeLimit was not reached at the declared cap')
        mean_speed = float(saved['speed'].mean())
        inversions = int(saved['inverted'].sum())
        qualified = n == 1000 and not bool(saved['terminated'][-1]) and inversions == 0 and mean_speed >= .5
        row = dict(case_id=case_id, condition=condition, bits=f'{condition:03b}', reset_seed=seed,
                   assigned_steps=1000, observed_steps=n, return_total=float(saved['rewards'].sum()),
                   mean_speed_m_s=mean_speed, inverted_steps=inversions,
                   native_termination=bool(saved['terminated'][-1]), native_truncation=bool(saved['truncated'][-1]),
                   qualifies=qualified, initial_law=initial_law, final_law=env.audit_condition(),
                   every_observed_transition_has_3x_gravity=True, wall_seconds=time.monotonic()-started,
                   trace=str(stem.with_suffix('.npz')))
        save_arrays(stem.with_suffix('.npz'), saved)
        atomic_json(stem.with_suffix('.json'), dict(state='completed', **row))
        return row
    except BaseException as error:
        saved = arrays()
        save_arrays(stem.with_name(stem.name+'_failed_prefix.npz'), saved)
        atomic_json(stem.with_suffix('.json'), dict(state='failed', case_id=case_id, condition=condition,
                    reset_seed=seed, observed_steps=len(saved['rewards']), error=repr(error),
                    traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        env.close()

def execute(plan):
    # Preserve the actual completed attempt location; its marker prevents reruns.
    output = HERE/'expert_transfer'
    output.mkdir(exist_ok=True)
    # Atomic attempt marker: no automatic reuse/restart of an interrupted diagnostic.
    with (output/'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), started_at=time.time(), declaration=plan), f, indent=2)
    atomic_json(output/'declaration.json', plan)
    # Bundled imports.
    import torch
    from stable_baselines3 import SAC
    from work2.environment import physical_checks
    torch.set_num_threads(2)
    torch.manual_seed(13)
    rows = []
    start = time.monotonic()
    try:
        atomic_json(output/'physical_checks.json', physical_checks())
        for source in plan['expert_sources']:
            model = SAC.load(source['resolved_policy'], device=plan['inference_device'], custom_objects={'seed':None})
            model.policy.eval().requires_grad_(False)
            for seed in SEEDS:
                row = run_case(model, source['condition'], seed, len(rows), output)
                rows.append(row)
                atomic_json(output/'progress.json', dict(state='running', completed_cases=len(rows),
                            observed_transitions=sum(r['observed_steps'] for r in rows), latest=row))
                print(json.dumps({k:row[k] for k in ('case_id','condition','return_total','mean_speed_m_s','inverted_steps','qualifies')}), flush=True)
            del model
        cells = {}
        for c in CONDITIONS:
            cell = [r for r in rows if r['condition'] == c]
            assert len(cell) == 3
            cells[str(c)] = dict(qualifies=all(r['qualifies'] for r in cell), qualified=all(r['qualifies'] for r in cell),
                                 mean_return=float(np.mean([r['return_total'] for r in cell])),
                                 mean_speed_m_s=float(np.mean([r['mean_speed_m_s'] for r in cell])),
                                 observed_steps=sum(r['observed_steps'] for r in cell),
                                 inverted_steps=sum(r['inverted_steps'] for r in cell), case_ids=[r['case_id'] for r in cell])
        result = dict(state='completed', completed_cases=len(rows), cells=cells, per_condition=cells, rows=rows,
                      policies=plan['policies'], qualified=all(r['qualifies'] for r in cells.values()),
                      all_high_gravity_cells_qualified=all(r['qualifies'] for r in cells.values()),
                      physical_transitions=sum(r['observed_steps'] for r in rows), maximum_physical_transitions=12000,
                      training_updates=0, wall_seconds=time.monotonic()-start,
                      automatic_expert_adaptation=False, no_automatic_reruns=True,
                      scope=plan['scope'])
        atomic_json(output/'result.json', result)
        return result
    except BaseException as error:
        atomic_json(output/'failed.json', dict(state='failed', error=repr(error), traceback=traceback.format_exc(),
                    completed_cases=len(rows), automatic_retry=False, automatic_expert_adaptation=False))
        raise
