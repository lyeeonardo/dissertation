"""Native qualified3× replay collection,400parents and368000transitions."""
from __future__ import annotations
import os,time,traceback
from collections import Counter
from pathlib import Path
from .common import save as write_json,identity,verify_identity,stage_lock
HERE = None
PLAN = None
def collection_plan(root, gravity):
    if PLAN is None or gravity != 3 or Path(root).resolve() != HERE:
        raise ValueError('Use the explicit collection CLI and prepared3×plan')
    return PLAN
DEFAULTS = dict(seed=13, expert_parents_per_cell=80, recovery_parents_per_cell=20,
    expert_steps=1000,recovery_steps=600,reset_seed_base=471000000,physical_gravities=[3,5])
SPLITS=('train','development','calibration','test')

def split_for_episode(index):
    slot = index % 10
    return SPLITS[0 if slot < 7 else slot - 6]


def collect_parent(env, models, condition, index, kind, gravity, config, progress):
    """Native trunk protocol with a same-gravity intervention actor and fresh IDs."""
    import numpy as np
    recovery = kind == 'recovery'
    seed = config['reset_seed_base'] + gravity * 1000000 + condition * 10000 + int(recovery) * 1000 + index
    noise_seed = [config['seed'], 20261017, gravity, condition, index, int(recovery)]
    rng = np.random.default_rng(np.random.SeedSequence(noise_seed))
    requested = config['recovery_steps' if recovery else 'expert_steps']
    obs, _ = env.reset(seed=seed)
    snapshot = env.snapshot()
    stores = dict(observations=[obs.copy()], integration=[snapshot['integration']],
                  actions=[], rewards=[], terminated=[], truncated=[], proposal_eligible=[],
                  intervention=[], speed=[], inverted=[], gravity_scale=[],
                  gravity_vector=[], simulation_time=[])
    periods = []
    if recovery:
        periods = [(s, s + int(rng.integers(5, 21))) for s in
                   (int(rng.integers(100, 151)), int(rng.integers(375, 426)))]
    wrong = condition ^ (1 << int(rng.integers(0, 2)))
    assert wrong in range(4, 8)
    metadata = dict(condition=condition, source_condition=condition, kind=kind,
                    episode_index=index, parent_id=f'expert_refresh_20261017:g{gravity}:c{condition}:{kind}:{index:04d}',
                    split=split_for_episode(index), reset_seed=seed, rng=snapshot['rng'],
                    requested_length=requested, gravity_scale=float(gravity),
                    sliding_friction_scale=1. if not condition & 2 else .5,
                    joint_range_scale=1. if not condition & 1 else .8,
                    action_noise_seed=noise_seed, intervention_windows=periods,
                    wrong_condition=wrong, wrong_actor_gravity=float(gravity),
                    initial_elapsed=0, proposal_rule='expert-active, noninverted, speed >= 0.5 m/s')

    def arrays():
        booleans = {'terminated', 'truncated', 'proposal_eligible', 'intervention', 'inverted'}
        doubles = {'integration', 'gravity_vector', 'simulation_time'}
        return {key: np.asarray(value, bool if key in booleans else np.float64 if key in doubles else np.float32)
                for key, value in stores.items()}

    uncommitted = None
    try:
        for step in range(requested):
            intervene = any(a <= step < b for a, b in periods)
            source = models[wrong] if intervene and index % 2 == 0 else models[condition]
            action, _ = source.predict(obs, deterministic=True)
            if intervene and index % 2:
                action = np.clip(.3 * action + rng.normal(0, .5, 6), -1, 1).astype(np.float32)
            previous_obs = obs.copy()
            obs, reward, terminal, truncated, info = env.step(action)
            progress['physical_transitions'] += 1
            # Preserve the returned transition even if a later diagnostic or
            # integration-state extraction fails before the aligned row commits.
            uncommitted = dict(observations=np.stack([previous_obs, obs]), actions=np.asarray(action)[None],
                               rewards=np.asarray([reward]), terminated=np.asarray([terminal]),
                               truncated=np.asarray([truncated]), gravity_scale=np.asarray([gravity]),
                               step=np.asarray([step]))
            diagnostic = env.diagnostics()
            speed = float(info.get('x_velocity', 0.))
            for key, value in dict(observations=obs.copy(), integration=env.snapshot()['integration'],
                                   actions=action.copy(), rewards=reward, terminated=terminal,
                                   truncated=truncated, proposal_eligible=not intervene and not diagnostic['inverted'] and speed >= .5,
                                   intervention=intervene, speed=speed, inverted=diagnostic['inverted'],
                                   gravity_scale=float(gravity), gravity_vector=env.unwrapped.model.opt.gravity.copy(),
                                   simulation_time=float(env.unwrapped.data.time)).items():
                stores[key].append(value)
            uncommitted = None
            if terminal or truncated:
                break
    except BaseException as error:
        error.collection_partial = (arrays(), dict(metadata, length=len(stores['actions']), complete=False))
        error.collection_uncommitted = uncommitted
        raise
    metadata.update(length=len(stores['actions']),
                    collection_cutoff=not (stores['terminated'][-1] or stores['truncated'][-1]),
                    full_requested_duration=len(stores['actions']) == requested)
    return arrays(), metadata


def execute(root, gravity, parent_lock_held):
    plan = collection_plan(root, gravity)
    output = HERE / f'data/gravity_{gravity}'
    with stage_lock(HERE / f'collection/gravity_{gravity}', parent_lock_held):
        if output.exists():
            raise RuntimeError('Existing dataset attempt retained; no retry/overwrite')
        output.mkdir(parents=True)
        started = time.monotonic()
        progress = dict(state='running', gravity=gravity, pid=os.getpid(), physical_transitions=0, parents=0)
        write_json(output / 'declaration.json', plan)
        write_json(output / 'progress.json', progress)
        # Bundled imports; no source-workspace lookup.
        import h5py
        import numpy as np
        import torch
        from stable_baselines3 import SAC
        from work2.collection.expert_environment import ExpertGravityEnv
        from work2.collection.native_data import write_episode
        torch.set_num_threads(2); torch.manual_seed(plan['config']['seed'])
        rows, env, active_name = [], None, None
        current_arrays, current_metadata = None, None
        try:
            models = {row['condition']: SAC.load(row['selected_policy']['path'], device='cpu',
                                                 custom_objects={'seed': None}) for row in plan['solvers']}
            for model in models.values():
                model.policy.eval().requires_grad_(False)
            solver_sources = {row['condition']: row for row in plan['solvers']}
            for condition in range(4, 8):
                env = ExpertGravityEnv(gravity, condition, max_episode_steps=1000)
                physics = env.audit_condition()
                path = output / f'cell_{condition}.h5'
                with h5py.File(path, 'x') as handle:
                    for kind in ('expert', 'recovery'):
                        for index in range(plan['config'][f'{kind}_parents_per_cell']):
                            active_name = f'{kind}/{index:04d}'
                            a, metadata = collect_parent(env, models, condition, index, kind, gravity, plan['config'], progress)
                            current_arrays, current_metadata = a, metadata
                            metadata['teacher_policy'] = solver_sources[condition]['selected_policy']
                            metadata['wrong_actor_policy'] = solver_sources[metadata['wrong_condition']]['selected_policy']
                            env.audit_condition()
                            write_episode(handle, active_name, a, dict(metadata, physics=physics))
                            rows.append(dict(path=str(path), episode=active_name, condition=condition, kind=kind,
                                             split=metadata['split'], parent_id=metadata['parent_id'], length=metadata['length'],
                                             gravity_scale=gravity, full_requested_duration=metadata['full_requested_duration'],
                                             inverted_steps=int(a['inverted'].sum()), proposal_eligible=int(a['proposal_eligible'].sum()),
                                             native_termination=bool(a['terminated'].any()), native_truncation=bool(a['truncated'].any())))
                            progress['parents'] += 1
                            write_json(output / 'progress.json', dict(progress, condition=condition, kind=kind, episode_index=index))
                            current_arrays, current_metadata = None, None
                            if not metadata['full_requested_duration']:
                                raise ValueError('Early termination retained; no automatic replacement or completion claim')
                env.close(); env = None
            counts = Counter()
            for row in rows:
                counts[row['split']] += row['length']
            assert len(rows) == len({r['parent_id'] for r in rows}) == 400
            assert dict(counts) == plan['expected_split_transitions']
            assert progress['physical_transitions'] == 368000
            for record in plan['source_files'] + [plan['normalization']]:
                verify_identity(record)
            for solver in plan['solvers']:
                verify_identity(solver['selected_policy']); verify_identity(solver['qualification'])
            write_json(output / 'splits/R_index.json', dict(episodes=rows, physical_gravity=gravity))
            # Byte-identical frozen normalization; no target-data normalization fit.
            (output / 'normalization.json').write_bytes(Path(plan['normalization']['path']).read_bytes())
            files = [identity(output / f'cell_{c}.h5') for c in range(4, 8)]
            files += [identity(output / 'splits/R_index.json'), identity(output / 'normalization.json')]
            result = dict(progress, state='completed', requested_transitions=368000,
                          split_transitions=dict(counts), parent_split_disjoint=True,
                          proposal_eligible=sum(r['proposal_eligible'] for r in rows),
                          inverted_steps=sum(r['inverted_steps'] for r in rows),
                          native_terminations=sum(r['native_termination'] for r in rows),
                          native_truncations=sum(r['native_truncation'] for r in rows),
                          native_early_terminations=0, automatic_replacements=0, elapsed_seconds=time.monotonic()-started,
                          files=files, index=str(output / 'splits/R_index.json'),
                          solver_sources=plan['solvers'], normalization_source=plan['normalization'])
            write_json(output / 'result.json', result)
            write_json(output / 'progress.json', result)
            return result
        except BaseException as error:
            partial = getattr(error, 'collection_partial', None)
            if partial is None and current_arrays is not None:
                partial = (current_arrays, current_metadata)
            if partial is not None:
                a, metadata = partial
                np.savez_compressed(output / 'failed_prefix.npz', **a)
                write_json(output / 'failed_prefix.json', metadata)
            uncommitted = getattr(error, 'collection_uncommitted', None)
            if uncommitted is not None:
                np.savez_compressed(output / 'failed_returned_transition.npz', **uncommitted)
            write_json(output / 'result.json', dict(progress, state='failed', error=repr(error),
                       traceback=traceback.format_exc(), completed_parent_rows=rows,
                       elapsed_seconds=time.monotonic()-started, automatic_retry=False))
            raise
        finally:
            if env is not None:
                env.close()
