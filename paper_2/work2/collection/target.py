"""Exact final-pilot target roots and fixed source replay parent selection.

Only expert TRAIN data enter the retained pilot. All three DEV behavior groups
remain necessary for its fixed checkpoint-selection loss. The unused random
and weak TRAIN rollouts from the broader historical campaign are not collected.
"""
from pathlib import Path
import time
import traceback
from .common import read, save, identity, require, verify_identity, relocate_index, bound_identity, teacher_bank

KINDS = ('expert', 'random', 'weak')
INDICES = {'train': (0, 1, 2, 3), 'development': (7, 17)}
ROOTS = {'train': (617100001, 617100002), 'development': (617110001, 617110002)}
EVALUATION_ROOTS = {617200001, 617200002, 617200003, 617200004}


def target_rows(output):
    """Same native reset and action-noise seeds; no trajectory is opened."""
    pools = {'target_train': {}, 'target_dev': {}}
    for split_index, (split, seeds) in enumerate(ROOTS.items()):
        key = 'target_train' if split == 'train' else 'target_dev'
        for kind_index, kind in enumerate(KINDS):
            pools[key][kind] = []
            if split == 'train' and kind != 'expert':
                continue
            for index, seed in enumerate(seeds):
                pools[key][kind].append(dict(
                    path=str(Path(output) / f'target/{kind}.h5'), episode=f'{split}/{index:04d}',
                    length=500, requested_length=500, condition=4, gravity_scale=5, actual_gravity=5,
                    kind=kind, domain='five_x', split=split, reset_seed=seed,
                    root_id=f'semi_online_data_ablation_20261017:g5:c4:root:{seed}',
                    parent_id=f'semi_online_data_ablation_20261017:g5:c4:{kind}:root:{seed}',
                    action_noise_seed=[13, 20261017, 617, kind_index, split_index, index]))
    return pools


def prepare(base, refreshed, specialists, output, *, teachers, regenerated_teachers=False):
    import h5py
    base, refreshed, specialists, output = map(lambda p: Path(p).resolve(), (base, refreshed, specialists, output))
    norm = base / 'normalization.json'
    require(norm.read_bytes() == (refreshed / 'normalization.json').read_bytes(), 'Source normalization differs')
    pools = dict(source_train=[], source_dev=[], **target_rows(output))
    sources = [identity(norm)]
    ancestry = {'train': set(), 'development': set()}; source_seeds = set()
    for domain, gravity, root, conditions in (('one_x', 1, base, range(4)), ('three_x', 3, refreshed, range(4, 8))):
        rows = relocate_index(root)
        sources.append(identity(root / 'splits/R_index.json'))
        if gravity == 3:
            result = read(root / 'result.json')
            require(result['state'] == 'completed' and result['gravity'] == 3, 'Refreshed3×corpus incomplete')
            for record in result['files']:
                bound_identity(record, refreshed / ('splits/R_index.json' if Path(record['path']).name == 'R_index.json' else Path(record['path']).name))
            for condition in conditions:
                teacher = read(specialists / f'solvers/g3_c{condition}/result.json')
                require(teacher['state'] == 'completed' and teacher['qualification']['qualified'], 'Source specialist did not qualify')
                sources += [bound_identity(teacher['selected_policy_identity'],specialists/f'solvers/g3_c{condition}/selected.zip'), bound_identity(teacher['qualification_receipt_identity'],specialists/f'solvers/g3_c{condition}/qualification.json')]
        for split, indices in INDICES.items():
            selected = [r for r in rows if r['kind'] == 'expert' and r['split'] == split and
                        r['condition'] in conditions and int(Path(r['episode']).name) in indices]
            expected = {(c, i) for c in conditions for i in indices}
            require(len(selected) == len(expected) and
                    {(r['condition'], int(Path(r['episode']).name)) for r in selected} == expected,
                    'Missing or duplicate fixed source parent')
            for row in sorted(selected, key=lambda r: (r['condition'], r['episode'])):
                with h5py.File(row['path'], 'r') as handle:
                    g = handle[row['episode']]
                    require(g.attrs['complete'] and int(g.attrs['length']) == row['length'] == 1000 and
                            int(g.attrs['condition']) == row['condition'] and
                            str(g.attrs['parent_id']) == str(row['parent_id']) and g.attrs['split'] == split and
                            float(g.attrs['gravity_scale']) == gravity and
                            g['observations'].shape == (1001, 17) and g['actions'].shape == (1000, 6),
                            'Source row/HDF5 ancestry or physics differs')
                    seed = int(g.attrs['reset_seed'])
                root_id = f'{domain}:reset:{seed}:parent:{row["parent_id"]}'
                ancestry[split].add(root_id); source_seeds.add(seed)
                key = 'source_train' if split == 'train' else 'source_dev'
                pools[key].append(dict(row, domain=domain, gravity_scale=gravity, actual_gravity=gravity,
                                       reset_seed=seed, root_id=root_id, full_requested_duration=True))
    target_roots = set(ROOTS['train'] + ROOTS['development'])
    require(not ancestry['train'] & ancestry['development'] and not source_seeds & target_roots and
            not (source_seeds | target_roots) & EVALUATION_ROOTS, 'Source/target/evaluation ancestry overlaps')
    teacher = read(specialists / 'solvers/g5_c4/result.json')
    require(teacher['state'] == 'completed' and teacher['qualification']['qualified'], 'Target specialist did not qualify')
    weak=teacher_bank(teachers, regenerated_teachers)[4]
    weak=bound_identity(teacher['old_teacher_identity'],weak['path'])
    actors = dict(expert=bound_identity(teacher['selected_policy_identity'],specialists/'solvers/g5_c4/selected.zip'), weak=weak)
    sources += list(actors.values()) + [bound_identity(teacher['qualification_receipt_identity'],specialists/'solvers/g5_c4/qualification.json')]
    sources += [identity(p) for p in sorted({r['path'] for k in ('source_train', 'source_dev') for r in pools[k]})]
    return dict(schema='dissertation_final_target_pools_v1', state='prepared_not_launched',
                regenerated_teachers=regenerated_teachers, **pools,
                actors=actors, normalization=identity(norm), source_records=list({r['path']: r for r in sources}.values()),
                target_physical_transitions=4000, target_train_transitions=1000, target_development_transitions=3000,
                target_parents=8, target_independent_reset_roots=4,
                omitted='Random/weak TRAIN trajectories and all unused adaptation arms', physical_transitions=0)


def execute(plan, output):
    import h5py
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from .expert_environment import ExpertGravityEnv
    from .target_native import collect_parent, write_parent
    output = Path(output).resolve(); target = output / 'target'
    require(not target.exists() and not (output / 'started.json').exists(), 'Existing attempt retained; no retry')
    for record in plan['source_records']:
        verify_identity(record)
    target.mkdir(parents=True)
    save(output / 'started.json', dict(plan=plan, time=time.time()))
    progress = dict(state='running', physical_transitions=0, parents=0)
    queries = {'policy_predict_calls': {'expert': 0, 'weak': 0}}
    pools = {k: plan[k] for k in ('source_train', 'source_dev')}
    pools.update(target_train={k: [] for k in KINDS}, target_dev={k: [] for k in KINDS})
    torch.set_num_threads(2); torch.manual_seed(13)
    actors = {k: SAC.load(v['path'], device='cpu', custom_objects={'seed': None}) for k, v in plan['actors'].items()}
    for actor in actors.values():
        actor.policy.eval().requires_grad_(False)
    env = None; initial = {}; completed = []
    try:
        for kind in KINDS:
            env = ExpertGravityEnv(5, 4, max_episode_steps=500)
            with h5py.File(target / f'{kind}.h5', 'x') as handle:
                for key in ('target_train', 'target_dev'):
                    for declared in plan[key][kind]:
                        arrays, metadata = collect_parent(env, actors.get(kind), declared, progress, queries)
                        seed = declared['reset_seed']
                        if seed in initial:
                            np.testing.assert_array_equal(initial[seed], arrays['integration'][0])
                        else:
                            initial[seed] = arrays['integration'][0].copy()
                        write_parent(handle, declared['episode'], arrays, metadata)
                        pools[key][kind].append(dict(declared, **{k: metadata[k] for k in (
                            'length', 'full_requested_duration', 'native_termination', 'native_truncation',
                            'inverted_steps', 'proposal_eligible')}))
                        progress['parents'] += 1; completed.append(metadata)
                        save(output / 'progress.json', dict(progress, policy_queries=queries))
                        require(metadata['length'] == 500, 'Short target parent retained; no replacement')
            env.close(); env = None
        require(progress['physical_transitions'] == 4000 and progress['parents'] == 8, 'Target inventory differs')
        normalization = output / 'normalization.json'
        normalization.write_bytes(Path(plan['normalization']['path']).read_bytes())
        files = [identity(target / f'{k}.h5') for k in KINDS] + [identity(normalization)]
        save(output / 'result.json', dict(progress, state='completed', policy_queries=queries,
             completed_parent_rows=completed, shared_DEV_initial_states_exact=True, automatic_replacements=0))
        save(output / 'pools.json', dict(state='completed', **pools, normalization=identity(normalization),
             files=files, source_records=plan['source_records'], actual_physical_transitions=4000,
             collection_result=identity(output / 'result.json')))
    except BaseException as error:
        partial = getattr(error, 'collection_partial', None)
        if partial is not None:
            np.savez_compressed(output / 'failed_prefix.npz', **partial[0]); save(output / 'failed_prefix.json', partial[1])
        returned = getattr(error, 'collection_uncommitted', None)
        if returned is not None:
            np.savez_compressed(output / 'failed_returned_transition.npz', **returned)
        save(output / 'result.json', dict(progress, state='failed', error=repr(error),
             completed_parent_rows=completed, traceback=traceback.format_exc(), automatic_retry=False))
        raise
    finally:
        if env is not None:
            env.close()
