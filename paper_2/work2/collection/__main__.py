"""Collect the corpora and specialists used by the retained dissertation fits.

Commands are metadata-only until --execute. Input models/data are deliberately
omitted from the archive and supplied with explicit paths. No project discovery.
"""
from pathlib import Path
import argparse
import copy
import json
from .common import read, save, identity, require, teacher_bank, stage_lock, verify_identity, bound_identity


def specialist_plan(teachers, regenerated=False):
    recipe = read(Path(__file__).with_name('recipe.json'))
    bank = teacher_bank(teachers, regenerated=regenerated)
    cells = copy.deepcopy(recipe['solver_cells'])
    for cell in cells:
        source = bank[cell['condition']]
        cell['old_teacher'] = source['path']
        cell['old_teacher_identity'] = {k: source[k] for k in ('path', 'bytes', 'sha256')}
    return dict(solver_config=recipe['solver_config'], solver_cells=cells, regenerated_teachers=regenerated,
                frozen_files=[identity(teachers), identity(Path(__file__).with_name('recipe.json'))])


def original_trunks(bank, output):
    """The reused1×parents, with native seeds, masks and1.5×wrong actors."""
    import h5py
    from stable_baselines3 import SAC
    from work2.native.environment import PaperEnv
    from work2.native.common import require_cuda, seed_all
    from .native_data import collect_rollout, write_episode
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    directory = output / 'data/HalfCheetah-v5'; directory.mkdir(parents=True)
    seed_all(13); require_cuda()
    models = {r['condition']: SAC.load(r['path'], device='cuda', custom_objects={'seed': None}) for r in bank}
    for model in models.values():
        model.policy.eval().requires_grad_(False)
    steps = 0
    for condition in range(4):
        env = PaperEnv('HalfCheetah-v5', condition, 1000)
        try:
            with h5py.File(directory / f'cell_{condition}.h5', 'x') as handle:
                for kind, count, length in (('expert', 80, 1000), ('recovery', 20, 600)):
                    for index in range(count):
                        arrays, metadata = collect_rollout(env, models, condition, index, kind)
                        write_episode(handle, f'{kind}/{index:04d}', arrays, metadata)
                        steps += len(arrays['actions'])
                        save(output / 'progress.json', dict(physical_transitions=steps, condition=condition,
                                                           kind=kind, index=index))
                        require(len(arrays['actions']) == length, 'Short parent retained; no replacement')
        finally:
            env.close()
    require(steps == 368000, 'Native1×trunk inventory differs')
    save(output / 'result.json', dict(state='completed', physical_transitions=steps,
                                    expert_teacher_identities=bank, automatic_replacements=0))


def base_plan(teachers, original, transfer_result, regenerated=False):
    bank = teacher_bank(teachers, regenerated=regenerated)
    receipt = read(transfer_result)
    require(receipt['state'] == 'completed' and receipt['all_high_gravity_cells_qualified'] and
            receipt['completed_cases'] == 12 and receipt['physical_transitions'] == 12000,
            'The native twelve-case transfer prerequisite must pass')
    for condition in range(4):
        require((Path(original) / f'data/HalfCheetah-v5/cell_{condition}.h5').is_file(), 'Missing original1×trunks')
    return dict(state='prepared_not_launched', expert_sources=bank, transfer_result=identity(transfer_result),
                gravity_levels=[1., 3.], seed=13,
                reused_recovery_behavior='Original1×physics, original1.5×intervention actors; provenance preserved',
                dataset_transitions=800000, new_collection_physical_transitions=436000,
                training_updates=0, automatic_retry=False)


def refresh_plan(specialists, normalization, output):
    from .refresh import DEFAULTS
    records = []
    for condition in range(4, 8):
        file = Path(specialists) / f'solvers/g3_c{condition}/result.json'
        row = read(file)
        require(row['state'] == 'completed' and row['qualification']['qualified'], 'Unqualified specialist')
        policy = bound_identity(row['selected_policy_identity'], file.parent/'selected.zip')
        qualification = bound_identity(row['qualification_receipt_identity'], file.parent/'qualification.json')
        require(read(qualification['path'])['qualified'] is True, 'Independent qualification failed')
        records.append(dict(condition=condition, gravity=3, selected_policy=policy,
                            qualification=qualification, solver_result=row))
    return dict(state='prepared_not_launched', config=DEFAULTS, gravity=3,
                dataset_root=str(Path(output).resolve() / 'data/gravity_3'),
                normalization=identity(normalization), solvers=records,
                source_files=[identity(Path(__file__).with_name('refresh.py'))],
                expected_split_transitions=dict(train=257600, development=36800, calibration=36800, test=36800),
                requested_parents=400, requested_transitions=368000)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('original-trunks', 'transfer', 'base', 'specialist', 'refresh', 'target'):
        q = sub.add_parser(name)
        q.add_argument('--output', type=Path, required=True)
        q.add_argument('--execute', action='store_true')
        if name in ('original-trunks', 'transfer', 'base', 'specialist', 'target'):
            q.add_argument('--teachers', type=Path, required=True, help='JSON mapping0..7 to original policy files')
            q.add_argument('--regenerated-teachers', action='store_true',
                           help='Use newly trained teachers and record their identities instead of requiring original weights')
        if name == 'base':
            q.add_argument('--original-trunks', type=Path, required=True)
            q.add_argument('--transfer-result', type=Path, required=True)
        if name == 'specialist':
            q.add_argument('--cell', required=True, choices=('g3_c4','g3_c5','g3_c6','g3_c7','g5_c4'))
        if name in ('refresh', 'target'):
            q.add_argument('--specialists', type=Path, required=True)
        if name == 'refresh':
            q.add_argument('--normalization', type=Path, required=True)
        if name == 'target':
            q.add_argument('--base-data', type=Path, required=True)
            q.add_argument('--refreshed-data', type=Path, required=True)
    return p


def main():
    a = parser().parse_args(); output = a.output.resolve()
    if a.command == 'specialist':
        plan = specialist_plan(a.teachers, a.regenerated_teachers)
        if a.execute:
            from . import specialists
            output.mkdir(parents=True, exist_ok=True)
            inputs = output / 'inputs.json'
            if inputs.exists():
                require(read(inputs) == plan, 'Specialist output has a different input manifest')
            else:
                save(inputs, plan)
            specialists.HERE = output
            cell = next(c for c in plan['solver_cells'] if c['id'] == a.cell)
            specialists.train(plan, plan['solver_config'], cell)
    elif a.command == 'original-trunks':
        bank = teacher_bank(a.teachers, a.regenerated_teachers)
        plan = dict(physical_transitions=368000, original_teacher_identities=bank)
        if a.execute:
            original_trunks(bank, output)
    elif a.command == 'transfer':
        from . import transfer
        transfer.BANK = teacher_bank(a.teachers, a.regenerated_teachers); transfer.HERE = output
        plan = transfer.declaration('cpu')
        if a.execute:
            output.mkdir(parents=True, exist_ok=True)
            transfer.execute(plan)
    elif a.command == 'base':
        from . import base_corpus
        plan = base_plan(a.teachers, a.original_trunks, a.transfer_result, a.regenerated_teachers)
        if a.execute:
            output.mkdir(parents=True, exist_ok=True)
            base_corpus.HERE = output; base_corpus.ORIGINAL = a.original_trunks.resolve()
            base_corpus.execute(plan)
    elif a.command == 'refresh':
        from . import refresh
        plan = refresh_plan(a.specialists, a.normalization, output)
        if a.execute:
            refresh.HERE = output
            refresh.PLAN = plan
            refresh.execute(output, 3, True)
    else:
        from .target import prepare, execute
        plan = prepare(a.base_data, a.refreshed_data, a.specialists, output, teachers=a.teachers,
                       regenerated_teachers=a.regenerated_teachers)
        if a.execute:
            with stage_lock(output):
                execute(plan, output)
    if not a.execute:
        print(json.dumps(dict(state='prepared_not_launched', command=a.command, plan=plan,
                              physical_transitions_executed=0, optimizer_calls=0), indent=2))


if __name__ == '__main__':
    main()
