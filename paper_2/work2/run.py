"""Execute only fixed dissertation cases using explicitly supplied checkpoints."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import traceback
from .io import read, save, identity

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / 'configs'
STUDIES = ('benchmark', 'preference_a', 'preference_b', 'all_conditions', 'gravity_sweep', 'adaptation')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', choices=STUDIES, required=True)
    p.add_argument('--assets', type=Path, help='Directory of separately provided shared.pt, history.pt and dadp.pt')
    p.add_argument('--output', type=Path)
    p.add_argument('--method', action='append', choices=('genaif', 'history', 'dadp', 'oracle', 'reward_only'))
    p.add_argument('--case', action='append', type=int, help='Exact saved case_id; omission uses every case in the study')
    p.add_argument('--arm', choices=('zero', 'replay_only', 'expert_1000'), default='zero', help='Adaptation checkpoint suffix; e.g. shared_expert_1000.pt')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--regenerated-checkpoints', action='store_true', help='Evaluate explicitly regenerated fits; mark results as a new reproduction rather than exact archived weights')
    p.add_argument('--list', action='store_true', help='Print fixed design without loading models or physics')
    return p


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['list']:
        cases = read(CONFIG / 'cases.json')
        print(json.dumps({name: dict(cases=len(cases[name]),steps=sorted({c['steps'] for c in cases[name]})) for name in STUDIES},indent=2))
        return
    args = parser().parse_args(argv)
    cases = read(CONFIG / 'cases.json')[args.study]
    if args.case:
        selected = set(args.case)
        if not selected <= {c['case_id'] for c in cases}:
            raise ValueError('A requested case is absent from the frozen study')
        cases = [c for c in cases if c['case_id'] in selected]
    default = ['genaif'] if args.study == 'gravity_sweep' else ['genaif', 'reward_only'] if args.study.startswith('preference') else ['genaif', 'history', 'dadp', 'oracle']
    methods = args.method or default
    if args.study == 'gravity_sweep' and methods != ['genaif']:
        raise ValueError('Only final GenAIF was evaluated in the fixed-bank gravity sweep')
    if args.study.startswith('preference') and not set(methods) <= {'genaif', 'reward_only'}:
        raise ValueError('The matched preference studies compare only reward-only and final GenAIF')
    if not args.study.startswith('preference') and 'reward_only' in methods:
        raise ValueError('Reward-only control is archived only for the included preference comparison')
    if args.arm != 'zero' and args.study != 'adaptation':
        raise ValueError('Model updates belong to the separate adaptation study')
    design = dict(study=args.study, methods=methods, cases=cases, arm=args.arm, evidence_temperature=.007242573030413831,
                  model_bank=[1, 3] if args.study != 'adaptation' else 'explicit per case: 1/5 target or 1/3 retention',
                  training_updates=0, selected_by_outcomes=args.study == 'all_conditions')
    if args.list:
        print(json.dumps(design, indent=2)); return
    if args.assets is None or args.output is None:
        raise ValueError('--assets and --output are required to execute')
    assets, output = args.assets.resolve(), args.output.resolve()
    asset_map = read(assets).get('checkpoints', {}) if assets.is_file() else None
    asset_spec = read(Path(__file__).resolve().parent / 'required_assets.json')
    required = asset_spec['checkpoints']
    paths, identities = {}, {}
    for method in methods:
        key = method if method in ('history', 'dadp') else 'shared'
        name = key + ('' if args.arm == 'zero' else '_' + args.arm) + '.pt'
        entry = asset_map.get(name, asset_map.get(key)) if asset_map is not None else None
        path = Path(entry['path'] if isinstance(entry, dict) else entry) if entry is not None else assets / name
        if entry is not None and not path.is_absolute(): path = assets.parent / path
        if not path.is_file():
            raise FileNotFoundError(f'Missing {name}; see required_assets.json. Models are intentionally omitted.')
        paths[method] = path
        if name not in identities:
            identities[name] = identity(path)
            expected = required[key] if args.arm == 'zero' else asset_spec['adapted_checkpoints'][key + '_' + args.arm]
            if not args.regenerated_checkpoints and identities[name]['sha256'] != expected['sha256']:
                raise ValueError('Input is not the selected dissertation checkpoint: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'design.json', dict(design, checkpoints=identities, device=args.device,
                                     exact_runtime_reproduction_requires_original_cuda_environment=True, regenerated_checkpoints=args.regenerated_checkpoints))
    import numpy as np
    import torch
    import mujoco
    from work2.native.common import seed_all
    from work2.native.environment import Condition
    from .environment import Gravity3Env
    from .policies import make_policy
    from . import episode, preference_episode, sweep_episode, adaptation_control, baseline_adaptation_episode
    from .metrics import episode_metrics
    config = read(CONFIG / 'controller.json')
    seed_all(13, threads=4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    class SweepEnv(Gravity3Env):
        def __init__(self, condition=4, max_episode_steps=500, *, actual_gravity):
            self.actual_gravity = float(actual_gravity)
            super().__init__(condition=condition, max_episode_steps=max_episode_steps)
        def apply_condition(self, condition):
            if condition != 4:
                raise ValueError('The gravity sweep uses evaluator slot 4')
            spec = replace(Condition.from_index(condition), gravity=self.actual_gravity)
            model, data = self.unwrapped.model, self.unwrapped.data
            model.opt.gravity[:] = self.nominal['gravity'] * spec.gravity
            model.geom_friction[:] = self.nominal['friction']
            model.geom_friction[:, 0] *= spec.sliding_friction
            model.jnt_range[:] = self.nominal['joint_range']
            ranges = self.nominal['joint_range'][self.joint_ids]
            center = ranges.mean(1)
            half_width = (ranges[:, 1] - ranges[:, 0]) * .5 * spec.joint_range
            model.jnt_range[self.joint_ids] = np.stack([center-half_width, center+half_width], 1)
            self.condition = spec
            warmstart = data.qacc_warmstart.copy()
            mujoco.mj_forward(model, data)
            data.qacc_warmstart[:] = warmstart

    rows = []
    for method in methods:
        for case in cases:
            directory = output / method / f"case_{case['case_id']:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            policy = engine = None
            try:
                high = case.get('high_gravity', 3)
                policy, engine, state = make_policy(method, paths[method], case['sampling_seed'], args.device, high)
                del state
                native_method = {'genaif': 'upright_map_replay', 'reward_only': 'map_replay_evidence'}.get(method, method)
                plan = dict(env_id='HalfCheetah-v5', high_gravity=high, temperature=config['evidence_temperature'], reference_speeds=config['reference_speeds'])
                if args.study == 'gravity_sweep':
                    gravity = case['actual_gravity']; plan['actual_gravity'] = gravity
                    arrays, metadata = sweep_episode.run_episode(policy, case, native_method, plan,
                        env_factory=lambda **kw: SweepEnv(actual_gravity=gravity, **kw))
                elif args.study == 'adaptation':
                    if method in ('history', 'dadp'):
                        arrays, metadata = baseline_adaptation_episode.run_episode(policy, case, native_method, plan,
                            env_factory=adaptation_control.SlotGravityEnv)
                    else:
                        arrays, metadata = adaptation_control.run_episode(policy, case, native_method, plan,
                            env_factory=adaptation_control.SlotGravityEnv)
                elif args.study.startswith('preference'):
                    arrays, metadata = preference_episode.run_episode(policy, case, native_method, plan, env_factory=Gravity3Env)
                else:
                    arrays, metadata = episode.run_episode(policy, case, native_method, plan,
                        env_factory=Gravity3Env, prefix_validator=lambda arrays: None)
                np.savez_compressed(directory / 'trace.npz', **arrays)
                # Use common event metrics; the older special 15 s segment summary is not applied to the 20 s demo.
                metrics = episode_metrics('main', case, arrays, method, plan)
                row = dict(state='completed', study=args.study, method=method, case=case,
                           metrics=metrics, episode_metadata=metadata,
                           query_counts=engine.stats.snapshot() if engine is not None else None,
                           trace=str(directory / 'trace.npz'))
                save(directory / 'result.json', row); rows.append(row)
                save(output / 'results.json', rows)
                print(f"{method} case {case['case_id']}: return={metrics['return']:.8f}", flush=True)
            except BaseException as error:
                partial = getattr(error, 'repair_partial', None)
                if partial: np.savez_compressed(directory / 'failed_prefix.npz', **partial)
                pending = getattr(error, 'repair_uncommitted', None)
                if pending: np.savez_compressed(directory / 'uncommitted_transition.npz', **pending)
                save(directory / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                     phase=getattr(error, 'repair_phase', None), automatically_retried=False))
                raise
            finally:
                if engine is not None: engine.close()
                del policy
    print(f'Saved {len(rows)} episodes to {output}')

if __name__ == '__main__': main()
