"""Native SAC fit, development selection and independent qualification."""
from __future__ import annotations
import copy,fcntl,functools,json,os,shutil,time,traceback
from pathlib import Path
from .common import save as atomic_json,append_json,digest,identity,require,verify_files
HERE = None

def runtime_paths():
    pass  # Every runtime import is bundled in this distribution.


def quality(summary, reference, config, cell, *, paired=False):
    episodes = summary['episodes']
    margin = max(config['improvement_absolute'], config['improvement_fraction'] * abs(reference['return']))
    old_threshold = reference['return'] + margin
    dadp_threshold = (config['dadp_multiplier'] * cell['dadp_reference_return']) if cell['gravity'] == 3. else None
    physical = all(e['native_steps'] == config['episode_steps'] and not e['native_termination'] and
                   e['inverted_steps'] == 0 and e['mean_speed'] >= config['min_each_speed'] for e in episodes)
    score = summary['return'] > 0 and summary['return'] >= old_threshold and (
        dadp_threshold is None or summary['return'] >= dadp_threshold)
    differences = []
    if paired:
        require([e['seed'] for e in episodes] == [e['seed'] for e in reference['episodes']], 'Qualification pairing changed')
        differences = [new['return'] - old['return'] for new, old in zip(episodes, reference['episodes'], strict=True)]
    wins = sum(d > 0 for d in differences)
    return dict(qualified=bool(physical and score and (not paired or wins >= config['qualification_min_paired_wins'])),
                physical_gate=physical, return_gate=score, old_teacher_return=reference['return'],
                old_teacher_margin=margin, old_teacher_threshold=old_threshold, dadp_threshold=dadp_threshold,
                paired_wins=wins if paired else None, paired_return_differences=differences,
                criterion='Full native duration, zero inversions/deaths, every mean speed≥1, positive mean return and declared improvements')


def make_training_env(gravity, condition, worker, directory, episode_steps):
    """Construct inside its child process; retain every episode summary/failure."""
    runtime_paths()
    import numpy as np
    from stable_baselines3.common.monitor import Monitor
    from work2.collection.expert_environment import ExpertGravityEnv
    from work2.native.evaluation import atomic_npz

    class TrackedEnv(ExpertGravityEnv):
        def __init__(self):
            self.total_native_steps, self.completed_episodes, self.total_inverted_steps = 0, 0, 0
            self.trace = {}
            super().__init__(gravity, condition, max_episode_steps=episode_steps)

        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            self.trace = dict(observations=[observation.copy()], actions=[], rewards=[], speed=[], inverted=[])
            return observation, info

        def step(self, action):
            before = float(self.unwrapped.data.time)
            try:
                observation, reward, terminated, truncated, info = super().step(action)
                self.total_native_steps += 1
                inverted = bool(self.diagnostics()['inverted']); self.total_inverted_steps += int(inverted)
                for key, value in dict(observations=observation.copy(), actions=np.asarray(action).copy(),
                                       rewards=reward, speed=float(info['x_velocity']), inverted=inverted).items():
                    self.trace[key].append(value)
                if terminated or truncated:
                    self.completed_episodes += 1
                    append_json(Path(directory) / f'worker_{worker:02d}_episodes.jsonl',
                                dict(worker=worker, episode=self.completed_episodes, gravity=gravity, condition=condition,
                                     native_steps=len(self.trace['rewards']), total_native_steps=self.total_native_steps,
                                     return_value=float(np.sum(self.trace['rewards'])),
                                     mean_speed=float(np.mean(self.trace['speed'])),
                                     inverted_steps=int(np.sum(self.trace['inverted'])),
                                     terminated=bool(terminated), truncated=bool(truncated)))
                return observation, reward, terminated, truncated, info
            except BaseException as error:
                prefix = {key: np.asarray(value) for key, value in self.trace.items()}
                path = Path(directory) / f'worker_{worker:02d}_failed_prefix.npz'
                atomic_npz(path, prefix)
                atomic_json(path.with_suffix('.json'), dict(state='failed', worker=worker, gravity=gravity,
                            condition=condition, total_observed_native_steps=self.total_native_steps,
                            current_simulation_time=float(self.unwrapped.data.time), previous_simulation_time=before,
                            elapsed_steps_in_wrapper=self.elapsed, error=repr(error), traceback=traceback.format_exc(),
                            failing_native_call_may_have_advanced_physics=True, automatic_retry=False))
                raise

        def training_snapshot(self):
            return dict(physical_state=self.snapshot(), physical_audit=self.audit_condition(),
                        action_space_rng=copy.deepcopy(self.action_space.np_random.bit_generator.state),
                        total_native_steps=self.total_native_steps, completed_episodes=self.completed_episodes,
                        total_inverted_steps=self.total_inverted_steps,
                        current_episode_prefix={key: np.asarray(value) for key, value in self.trace.items()})

    return Monitor(TrackedEnv())


def evaluate(model, config, cell, seeds, directory, purpose, ledger):
    import numpy as np
    from work2.collection.expert_environment import ExpertGravityEnv
    from work2.native.evaluation import atomic_npz
    path = Path(directory); path.mkdir(parents=True, exist_ok=False)
    rows = []
    env = ExpertGravityEnv(cell['gravity'], cell['condition'], config['episode_steps'])
    try:
        for number, seed in enumerate(seeds):
            trace = {}
            def add(key, value): trace.setdefault(key, []).append(np.asarray(value).copy())
            stem = path / f'episode_{number:02d}'
            atomic_json(stem.with_name(stem.name + '_started.json'), dict(seed=seed, purpose=purpose, started_at=time.time()))
            try:
                observation, _ = env.reset(seed=seed); audit = env.audit_condition()
                add('observations', observation)
                for step in range(config['episode_steps']):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    # Account and retain the completed physical transition immediately.
                    ledger[purpose] = ledger.get(purpose, 0) + 1
                    for key, value in dict(observations=observation, actions=action, rewards=reward,
                                           speed=float(info['x_velocity']), inverted=env.diagnostics()['inverted'],
                                           terminated=terminated, truncated=truncated,
                                           gravity_scale=env.condition.gravity, condition=env.condition.index,
                                           gravity_vector=env.unwrapped.model.opt.gravity.copy(),
                                           sliding_friction_scale=env.condition.sliding_friction,
                                           joint_range_scale=env.condition.joint_range,
                                           simulation_time=float(env.unwrapped.data.time)).items(): add(key, value)
                    if terminated or truncated:
                        break
                arrays = {key: np.asarray(value) for key, value in trace.items()}
                n = len(arrays['rewards'])
                require(0 < n <= config['episode_steps'] and all(np.isfinite(v).all() for v in arrays.values()),
                        'Invalid evaluation trace')
                np.testing.assert_allclose(arrays['simulation_time'], (np.arange(n) + 1) * .05, atol=1e-8, rtol=0)
                np.testing.assert_array_equal(arrays['gravity_scale'], np.full(n, cell['gravity']))
                np.testing.assert_array_equal(arrays['condition'], np.full(n, cell['condition']))
                require(not (arrays['terminated'] | arrays['truncated'])[:-1].any() and
                        bool((arrays['terminated'] | arrays['truncated'])[-1]), 'Native episode endpoint differs')
                np.testing.assert_allclose(arrays['rewards'], arrays['speed'] - .1 * np.square(arrays['actions'].astype(float)).sum(-1),
                                           atol=3e-6, rtol=1e-6)
                row = dict(seed=seed, native_steps=n, return_value=float(arrays['rewards'].astype(float).sum()),
                           mean_speed=float(arrays['speed'].astype(float).mean()), inverted_steps=int(arrays['inverted'].sum()),
                           native_termination=bool(arrays['terminated'][-1]), native_truncation=bool(arrays['truncated'][-1]),
                           saturated_action_fraction=float((abs(arrays['actions']) > .99).mean()))
                row['return'] = row['return_value']
                atomic_npz(stem.with_suffix('.npz'), arrays)
                atomic_json(stem.with_suffix('.json'), dict(state='completed', purpose=purpose, gravity=cell['gravity'],
                            condition=cell['condition'], metrics=row, physical_audit=audit,
                            arrays=identity(stem.with_suffix('.npz')), completed_at=time.time()))
                rows.append(row)
                atomic_json(path / 'progress.json', dict(state='running', completed_episodes=len(rows), ledger=ledger))
            except BaseException as error:
                arrays = {key: np.asarray(value) for key, value in trace.items()}
                atomic_npz(stem.with_name(stem.name + '_failed_prefix.npz'), arrays)
                atomic_json(path / 'fatal.json', dict(state='failed', seed=seed, completed_episodes=len(rows),
                            native_steps_in_failed_prefix=len(arrays.get('rewards', [])), ledger=ledger,
                            error=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
                raise
        result = dict(state='completed', purpose=purpose, gravity=cell['gravity'], condition=cell['condition'],
                      episodes=rows, native_steps=sum(r['native_steps'] for r in rows),
                      return_value=float(np.mean([r['return'] for r in rows])),
                      mean_speed=float(np.mean([r['mean_speed'] for r in rows])),
                      inverted_steps=sum(r['inverted_steps'] for r in rows))
        result['return'] = result['return_value']
        atomic_json(path / 'result.json', result)
        return result
    finally:
        env.close()


def train(plan, config, cell):
    directory = HERE / 'solvers' / cell['id']
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / f"{cell['id']}.lock").open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not directory.exists(), 'Solver directory already exists; no overwrite, retry or implicit resume')
        directory.mkdir()
        atomic_json(directory / 'launch.json', dict(state='running', pid=os.getpid(), cell=cell, config=config,
                    inputs=identity(HERE / 'inputs.json'), started_at=time.time(), automatic_retry=False,
                    scientific_lock_owner='local exclusive cell lock', fresh_initialization=True))
        started, ledger, records = time.monotonic(), {}, []
        model, vector = None, None
        phase = 'runtime_imports'
        selected, raw_best = None, None
        try:
            runtime_paths()
            import numpy as np
            import torch
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.vec_env import SubprocVecEnv
            from work2.native.common import require_cuda, seed_all, rng_state, restore_rng, atomic_torch_save
            gpu = require_cuda()
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            # Loading/evaluating the old source precedes the fresh seed13 initialization.
            old = SAC.load(cell['old_teacher'], device='cuda', custom_objects={'seed': None})
            old.policy.eval().requires_grad_(False)
            phase = 'old_teacher_development'
            old_dev = evaluate(old, config, cell, cell['dev_seeds'], directory / 'old_teacher_dev', 'old_teacher_dev', ledger)
            seed_all(config['seed'], threads=config['threads'])
            vector = SubprocVecEnv([functools.partial(make_training_env, cell['gravity'], cell['condition'], i,
                                   str(directory / 'training'), config['episode_steps']) for i in range(config['workers'])],
                                  start_method='spawn')
            model = SAC('MlpPolicy', vector, learning_rate=config['learning_rate'], buffer_size=config['replay_capacity'],
                        learning_starts=config['warmup_steps'], batch_size=config['batch_size'], train_freq=1,
                        gradient_steps=1, gamma=config['gamma'], policy_kwargs={'net_arch': config['network']},
                        seed=config['seed'], device='cuda', verbose=0)
            # SAC's constructor seeds the vector with13; replace only its declared physical reset streams.
            vector.seed(cell['train_seed_base'])
            plateau_anchor, stale_checks, stop_reason = None, 0, None

            def status(state='running'):
                training_steps = int(model.num_timesteps)
                return dict(state=state, pid=os.getpid(), cell=cell['id'], gravity=cell['gravity'], condition=cell['condition'],
                            seed=13, training_transitions=training_steps, gradient_updates=int(model._n_updates),
                            physical_transitions_by_purpose=dict(ledger, training=training_steps),
                            total_physical_transitions=training_steps + sum(ledger.values()),
                            min_steps=config['min_steps'], max_steps=config['max_steps'],
                            elapsed_seconds=time.monotonic() - started,
                            training_transitions_per_wall_second=training_steps / max(time.monotonic() - started, 1e-9),
                            selected_step=None if selected is None else selected['steps'],
                            raw_best_step=None if raw_best is None else raw_best['steps'],
                            qualified_selected=selected is not None, plateau_anchor=plateau_anchor,
                            stale_checks=stale_checks, development=records, updated_at=time.time())

            class TrainingProgress(BaseCallback):
                def __init__(self):
                    super().__init__(); self.last = 0

                def _on_step(self):
                    if self.num_timesteps - self.last >= 5000:
                        self.last = self.num_timesteps
                        require(all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters()), 'Nonfinite SAC parameter')
                        atomic_json(directory / 'progress.json', status())
                    return True

            progress = TrainingProgress()

            def checkpoint(step, dev):
                target = directory / 'checkpoints' / f'step_{step:09d}'
                target.mkdir(parents=True, exist_ok=False)
                temporary = target / 'policy.next.zip'; model.save(temporary); temporary.replace(target / 'policy.zip')
                # Retain every policy/optimizer and RNG/environment snapshot. Replay has
                # one explicitly mutable latest transaction, never an automatic resume.
                replay = directory / 'latest_replay.pkl'
                model.save_replay_buffer(directory / 'latest_replay.next.pkl')
                (directory / 'latest_replay.next.pkl').replace(replay)
                snapshots = vector.env_method('training_snapshot')
                require(sum(s['total_native_steps'] for s in snapshots) == step, 'Vector physical transition counts differ')
                require(int(model._n_updates) == (step - config['warmup_steps']) // config['workers'],
                        'Native SAC update count differs from one update per vector transition')
                atomic_torch_save(target / 'rng_environment.pt', dict(rng=rng_state(), training_steps=step,
                    gradient_updates=int(model._n_updates), workers=snapshots,
                    vector_action_space_rng=copy.deepcopy(vector.action_space.np_random.bit_generator.state),
                    last_observation=copy.deepcopy(model._last_obs), replay_position=int(model.replay_buffer.pos),
                    replay_full=bool(model.replay_buffer.full), replay_capacity=int(model.replay_buffer.buffer_size)))
                row = dict(steps=step, gradient_updates=int(model._n_updates), policy=identity(target / 'policy.zip'),
                           rng_environment=identity(target / 'rng_environment.pt'), development=dev,
                           replay_retention='Only latest_replay.pkl remains; earlier policy/RNG checkpoints remain immutable',
                           replay_sha256_at_checkpoint=digest(replay), replay_bytes_at_checkpoint=replay.stat().st_size)
                atomic_json(target / 'checkpoint.json', row)
                atomic_json(directory / 'latest_transaction.json', dict(state='complete', steps=step,
                            policy=row['policy'], rng_environment=row['rng_environment'], replay=identity(replay),
                            automatic_resume=False, environment_restore_required=True))
                return row

            for target_step in range(config['eval_interval'], config['max_steps'] + 1, config['eval_interval']):
                phase = 'training'
                # Each segment completes its replay insertion AND gradient update
                # before DEV/checkpointing; no callback-mid-transition checkpoint.
                model.learn(config['eval_interval'], callback=progress, reset_num_timesteps=False, progress_bar=False)
                require(model.num_timesteps == target_step, 'Training segment budget differs')
                phase = 'development'
                preserved_rng = rng_state()
                try:
                    dev = evaluate(model, config, cell, cell['dev_seeds'], directory / 'development' / f'step_{target_step:09d}',
                                   'development', ledger)
                finally:
                    restore_rng(preserved_rng)
                gate = quality(dev, old_dev, config, cell)
                dev.update(steps=target_step, quality=gate, qualified=gate['qualified'])
                phase = 'checkpoint'
                check = checkpoint(target_step, dev)
                candidate = dict(steps=target_step, return_value=dev['return'], policy=check['policy'],
                                 checkpoint=str(directory / 'checkpoints' / f'step_{target_step:09d}' / 'checkpoint.json'),
                                 development=dev)
                if raw_best is None or dev['return'] > raw_best['return_value']:
                    raw_best = candidate; atomic_json(directory / 'raw_best.json', raw_best)
                if gate['qualified'] and (selected is None or dev['return'] > selected['return_value']):
                    selected = candidate; atomic_json(directory / 'selected_development.json', selected)
                if target_step >= config['min_steps'] and selected is not None:
                    best = selected['return_value']
                    if plateau_anchor is None or best >= plateau_anchor + config['plateau_relative_improvement'] * max(abs(plateau_anchor), 1.):
                        plateau_anchor, stale_checks = best, 0
                    else:
                        stale_checks += 1
                records.append(dict(steps=target_step, mean_return=dev['return'], qualified=gate['qualified'],
                                    raw_best_return=raw_best['return_value'],
                                    selected_return=None if selected is None else selected['return_value'],
                                    plateau_anchor=plateau_anchor, stale_checks=stale_checks,
                                    receipt=str(directory / 'development' / f'step_{target_step:09d}' / 'result.json')))
                atomic_json(directory / 'progress.json', status())
                print(json.dumps(dict(cell=cell['id'], steps=target_step, dev_return=dev['return'],
                                      qualified=gate['qualified'], stale_checks=stale_checks)), flush=True)
                if selected is not None and stale_checks >= config['plateau_patience']:
                    stop_reason = 'qualified_development_plateau'; break
            if stop_reason is None:
                stop_reason = 'budget_cap_qualified' if selected is not None else 'budget_cap_unqualified'
            require(raw_best is not None, 'No development checkpoint retained')
            chosen = selected if selected is not None else raw_best
            selection_basis = 'highest_return_qualified_DEV' if selected is not None else 'unqualified_raw_best_for_diagnosis_only'
            final_checkpoint = directory / 'checkpoints' / f'step_{int(model.num_timesteps):09d}' / 'policy.zip'
            shutil.copyfile(final_checkpoint, directory / 'final.zip')
            shutil.copyfile(chosen['policy']['path'], directory / 'selected.zip')
            selection = dict(state='frozen_before_qualification', selected_step=chosen['steps'],
                             selected_policy=identity(directory / 'selected.zip'), selection_basis=selection_basis,
                             development_qualified=selected is not None, stop_reason=stop_reason,
                             training_steps=int(model.num_timesteps), raw_best=raw_best, selected=chosen,
                             frozen_at=time.time())
            atomic_json(directory / 'selection.json', selection)
            phase = 'independent_qualification'
            qualified_model = SAC.load(directory / 'selected.zip', device='cuda', custom_objects={'seed': None})
            qualified_model.policy.eval().requires_grad_(False)
            old_qual = evaluate(old, config, cell, cell['qualification_seeds'], directory / 'old_teacher_qualification',
                                'old_teacher_qualification', ledger)
            new_qual = evaluate(qualified_model, config, cell, cell['qualification_seeds'], directory / 'qualification',
                                'qualification', ledger)
            for episode_index in range(config['qualification_episodes']):
                with np.load(directory / 'old_teacher_qualification' / f'episode_{episode_index:02d}.npz', allow_pickle=False) as old_arrays, \
                     np.load(directory / 'qualification' / f'episode_{episode_index:02d}.npz', allow_pickle=False) as new_arrays:
                    np.testing.assert_array_equal(old_arrays['observations'][0], new_arrays['observations'][0])
            qualification = quality(new_qual, old_qual, config, cell, paired=True)
            qualification.update(development_qualified=selected is not None)
            qualification['qualified'] = qualification['qualified'] and selected is not None
            qualification.update(episodes=new_qual['episodes'], return_value=new_qual['return'],
                                 mean_speed=new_qual['mean_speed'], inverted_steps=new_qual['inverted_steps'],
                                 selected_before_any_qualification=True, no_qualification_tuning=True,
                                 all_paired_initial_observations_bitwise_equal=True)
            qualification['return'] = qualification['return_value']
            atomic_json(directory / 'qualification.json', qualification)
            verify_files(plan['frozen_files']); verify_files([cell['old_teacher_identity'], selection['selected_policy']])
            result = dict(state='completed', id=cell['id'], gravity=cell['gravity'], condition=cell['condition'],
                          seed=13, fresh_initialization=True, training_steps=int(model.num_timesteps),
                          gradient_updates=int(model._n_updates), selected_step=chosen['steps'],
                          selected_policy=str(directory / 'selected.zip'), selected_policy_identity=identity(directory / 'selected.zip'),
                          final_policy_identity=identity(directory / 'final.zip'), selection_basis=selection_basis,
                          qualification=qualification, qualification_receipt=str(directory / 'qualification.json'),
                          qualification_receipt_identity=identity(directory / 'qualification.json'),
                          physical_transitions_by_purpose=dict(ledger, training=int(model.num_timesteps)),
                          total_physical_transitions=int(model.num_timesteps) + sum(ledger.values()),
                          stop_reason=stop_reason, converged_claim=False,
                          old_teacher_identity=cell['old_teacher_identity'], config=config,
                          inputs=identity(HERE / 'inputs.json'), gpu=gpu, elapsed_seconds=time.monotonic() - started,
                          automatic_retry=False, checkpoint_source='DEV only; qualification never selects or retrains',
                          completed_at=time.time())
            atomic_json(directory / 'result.json', result)
            atomic_json(directory / 'progress.json', dict(status('completed'), stop_reason=stop_reason,
                        qualification_passed=qualification['qualified']))
            print(json.dumps(dict(cell=cell['id'], state='completed', qualified=qualification['qualified'],
                                  selected_step=chosen['steps'], stop_reason=stop_reason,
                                  qualification_return=new_qual['return'])), flush=True)
            return result
        except BaseException as error:
            # The last completed transaction remains intact; any current failure
            # state is saved separately and is never treated as resumable success.
            diagnostics = {}
            if model is not None:
                diagnostics.update(training_transitions=int(model.num_timesteps), gradient_updates=int(model._n_updates))
                try:
                    model.save(directory / 'failed_policy.zip')
                    model.save_replay_buffer(directory / 'failed_replay.pkl')
                    from work2.native.common import atomic_torch_save, rng_state
                    atomic_torch_save(directory / 'failed_rng.pt', dict(rng=rng_state(), steps=int(model.num_timesteps)))
                except BaseException as save_error:
                    diagnostics['failure_snapshot_error'] = repr(save_error)
            atomic_json(directory / 'fatal.json', dict(state='failed', cell=cell['id'], phase=phase,
                        physical_transitions_by_purpose=ledger, **diagnostics, error=repr(error),
                        traceback=traceback.format_exc(), automatic_retry=False, failed_at=time.time()))
            raise
        finally:
            if vector is not None:
                vector.close()
