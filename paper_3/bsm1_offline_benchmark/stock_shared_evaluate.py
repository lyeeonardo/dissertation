"""Whole-episode matched evaluation for the frozen stock/shared-data campaign."""

from __future__ import annotations

import copy

import pickle

import random

import time

import traceback

from pathlib import Path

import numpy as np

from .evaluate import _save_arrays, _synchronize

from .stock_shared_data import public55_from_public57, write

PUBLIC_FEEDBACK = ('public_pi', 'expert_pi')

def controller_action(controller, method, public57, previous_action, truth_supplier):
    """Only the declared oracle can invoke the private input supplier."""
    public57 = np.asarray(public57, dtype=np.float64)
    previous_action = np.asarray(previous_action, dtype=np.float64)
    if public57.shape != (57,) or previous_action.shape != (2,) or not np.isfinite(public57).all() or not np.isfinite(previous_action).all():
        raise ValueError('Finite public57 and previous executed action2 required')
    if method in PUBLIC_FEEDBACK:
        action = controller.act(public57.copy(), previous_action.copy())
    elif method.startswith('oracle'):
        truth = np.asarray(truth_supplier(), dtype=np.float64)
        if truth.shape != (159,) or not np.isfinite(truth).all():
            raise ValueError('Oracle requires finite current159 only')
        action = controller.act(public55_from_public57(public57), previous_action.copy(),
                                current_truth=truth.copy())
    else:
        action = controller.act(public55_from_public57(public57), previous_action.copy())
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,) or not np.isfinite(action).all():
        raise FloatingPointError('Invalid controller action')
    if np.any(action < -1e-6) or np.any(action > 1+1e-6):
        raise ValueError('Out-of-box normalized action')
    return np.clip(action, 0., 1.)

def controller_runtime(controller):
    runtime = copy.deepcopy(controller.runtime_state()) if hasattr(controller, 'runtime_state') else dict(
        policy=copy.deepcopy(getattr(controller, 'policy', None)), step=getattr(controller, 'step', None),
        previous_plan=copy.deepcopy(getattr(controller, 'previous_plan', None)))
    if hasattr(controller, 'generator'):
        runtime['generator_state'] = controller.generator.get_state().cpu().clone()
    if hasattr(controller, 'last_diagnostics'):
        runtime['last_diagnostics'] = copy.deepcopy(controller.last_diagnostics)
    return runtime

def execute(env, controller, job, folder, device='cpu', checkpoint_interval=96, setup_seconds=0.):
    if env.decisions != 0 or checkpoint_interval < 1:
        raise ValueError('Fresh full-episode environment and positive checkpoint interval required')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    arrays = {key: [] for key in ('observations','privileged','actions','costs','components',
                                  'decision_seconds','time_end_days')}
    arrays['observations'].append(env.observation()); arrays['privileged'].append(env.privileged())
    diagnostics = []
    def progress(status):
        write(folder / 'progress.json', dict(status=status, job_id=job['job_id'],
              completed_decisions=env.decisions,total_decisions=env.episode_steps,updated_at=time.time()))
    def checkpoint():
        runtime = controller_runtime(controller)
        payload = dict(environment=env.snapshot(), controller=runtime, arrays=arrays,
                       job=job, completed_decisions=env.decisions, automatic_resume=False,
                       numpy_rng=np.random.get_state(), python_rng=random.getstate())
        if hasattr(controller, 'generator') or str(device).startswith('cuda'):
            import torch
            payload['torch_rng'] = torch.get_rng_state()
            if str(device).startswith('cuda'): payload['cuda_rng'] = torch.cuda.get_rng_state_all()
        path=folder/'checkpoint.pkl'; temporary=path.with_suffix('.tmp.pkl')
        with temporary.open('wb') as stream: pickle.dump(payload,stream,protocol=5)
        temporary.replace(path)
    progress('running')
    try:
        for step in range(env.decisions,env.episode_steps):
            _synchronize(device); began=time.perf_counter()
            action=controller_action(controller,job['method'],env.observation(),env.previous_action,env.privileged)
            _synchronize(device); decision=time.perf_counter()-began
            env.step(action); transition=env.last_transition
            if hasattr(controller,'observe'):
                began=time.perf_counter()
                controller.observe(public55_from_public57(transition['next_observation']),
                                   transition['action'],float(transition['cost']))
                _synchronize(device); decision+=time.perf_counter()-began
            arrays['observations'].append(transition['next_observation'])
            arrays['privileged'].append(transition['next_privileged'])
            for key,source in [('actions','action'),('costs','cost'),('components','components')]:
                arrays[key].append(transition[source])
            arrays['decision_seconds'].append(decision); arrays['time_end_days'].append(env.minute/1440.)
            if hasattr(controller,'last_diagnostics'): diagnostics.append(copy.deepcopy(controller.last_diagnostics))
            if env.decisions % checkpoint_interval == 0: checkpoint()
            # A completed physical decision is scientific progress, including expensive belief updates.
            progress('running')
        summary=env.summary(); costs=np.asarray(arrays['costs']); latency=np.asarray(arrays['decision_seconds'])
        if not np.isclose(costs.mean(),summary['economic_cost_per_day'],rtol=1e-8,atol=1e-5):
            raise ArithmeticError('Transition costs disagree with minute-integrated physical accounting')
        result=dict(**job,status='complete',whole_episode=summary,primary=summary,
                    completed_decisions=env.decisions,primary_start_step=0,
                    timing=dict(mean_seconds=float(latency.mean()),p95_seconds=float(np.quantile(latency,.95)),
                                maximum_seconds=float(latency.max()),total_seconds=float(latency.sum()),setup_seconds=float(setup_seconds),
                                scope='Action selection plus real-observation belief update, synchronized'),
                    wall_seconds=time.perf_counter()-start,completed_at=time.time(),
                    observation_scope='Learned public55; private current159 only for explicitly named oracle',
                    episode_days=float(env.episode_days), decision_minutes=env.decision_minutes,
                    task='Archived PaperPublic57Env, whole-episode EQ+0.3OCI, no added air constraint')
        _save_arrays(folder/'arrays.npz',arrays); checkpoint()
        if diagnostics: write(folder/'controller_diagnostics.json',diagnostics)
        write(folder/'result.json',result); progress('complete'); return result
    except BaseException as error:
        _save_arrays(folder/'partial_arrays.npz',arrays)
        write(folder/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),automatic_retry=False))
        progress('failed'); raise
