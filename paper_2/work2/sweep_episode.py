"""Native public-history control, isolated physical banks, complete failure prefixes.

Only oracle receives the current condition. Schedules, snapshots and physical
parameters are evaluator diagnostics and never enter a hidden controller.
"""
from __future__ import annotations

import time
import numpy as np
import torch

VARIANTS = ('upright_map_replay',)
EVIDENCE_SOURCE_CODES = {'upright_map_replay': 1}
ORIENTATION_COEFFICIENT = 4.0567890408452785
from work2.native.control import PublicHistory
from work2.native.evaluation import _synchronize, decision_seed

SHARED = (*VARIANTS, 'oracle')
DETAILS = ('expected_reward', 'selection_score', 'selected_index', 'original_index',
           'sampled_condition', 'current_condition', 'proposal_draws',
           'proposal_conditions', 'selected_proposal_condition', 'native_selected_index',
           'kappa', 'orientation_reward_std')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def numpy(value):
    return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()


def arrays_from(trace):
    return {key: np.asarray(values) for key, values in trace.items() if values}


def validate_case_arrays(case, arrays, method, plan, role='control'):
    """Checks execution arithmetic, not scientific success; adverse outcomes stay."""
    n = len(arrays['rewards'])
    require(0 < n <= case['steps'], 'Invalid observed native length')
    for key, value in arrays.items():
        require(np.isfinite(value).all(), f'Nonfinite {key}')
        expected = n + 1 if key in ('observations', 'beliefs', 'integration_state') else n
        require(len(value) == expected, f'Incomplete {key}')
    require(arrays['observations'].shape == (n + 1, 17) and arrays['actions'].shape == (n, 6),
            'Native observation/action interface differs')
    require(np.all(abs(arrays['actions']) <= 1.000001), 'Action bounds differ')
    require(arrays['integration_state'].shape == (n + 1, 91) and
            arrays['integration_state'].dtype == np.float64, 'Full native integration states required')
    require(arrays['planning_mask'].all(), 'Every action requires a planning call')
    require(np.isin(arrays['terminated'], (False, True)).all() and np.isin(arrays['truncated'], (False, True)).all(), 'Invalid terminal flags')
    done = arrays['terminated'].astype(bool) | arrays['truncated'].astype(bool)
    require(not done[:-1].any() and bool(done[-1]), 'Native terminal/time-limit flag missing')
    # Early termination OR truncation is a valid retained outcome, never a retry.
    truth = np.full(n, case['initial_condition'], dtype=np.int64)
    for event in case['events']:
        truth[event['step']:] = event['condition']
    np.testing.assert_array_equal(arrays['conditions'], truth)
    high = float(plan['actual_gravity'])
    require(plan['high_gravity'] == 3 and plan['actual_gravity'] in (1., 1.5, 2., 2.5, 3., 4., 5.), 'Frozen model bank1/3; declared sweep physics required')
    np.testing.assert_array_equal(arrays['gravity_scale'], np.where(truth < 4, 1., float(high)))
    np.testing.assert_allclose(arrays['gravity_vector'],
                              np.stack([np.zeros(n), np.zeros(n), -9.81 * arrays['gravity_scale']], axis=1),
                              atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays['sliding_friction_scale'], np.where((truth >> 1) & 1, .5, 1.), atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays['joint_range_scale'], np.where(truth & 1, .8, 1.), atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays['simulation_time'], (np.arange(n) + 1) * .05, atol=1e-8, rtol=0)
    np.testing.assert_allclose(arrays['integration_state'][:, 0], np.arange(n + 1) * .05, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(arrays['observations'][:, :8], arrays['integration_state'][:, 2:10].astype(np.float32))
    np.testing.assert_array_equal(arrays['observations'][:, 8:], arrays['integration_state'][:, 10:19].astype(np.float32))
    speed = np.diff(arrays['integration_state'][:, 1]) / .05
    np.testing.assert_allclose(arrays['speed'], speed, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(arrays['rewards'], speed - .1 * (arrays['actions'].astype(float) ** 2).sum(-1), atol=2e-6, rtol=1e-7)
    np.testing.assert_array_equal(arrays['batch_size'], np.ones(n, dtype=int))
    expected_seeds = (case['sampling_seed'] + np.arange(n) if method in SHARED else
                      np.asarray([decision_seed(case['sampling_seed'], i) for i in range(n)], dtype=np.int64))
    np.testing.assert_array_equal(arrays['query_seed'], expected_seeds)
    if method in VARIANTS:
        for key, shape in [('beliefs', (n + 1, 8)), ('priors', (n, 8)), ('decision_prior', (n, 8)),
                           ('log_likelihood', (n, 8)), ('denoising_errors', (n, 8)),
                           ('banks', (n, 3, 4, 6)), ('selection_score', (n, 3)),
                           ('expected_reward', (n, 3)), ('predicted_rewards', (n, 3, 8, 4)),
                           ('prediction_valid', (n, 3, 8, 4))]:
            require(arrays[key].shape == shape, f'Incomplete H4 {key}')
        beliefs, priors = arrays['beliefs'].astype(float), arrays['priors'].astype(float)
        require((beliefs >= 0).all() and np.allclose(beliefs.sum(-1), 1, atol=2e-6), 'Invalid posterior')
        np.testing.assert_allclose(priors, .94 * beliefs[:-1] + .06 * (1 - beliefs[:-1]) / 7, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(arrays['decision_prior'], arrays['priors'])
        for key in ('evidence_source_code', 'evidence_temperature', 'evidence_query_seed'):
            require(key in arrays and arrays[key].shape == (n,), f'Missing evidence provenance: {key}')
        np.testing.assert_array_equal(arrays['evidence_source_code'], np.full(n, EVIDENCE_SOURCE_CODES[method]))
        np.testing.assert_array_equal(arrays['evidence_temperature'], np.full(n, plan['temperature']))
        np.testing.assert_array_equal(arrays['evidence_query_seed'], arrays['query_seed'])
        np.testing.assert_allclose(arrays['log_likelihood'], -arrays['denoising_errors'] / plan['temperature'], atol=2e-5, rtol=2e-6)
        logits = np.log(priors) + arrays['log_likelihood']
        posterior = np.exp(logits - logits.max(-1, keepdims=True)); posterior /= posterior.sum(-1, keepdims=True)
        np.testing.assert_allclose(beliefs[1:], posterior, atol=2e-5, rtol=0)
        selected = arrays['selected_index'].astype(int)
        np.testing.assert_array_equal(selected, arrays['selection_score'].argmax(-1))
        np.testing.assert_array_equal(arrays['actions'], arrays['banks'][np.arange(n), selected, 0])
        require(arrays['predicted_observations'].shape == (n,3,8,4,17), 'Full scoring forecasts missing')
        if method.startswith('upright_'):
            for key in ('orientation_cost','orientation_penalty'):
                require(arrays[key].shape == (n,3), 'Orientation score component missing')
            cost = ((1-np.cos(arrays['predicted_observations'][...,1].astype(float))) *
                    arrays['prediction_valid'] * .99**np.arange(4) * priors[:,None,:,None]).sum(axis=(-1,-2))
            np.testing.assert_allclose(arrays['orientation_cost'],cost,atol=2e-5,rtol=2e-6)
            kappa = 0. if method == 'upright_map_replay_zero' else 1.
            np.testing.assert_array_equal(arrays['kappa'],np.full(n,kappa))
            np.testing.assert_array_equal(arrays['orientation_reward_std'],np.full(n,ORIENTATION_COEFFICIENT))
            np.testing.assert_array_equal(arrays['native_selected_index'],arrays['expected_reward'].argmax(-1))
            penalty = arrays['orientation_cost'] * np.float32(ORIENTATION_COEFFICIENT * kappa)
            np.testing.assert_array_equal(arrays['orientation_penalty'],penalty)
            np.testing.assert_array_equal(arrays['selection_score'],arrays['expected_reward']-arrays['orientation_penalty'])
        else:
            np.testing.assert_array_equal(arrays['selection_score'], arrays['expected_reward'])
        require(arrays['proposal_conditions'].shape == (n, 3) and
                arrays['selected_proposal_condition'].shape == (n,), 'Proposal condition trace missing')
        conditions = arrays['proposal_conditions']
        require(np.issubdtype(conditions.dtype, np.integer) and
                ((conditions >= 0) & (conditions < 8)).all(), 'Invalid proposal hypothesis labels')
        np.testing.assert_array_equal(arrays['sampled_condition'], conditions[:, 0])
        np.testing.assert_array_equal(arrays['selected_proposal_condition'], conditions[np.arange(n), selected])
        if method in VARIANTS:
            np.testing.assert_array_equal(conditions, np.repeat(conditions[:, :1], 3, axis=1))
        if method == 'map_replay_evidence' or method.startswith('upright_'):
            np.testing.assert_array_equal(arrays['priors'][np.arange(n), conditions[:, 0]], arrays['priors'].max(-1))
        require('truth_weights' not in arrays and 'current_condition' not in arrays,
                'Hidden controller must not record a privileged input')
        probability = arrays['prediction_termination_probability']
        require(probability.shape == (n, 3, 8, 4) and ((probability >= 0) & (probability <= 1)).all(),
                'Invalid forecast termination probabilities')
        require(np.isin(arrays['prediction_valid'], (False, True)).all() and
                arrays['prediction_valid'][..., 0].all(), 'Invalid forecast validity mask')
        returns = (arrays['predicted_rewards'].astype(float) * arrays['prediction_valid'] *
                   .99 ** np.arange(4)).sum(-1)
        expected = (returns * priors[:, None]).sum(-1)
        np.testing.assert_allclose(arrays['expected_reward'], expected, atol=2e-5, rtol=2e-6)
        require(np.all(arrays['update_mask']), 'One real evidence update required per executed action')
    elif method == 'oracle':
        np.testing.assert_array_equal(arrays['current_condition'], truth)
        np.testing.assert_array_equal(arrays['truth_weights'], np.eye(8, dtype=np.float32)[truth])
        require(np.all(arrays['proposal_draws'] == 1) and not arrays['update_mask'].any(), 'Native H8 K1 oracle contract differs')
    else:
        raise ValueError('Only declared H4 variants and direct oracle are supported')
    return dict(state='passed', native_steps=n)


@torch.inference_mode()
def run_episode(policy, case, method, plan, *, env_factory):
    trace, env, stage, uncommitted = {}, None, 'before_reset', None

    def add(key, value):
        trace.setdefault(key, []).append(numpy(value))

    try:
        require(plan['env_id'] == 'HalfCheetah-v5' and plan['high_gravity'] == 3 and plan['actual_gravity'] in (1., 1.5, 2., 2.5, 3., 4., 5.), 'Native bank1/3 with declared sweep physics required')
        require(method == 'upright_map_replay', 'Only unchanged final GenAIF is declared')
        env = env_factory(condition=case['initial_condition'], max_episode_steps=case['steps'])
        observation, _ = env.reset(seed=case['reset_seed'])
        initial_physics = env.audit_condition()
        snapshot = env.snapshot()
        initial_physics.update(initial_rng=snapshot['rng'], initial_elapsed=snapshot['elapsed'])
        history = PublicHistory(observation, policy.history_length)
        add('observations', observation); add('integration_state', snapshot['integration'])
        if method in VARIANTS:
            add('beliefs', policy.belief)
        schedule = {e['step']: e['condition'] for e in case['events']}
        for step in range(case['steps']):
            if step in schedule:
                stage = 'physical_switch'
                env.apply_condition(schedule[step]); env.audit_condition()
            stage = 'decision'
            _synchronize(policy.device); started = time.perf_counter()
            public = history.tensors(policy.device)
            if method == 'oracle':
                action, details = policy.act(public, step, oracle_condition=env.condition.index)
            else:
                action, details = policy.act(public, step)
            _synchronize(policy.device); decision_s = time.perf_counter() - started
            action = numpy(action).astype(np.float32)
            require(action.shape == (6,) and np.isfinite(action).all() and np.max(abs(action)) <= 1.000001,
                    'Invalid bounded action')
            stage = 'native_transition'; started = time.perf_counter()
            previous_observation = observation.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            simulator_s = time.perf_counter() - started
            # Retain every returned transition even if snapshot/diagnostics fail next.
            uncommitted = dict(observations=np.stack([previous_observation, observation]), actions=action[None].copy(),
                               rewards=np.asarray([reward]), terminated=np.asarray([terminated]), truncated=np.asarray([truncated]),
                               step=np.asarray([step]), conditions=np.asarray([env.condition.index]))
            stage = 'recording_returned_transition'
            friction_id = int(np.flatnonzero(env.nominal['friction'][:, 0])[0])
            joint_id = int(env.joint_ids[0])
            actual = env.unwrapped.model
            row = dict(observations=observation, integration_state=env.snapshot()['integration'], actions=action, rewards=reward,
                       speed=float(info['x_velocity']), inverted=env.diagnostics()['inverted'], conditions=env.condition.index,
                       gravity_scale=float(actual.opt.gravity[2] / env.nominal['gravity'][2]), gravity_vector=actual.opt.gravity.copy(),
                       sliding_friction_scale=float(actual.geom_friction[friction_id, 0] / env.nominal['friction'][friction_id, 0]),
                       joint_range_scale=float(np.ptp(actual.jnt_range[joint_id]) / np.ptp(env.nominal['joint_range'][joint_id])),
                       simulation_time=float(env.unwrapped.data.time), terminated=bool(terminated), truncated=bool(truncated),
                       decision_s=decision_s, simulator_s=simulator_s, batch_size=1, planning_mask=True,
                       query_seed=(case['sampling_seed'] + step if method in SHARED else decision_seed(case['sampling_seed'], step)))
            for key in DETAILS:
                if key in details and details[key] is not None:
                    row[key] = details[key]
            if method in VARIANTS:
                row.update(banks=details['candidate_actions'], decision_prior=details['prior'],
                           predicted_rewards=details['rewards'], prediction_valid=details['valid'],
                           prediction_termination_probability=details['termination_probability'],
                           predicted_observations=details['observations'])
                if method.startswith('upright_'):
                    row.update(orientation_cost=details['orientation_cost'], orientation_penalty=details['orientation_penalty'])
                require(int(details['query_seed']) == row['query_seed'], 'H4 query seed differs')
            if method == 'oracle':
                row['truth_weights'] = np.eye(8, dtype=np.float32)[env.condition.index]
            for key, value in row.items():
                add(key, value)
            uncommitted = None
            stage = 'real_evidence_update'; started = time.perf_counter()
            history.append(action, observation, reward)
            inference = policy.observe(action, observation, history.tensors(policy.device))
            _synchronize(policy.device)
            add('update_s', time.perf_counter() - started); add('update_mask', method in VARIANTS)
            if method in VARIANTS:
                add('beliefs', inference['belief']); add('priors', inference['prior'])
                add('log_likelihood', inference['log_likelihood']); add('denoising_errors', inference['denoising_errors'])
                for key in ('evidence_source_code', 'evidence_temperature', 'evidence_query_seed'):
                    if key in inference: add(key, inference[key])
            if terminated or truncated:
                break
        stage = 'post_episode_validation'
        env.audit_condition()
        arrays = arrays_from(trace)
        validation = validate_case_arrays(case, arrays, method, plan)
        return arrays, dict(validation=validation, initial_physics=initial_physics)
    except BaseException as error:
        error.repair_partial = arrays_from(trace)
        error.repair_uncommitted = uncommitted
        error.repair_phase = stage
        error.repair_failure_state = (dict(wrapper_elapsed=int(env.elapsed), simulation_time=float(env.unwrapped.data.time),
            failing_native_call_may_have_advanced_physics=stage == 'native_transition') if env is not None else None)
        raise
    finally:
        if env is not None:
            env.close()
