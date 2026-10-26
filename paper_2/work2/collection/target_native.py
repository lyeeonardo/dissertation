"""Native target-root collection with exact masks and transition accounting."""
import json


def collect_parent(env, actor, declared, progress, model_queries):
    import numpy as np

    row = dict(declared)
    kind = row['kind']
    generator = np.random.default_rng(np.random.SeedSequence(row['action_noise_seed']))
    observation, _ = env.reset(seed=row['reset_seed'])
    snapshot = env.snapshot()
    physics = env.audit_condition()
    if physics['gravity_scale'] != 5 or physics['condition'] != 4:
        raise ValueError('Collection must use actual 5x gravity in condition100')
    stores = dict(observations=[observation.copy()], integration=[snapshot['integration']],
                  actions=[], rewards=[], terminated=[], truncated=[], proposal_eligible=[],
                  intervention=[], speed=[], inverted=[], gravity_scale=[], gravity_vector=[], simulation_time=[])
    metadata = dict(row, rng=snapshot['rng'], physics=physics, initial_elapsed=0,
                    requested_length=500, source_condition=4,
                    proposal_rule=('always false for random actions' if kind == 'random'
                                   else 'policy-defined: noninverted and native float64 speed >= 0.5'),
                    native_time_limit_steps=500, collection_clock_limit_seconds=25.)

    def arrays():
        booleans = {'terminated', 'truncated', 'proposal_eligible', 'intervention', 'inverted'}
        doubles = {'integration', 'gravity_vector', 'simulation_time'}
        return {key: np.asarray(value, dtype=bool if key in booleans else np.float64 if key in doubles else np.float32)
                for key, value in stores.items()}

    uncommitted = None
    try:
        for step in range(500):
            if kind == 'random':
                action = generator.uniform(-1., 1., size=6).astype(np.float32)
            else:
                model_queries['policy_predict_calls'][kind] += 1
                action, _ = actor.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
            if action.shape != (6,) or not np.isfinite(action).all() or (np.abs(action) > 1.).any():
                raise ValueError('Collector action violates the native finite bounded action contract')
            previous = observation.copy()
            observation, reward, terminal, truncated, info = env.step(action)
            progress['physical_transitions'] += 1
            uncommitted = dict(observations=np.stack([previous, observation]), actions=action[None],
                               rewards=np.asarray([reward]), terminated=np.asarray([terminal]),
                               truncated=np.asarray([truncated]), step=np.asarray([step]))
            diagnostic = env.diagnostics()
            speed = float(info['x_velocity'])
            integration = env.snapshot()['integration']
            if (observation.shape != (17,) or not np.isfinite(observation).all()
                    or not np.isfinite([reward, speed]).all() or integration.shape != (91,)
                    or not np.isfinite(integration).all()):
                raise ValueError('Nonfinite or malformed native collection transition')
            for key, value in dict(observations=observation.copy(), integration=integration,
                    actions=action.copy(), rewards=reward, terminated=terminal, truncated=truncated,
                    proposal_eligible=kind != 'random' and not diagnostic['inverted'] and speed >= .5,
                    intervention=kind == 'random', speed=speed, inverted=diagnostic['inverted'],
                    gravity_scale=5., gravity_vector=env.unwrapped.model.opt.gravity.copy(),
                    simulation_time=float(env.unwrapped.data.time)).items():
                stores[key].append(value)
            uncommitted = None
            if terminal or truncated:
                break
    except BaseException as error:
        error.collection_partial = (arrays(), dict(metadata, length=len(stores['actions']), complete=False))
        error.collection_uncommitted = uncommitted
        raise
    n = len(stores['actions'])
    metadata.update(length=n, full_requested_duration=n == 500,
                    collection_cutoff=not (stores['terminated'][-1] or stores['truncated'][-1]),
                    native_termination=bool(any(stores['terminated'])), native_truncation=bool(any(stores['truncated'])),
                    inverted_steps=int(sum(stores['inverted'])), proposal_eligible=int(sum(stores['proposal_eligible'])))
    return arrays(), metadata


def write_parent(handle, name, arrays, metadata):
    # Exclusive groups: native datasets/attributes, without native overwrite semantics.
    if name in handle:
        raise RuntimeError('Existing parent cannot be overwritten')
    group = handle.create_group(name)
    for key, value in arrays.items():
        group.create_dataset(key, data=value, compression='lzf')
    for key, value in metadata.items():
        group.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
    group.attrs['complete'] = True
    handle.flush()
