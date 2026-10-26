"""Freeze one existing-physics dataset for stock public baselines and joint models.

No simulation, model inference or model fitting. Historical source arrays remain
read-only. All causal windows stop at episode/recording boundaries.
"""

from __future__ import annotations

import copy

import json

from pathlib import Path

import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

SOURCE = PROJECT / 'outputs/collection/shared/data'

DEFAULT_ROOT = PROJECT / 'outputs/prepared'

HORIZON = 30

HISTORIES = (1, 8, 24)

EPISODE_STEPS = 1344

BEHAVIORS = ('nominal_pi', 'development_expert', 'perturbed_expert', 'boundary_holds')

CALIBRATION_PARENTS = ('validation_p003', 'validation_p004', 'validation_p005')

PUBLIC_INDICES = np.array([13*r+j for r in range(5) for j in range(2, 7)]
    + list(range(135, 145)) + [65, 75, 95, 105, 115, 63] + list(range(145, 159)), dtype=np.int64)

PUBLIC57_INDICES = np.concatenate((PUBLIC_INDICES[:41], [59, 21], PUBLIC_INDICES[41:]))

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)

def public55_from_privileged(value):
    value = np.asarray(value)
    if value.shape[-1:] != (159,):
        raise ValueError('Current159 packet required')
    return value[..., PUBLIC_INDICES]

def public55_from_public57(value):
    value = np.asarray(value)
    if value.shape[-1:] != (57,):
        raise ValueError('Public57 packet required')
    return np.concatenate((value[..., :41], value[..., 43:]), axis=-1)

def load_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}

def take(arrays, index):
    return {k: v[index] for k, v in arrays.items()}

def repaired_source_splits(train, validation):
    """Join episode59 before building any histories or boundaries."""
    if set(train) != set(validation):
        raise ValueError('Source TRAIN and VAL fields differ')
    moved = train['episode_id'] == 59
    joined = {k: np.concatenate((train[k][moved], validation[k]), axis=0) for k in train}
    joined = take(joined, np.lexsort((joined['step'], joined['episode_id'])))
    train = take(train, ~moved)
    if set(train['episode_id']) & set(joined['episode_id']):
        raise ValueError('An episode still crosses TRAIN/VAL')
    return train, joined, int(moved.sum())

def sequence_metadata(episode_id, steps, horizon=HORIZON, histories=HISTORIES, episode_steps=EPISODE_STEPS):
    episodes, steps = np.asarray(episode_id), np.asarray(steps)
    n = len(episodes)
    if n == 0 or steps.shape != (n,) or np.any(steps < 0) or np.any(steps >= episode_steps):
        raise ValueError('Nonempty aligned valid episode steps required')
    connected = (episodes[1:] == episodes[:-1]) & (steps[1:] == steps[:-1] + 1)
    starts = np.r_[0, np.flatnonzero(~connected) + 1]
    ends = np.r_[starts[1:], n]
    segment_start = np.empty(n, dtype=np.int64); segment_end = np.empty(n, dtype=np.int64)
    for start, end in zip(starts, ends):
        segment_start[start:end] = start; segment_end[start:end] = end
    index = np.arange(n)
    record_end = index + 1 == segment_end
    time_limit = steps == episode_steps - 1
    future_valid = index + horizon <= segment_end
    previous_valid = index > segment_start
    metadata = dict(record_boundary=record_end, terminated=record_end.copy(),
        physical_terminal=np.zeros(n, dtype=bool), time_limit=time_limit,
        split_boundary=record_end & ~time_limit, bootstrap_mask=np.ones(n, dtype=np.float32),
        finite_task_bootstrap_mask=(~time_limit).astype(np.float32),
        previous_cost_valid=previous_valid, future_valid_h30=future_valid,
        han_valid_h30=future_valid & previous_valid,
        future_starts_h30=np.flatnonzero(future_valid), han_starts_h30=np.flatnonzero(future_valid & previous_valid))
    for h in histories:
        indices = index[:, None] + np.arange(1-h, 1)[None, :]
        valid = indices >= segment_start[:, None]
        metadata[f'history_indices_h{h}'] = np.where(valid, indices, -1).astype(np.int32)
        metadata[f'history_mask_h{h}'] = valid
    return metadata

def causal_history(arrays, rows, history, cost_scale):
    """Right-aligned public root packets and PREVIOUS commands only; zero padding.

    At decision t, neither a_t nor y_(t+1) is available in this history.
    """
    if not np.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError('Positive TRAIN-derived cost_scale required')
    if history not in HISTORIES:
        raise ValueError('Only supported history lengths1/8/24')
    index = arrays[f'history_indices_h{history}'][np.asarray(rows)]
    mask = arrays[f'history_mask_h{history}'][np.asarray(rows)]
    safe = np.maximum(index, 0)
    reward_valid = mask & arrays['previous_cost_valid'][safe]
    previous_rewards = np.where(reward_valid, -arrays['previous_costs'][safe] / cost_scale, 0.)[..., None]
    return dict(observations=np.where(mask[..., None], arrays['observations'][safe], 0.),
        previous_actions=np.where(mask[..., None], arrays['previous_actions'][safe], 0.),
        previous_rewards=previous_rewards, valid=mask.copy(), reward_valid=reward_valid, mask=mask.copy())

def quality_labels(arrays, jobs, episode_steps=EPISODE_STEPS):
    n = len(arrays['costs'])
    quality = np.zeros(n, dtype=np.float32); valid = np.zeros(n, dtype=bool)
    behavior_id = np.full(n, -1, dtype=np.int8); records = []
    job_by_id = {int(j['episode_id']): j for j in jobs}
    for parent in np.unique(arrays['parent_id']):
        if parent < 0:
            continue
        episodes = np.unique(arrays['episode_id'][arrays['parent_id'] == parent])
        group = []
        for episode in episodes:
            rows = np.flatnonzero(arrays['episode_id'] == episode)
            job = job_by_id.get(int(episode))
            if job is None:
                raise ValueError('Supplement episode lacks its frozen behavior identity')
            behavior_id[rows] = BEHAVIORS.index(job['behavior'])
            if len(rows) == episode_steps and np.array_equal(arrays['step'][rows], np.arange(episode_steps)):
                group.append(dict(episode_id=int(episode), parent_id=int(parent), behavior=job['behavior'],
                    mean_daily_cost=float(np.mean(arrays['costs'][rows], dtype=np.float64)), rows=rows))
        if len(group) != 4 or {g['behavior'] for g in group} != set(BEHAVIORS):
            continue
        # Equal costs get equal average ranks. Rank only within identical forcing.
        costs = np.array([g['mean_daily_cost'] for g in group])
        for g in group:
            rank = np.sum(costs < g['mean_daily_cost']) + .5*(np.sum(costs == g['mean_daily_cost']) - 1)
            q = float(1 - rank / 3)
            quality[g['rows']] = q; valid[g['rows']] = True
            records.append({k: v for k, v in g.items() if k != 'rows'} | dict(quality=q))
    return dict(quality=quality, quality_valid=valid, behavior_id=behavior_id), records

def canonical_arrays(raw, jobs):
    result = {k: v for k, v in raw.items() if k not in ('quality',)}
    for public, private, saved in (('observations', 'privileged', 'public57'),
                                   ('next_observations', 'next_privileged', 'next_public57')):
        result[saved] = raw[public]
        projected = public55_from_public57(raw[public])
        if not np.array_equal(projected, public55_from_privileged(raw[private])):
            raise ValueError('Public55 must exactly match declared current159 projection')
        result[public] = projected
    for key in ('clean_observations', 'next_clean_observations'):
        result[key] = public55_from_public57(raw[key])
    metadata = sequence_metadata(raw['episode_id'], raw['step'])
    result.update(metadata)
    previous = np.zeros(len(raw['costs']), dtype=raw['costs'].dtype)
    valid = metadata['previous_cost_valid']; index = np.flatnonzero(valid)
    previous[index] = raw['costs'][index - 1]
    result['previous_costs'] = previous
    labels, quality_records = quality_labels(raw, jobs)
    result.update(labels)
    connected = np.flatnonzero(metadata['previous_cost_valid'])
    if (not np.array_equal(raw['previous_actions'][connected], raw['actions'][connected-1])
            or not np.array_equal(raw['observations'][connected], raw['next_observations'][connected-1])
            or not np.array_equal(raw['privileged'][connected], raw['next_privileged'][connected-1])):
        raise ValueError('Joined episode action/state alignment is inconsistent')
    return result, quality_records

def freeze_test_parents(existing, seed=17):
    names = ('dry', 'rain', 'storm'); result = []
    for index in range(6):
        rng = np.random.default_rng(np.random.SeedSequence([20261014, seed, 77031, index]))
        result.append(dict(parent_id=f'stock_test_p{index:03d}', parent_number=7000+index,
            split='test', weather=names[index % 3], weather_id=index % 3,
            phase_days=float(rng.uniform(0, 14)), flow_scale=float(rng.uniform(.85, 1.15)),
            carbon_scale=float(rng.uniform(.85, 1.15)), nitrogen_scale=float(rng.uniform(.85, 1.15)),
            wave_amplitude=float(rng.uniform(.02, .08)), wave_phase=float(rng.uniform(0, 2*np.pi)),
            sensor_seed=77031000 + index, initial_state='Han published legacy initial state',
            observations_available=False, physical_outcomes_available=False))
    fields = ('weather', 'phase_days', 'flow_scale', 'carbon_scale', 'nitrogen_scale', 'wave_amplitude', 'wave_phase')
    old = {tuple(p[k] for k in fields) for p in existing}
    if len({p['parent_id'] for p in result}) != 6 or any(tuple(p[k] for k in fields) in old for p in result):
        raise ValueError('TEST forcing must be new and distinct before model fitting')
    return result

def normalization(train):
    result = {}
    for name in ('observations', 'public57', 'privileged', 'actions', 'costs', 'components'):
        value = train[name]
        result[name + '_mean'] = np.asarray(value.mean(axis=0, dtype=np.float64))
        result[name + '_std'] = np.asarray(value.std(axis=0, dtype=np.float64).clip(1e-6))
    result['cost_scale'] = np.asarray(float(np.mean(train['costs'], dtype=np.float64)))
    if not np.isfinite(result['cost_scale']) or result['cost_scale'] <= 0:
        raise ValueError('Economic reward normalization requires positive finite TRAIN mean')
    return result

def save_arrays(path, arrays):
    temporary = Path(path).with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays); temporary.replace(path)

def freeze(root=DEFAULT_ROOT, source=SOURCE):
    root, source = Path(root).resolve(), Path(source).resolve(); output = root / 'data'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Fresh dataset directory required; preserve earlier freeze')
    original = read(source / 'manifest.json')
    if original.get('status') != 'complete' or original['split_transitions'] != dict(train=187520, validation=42256):
        raise ValueError('Require the completed187520/42256 shared Han source bank')
    if original['return_flow'] != 18846 or original['waste_flow'] != 385 or original['sensor_noise'] is not False:
        raise ValueError('Source physical protocol mismatch')
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'progress.json', dict(status='running', phase='loading', updated_at=time.time()))
    train, val = load_arrays(source / 'train.npz'), load_arrays(source / 'validation.npz')
    for key, arrays in enumerate((train, val)):
        arrays['source_file_id'] = np.full(len(arrays['costs']), key, dtype=np.int8)
        arrays['source_row'] = np.arange(len(arrays['costs']), dtype=np.int32)
    train, all_val, moved = repaired_source_splits(train, val)
    if moved != 704 or len(train['costs']) != 186816 or len(all_val['costs']) != 42960:
        raise ValueError('Unexpected episode59 relocation counts')
    cal_numbers = {p['parent_number'] for p in original['parents'] if p['parent_id'] in CALIBRATION_PARENTS}
    calibration = np.isin(all_val['parent_id'], list(cal_numbers))
    raw = dict(train=train, validation=take(all_val, ~calibration), calibration=take(all_val, calibration), validation_all=all_val)
    expected = dict(train=186816, validation=26832, calibration=16128, validation_all=42960)
    if {k: len(v['costs']) for k, v in raw.items()} != expected:
        raise ValueError('Frozen model-validation/calibration counts disagree')
    for left, right in (('train', 'validation'), ('train', 'calibration'), ('validation', 'calibration')):
        if set(raw[left]['parent_id']) & set(raw[right]['parent_id']):
            raise ValueError('Physical forcing parents cross active splits')
    parents = copy.deepcopy(original['parents'])
    for parent in parents:
        parent['source_split'] = parent['split']
        if parent['parent_id'] in CALIBRATION_PARENTS:
            parent['split'] = 'calibration'
    existing = list(original['parents'])
    # Include previous preassigned v1/v2 TEST descriptions only; no result reads.
    old_root = Path(original['source_benchmark'])
    for path in (old_root / 'data/manifest.json', old_root / 'development/controller_manifest.json'):
        if path.exists():
            previous = read(path)
            existing.extend(previous.get('parents', [])); existing.extend(previous.get('fresh_test_parents', []))
            existing.extend(previous.get('test_parents', []))
    tests = freeze_test_parents(existing)
    evaluation = dict(status='frozen', seed=17, horizon=30, episode_days=14, decision_minutes=15,
        primary_start_step=0, environment='bsm1_offline_benchmark.paper_shared_dataset.PaperPublic57Env',
        version='legacy Han published concentration initialization; unchanged auxiliary startup temperature',
        development=[copy.deepcopy(p) for p in parents if p['parent_id'] in ('validation_p000','validation_p001','validation_p002')],
        test=tests, regression=[dict(case_id='official_'+w, weather=w, split='regression',
            forcing='official unperturbed14-day record', previously_examined=True) for w in ('dry','rain','storm')],
        physical_runs_launched=False, test_selection='Six fixed new forcing realizations, two per shared weather template; not new weather families')
    write(output / 'evaluation_manifest.json', evaluation)
    cases = dict(development=[dict(case_id=p['parent_id'], cohort='development', weather=p['weather'], parent=p)
        for p in evaluation['development']], evaluation=[dict(case_id=p['parent_id'], cohort='fresh_test',
        weather=p['weather'], parent=p) for p in tests] + [dict(case_id=c['case_id'], cohort='official_regression',
        weather=c['weather'], parent=None) for c in evaluation['regression']])
    write(output / 'cases.json', cases)
    summaries, quality = {}, {}
    for split, arrays in raw.items():
        write(output / 'progress.json', dict(status='running', phase='assembling_'+split, updated_at=time.time()))
        canonical, records = canonical_arrays(arrays, original['jobs'])
        save_arrays(output / (split + '.npz'), canonical)
        summaries[split] = dict(transitions=len(canonical['costs']), episodes=len(np.unique(canonical['episode_id'])),
            parents=len(np.unique(canonical['parent_id'])), future_h30=len(canonical['future_starts_h30']),
            han_h30_preceding_cost=len(canonical['han_starts_h30']), quality_valid_rows=int(canonical['quality_valid'].sum()),
            quality_missing_rows=int((~canonical['quality_valid']).sum()), terminal_rows=int(canonical['time_limit'].sum()))
        quality[split] = records
        if split == 'train':
            save_arrays(output / 'normalization.npz', normalization(canonical))
    public_names = original['observation_names'][:41] + original['observation_names'][43:]
    manifest = dict(schema='single_train_bsm1_stock_shared_v1', status='complete', seed=17,
        source_data=str(source), source_manifest=str(source / 'manifest.json'), source_files=['train.npz','validation.npz'],
        source_row_provenance='source_file_id0=originalTRAIN,1=originalVAL; source_row is unchanged source index',
        split_transitions=expected, summaries=summaries, parents=parents,
        source_original_episodes=original['original_episodes'],
        jobs=[dict(j, source_split=j['split'], split='calibration' if j['parent_id'] in CALIBRATION_PARENTS else j['split'])
              for j in original['jobs']],
        episode59_repair='Moved704 originalTRAIN rows intoVAL, joining complete1344-step episode; recomputed internal boundary',
        calibration_parent_ids=list(CALIBRATION_PARENTS), calibration_policy='Parent-disjoint from TRAIN/modelVAL in this freeze; historical VAL previously used, not a newly untouched holdout',
        active_model_splits=['train','validation'], calibration_file='calibration.npz',
        validation_all_policy='Archival union only; not an additional model-selection or fitting split',
        environment='bsm1_offline_benchmark.paper_shared_dataset.PaperPublic57Env',
        environment_version='legacy paper protocol; published concentrations with original auxiliary temperature initialization',
        episode_days=14, episode_steps=1344, decision_minutes=15, return_flow=18846, waste_flow=385, sensor_noise=False,
        objective='Whole14-day EQ +0.3 OCI, exact original interval-integrated rates and sludge inventory change; no physics/cost edits',
        observation_names=public_names, observation_dim=55, privileged_names=original['privileged_names'],
        component_names=original['component_names'], action_names=['Qa/92230', 'KLa5/240'], action_scale=[92230,240],
        privileged_dim=159, public_indices=PUBLIC_INDICES.tolist(), public57_indices=PUBLIC57_INDICES.tolist(),
        public_projection_exact=True, public57_fields=['public57','next_public57'], public57_purpose='PI/expert measured interface; not supplied to stock learnedpublic55 models',
        horizon=30, supported_history_lengths=list(HISTORIES), supported_sampler_steps=[10,20],
        windows='H30 actions/current-to-next physical targets; future_starts_h30 stays inside recorded episode. han_starts_h30 also requires actual preceding interval cost.',
        history='Right-aligned roots ending at current observation; previous_actions at each root contains only prior executed commands. -1 indices/false masks are zero padding, never read next observation. Previous reward=minus prior completed cost/TRAIN mean, separately masked at recording startup.',
        boundaries='terminated/record_boundary stop sequences. physical_terminal=false. bootstrap_mask1 for continuing-policyRL uses actual savednextstate; finite_task_bootstrap_mask0 only at day14. No bootstrap across a reset.',
        quality='Complete comparable4-behavior groups ranked by whole14-day mean cost within parent; best1/worst0, average ranks for ties; originalexcitation quality0 placeholder with quality_valid=false. Allphysical rows retained.',
        reward='Negative original interval daily economic cost divided by positive TRAIN mean; no clipping, shifting or physics change',
        normalization='normalization.npz uses TRAIN only; mean/std for public55/public57/private159/actions/costs/components; std floor1e-6',
        condition_labels='No eight-condition/change labels exist in this nominal bank. Intervention commands are not latent labels.',
        expert_specification=original['expert_specification'], evaluation_manifest='evaluation_manifest.json',
        no_new_simulation=True, no_model_fit=True, no_test_outcome_reads=True, hashes_computed=False,
        completed_at=time.time())
    write(output / 'quality_by_episode.json', quality)
    write(output / 'manifest.json', manifest)
    write(output / 'completion.json', dict(status='complete', split_transitions=expected, moved_episode59_rows=moved,
        public_projection_exact=True, source_unchanged=True, simulations=0, model_fits=0, completed_at=time.time()))
    (output / 'DATASET_CARD.md').write_text(dataset_card(manifest))
    write(output / 'progress.json', dict(status='complete', phase='complete', split_transitions=expected, updated_at=time.time()))
    return manifest

def dataset_card(manifest):
    lines = ['# One shared nominal BSM1 bank for stock baselines and joint models', '',
        'This freeze changes representation, grouping and labels only. Every physical observation, action and economic cost comes from the completed legacy Han-physics source; no trajectories were simulated.', '',
        '| File | Transitions | H30 starts | H30 plus preceding cost |', '|---|---:|---:|---:|']
    for split, item in manifest['summaries'].items():
        lines.append(f"| {split}.npz | {item['transitions']:,} | {item['future_h30']:,} | {item['han_h30_preceding_cost']:,} |")
    lines += ['', '`validation_all.npz` is the archival union, not a model-selection split. Model validation contains the three existing DEV forcing parents plus the original random-excitation validation episodes. Calibration contains the other three complete forcing-parent groups; it is disjoint for this new freeze but was historically part of VAL. Episode59 is now entirely in validation, and the false internal recording boundary is removed.', '',
        'Learned public baselines use `observations`/`next_observations` (55). PI/expert can use `public57`/`next_public57`. Private current/next159 remains separately named. `PUBLIC_INDICES` in `stock_shared_data.py` exactly projects both current and next public55 from private159; it does not make private channels available to public models.', '',
        'Quality is a whole14-day cost rank among four complete behavior trajectories under identical forcing. Originals have no comparable rank: `quality_valid=false`, with zero only a masked placeholder. All transitions remain available for physical training. Normalization is TRAIN-only.', '',
        'H30 future windows cannot cross a record boundary. Histories1/8/24 end at the current root and use previous executed commands; padding is explicit. Han root cost additionally needs the actual preceding interval. `terminated` is a recording boundary, not a Bellman terminal: continuing RL uses `bootstrap_mask`; a declared finite14-day task uses `finite_task_bootstrap_mask`. Both retain the true saved next physical state.', '',
        'The legacy whole14-day task, return flow18,846, waste flow385, actuator bounds and exact inventory contribution are unchanged. This is not the corrected-temperature V2 protocol. The data include no eight hidden regimes or change-event labels.', '',
        'Evaluation is frozen separately in `evaluation_manifest.json`: three existing DEV parents, six new TEST forcing realizations (two per weather), and three previously examined official regression records. No test trajectories or evaluation results are included. Shared dry/rain/storm templates mean these are new forcing realizations, not unseen weather families.', '',
        f"Source: `{manifest['source_data']}`. Row provenance is retained without hashes. Historical datasets remain untouched.", '']
    return '\n'.join(lines)
