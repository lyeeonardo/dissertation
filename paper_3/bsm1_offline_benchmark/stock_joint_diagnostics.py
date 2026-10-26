"""Fixed CAL-only forecasts and proper-score temperature calibration.

The finite temperature is a generalized-denoising-energy calibration, not a
likelihood calibration. No controller selection, physical rollout or fitting
of model weights occurs here. Evaluator truth never enters the evidence query.
"""

from __future__ import annotations

import csv

import math

from pathlib import Path

import time

import numpy as np

from scipy.optimize import minimize_scalar

from .stock_shared_data import BEHAVIORS, CALIBRATION_PARENTS, causal_history, load_arrays, read, write

HISTORIES = (1, 8, 24)

PASSES = (10, 20)

LEADS = (1, 8, 30)

CONDITIONS = ('oracle_current_truth', 'initializer_mean')

PARTICLES = 8

EVIDENCE_TIMES = (.2, .4, .6, .8)

SEED = 17

def select_roots(arrays, manifest, per_episode=8):
    """Select equally spaced valid starts before querying any model outputs."""
    if manifest.get('status') != 'complete':
        raise ValueError('Completed shared-stock dataset required')
    allowed = set(CALIBRATION_PARENTS)
    if set(manifest['calibration_parent_ids']) != allowed:
        raise ValueError('Use the three preassigned calibration parents')
    jobs = [j for j in manifest['jobs'] if j['split'] == 'calibration']
    if len(jobs) != 12 or {(j['parent_id'], j['behavior']) for j in jobs} != {
            (p, b) for p in allowed for b in BEHAVIORS}:
        raise ValueError('CAL must contain exactly three parents by four behaviors')
    if set(map(int, np.unique(arrays['episode_id']))) != {j['episode_id'] for j in jobs}:
        raise ValueError('Calibration arrays contain unexpected or missing episodes')
    roots = []
    for job in sorted(jobs, key=lambda j: (j['parent_id'], BEHAVIORS.index(j['behavior']))):
        candidates = arrays['future_starts_h30']
        candidates = candidates[arrays['episode_id'][candidates] == job['episode_id']]
        if len(candidates) < per_episode:
            raise ValueError('Insufficient complete H30 starts in calibration episode')
        selected = candidates[np.linspace(0, len(candidates)-1, per_episode, dtype=int)]
        for ordinal, row in enumerate(selected):
            future = np.arange(row, row+30)
            if (np.any(arrays['episode_id'][future] != job['episode_id'])
                    or not np.array_equal(arrays['step'][future], arrays['step'][row]+np.arange(30))):
                raise ValueError('Selected forecast crosses a recording boundary')
            if int(arrays['parent_id'][row]) != job['parent_number']:
                raise ValueError('Calibration parent identity mismatch')
            roots.append(dict(root_id=f"cal_{len(roots):03d}", row=int(row),
                episode_id=int(job['episode_id']), parent_id=job['parent_id'],
                parent_number=int(job['parent_number']), weather=job['weather'],
                behavior=job['behavior'], step=int(arrays['step'][row]), ordinal=ordinal))
    return roots

def energy_weights(energy, tau):
    energy = np.asarray(energy, dtype=np.float64)
    if not np.isfinite(energy).all() or not math.isfinite(tau) or tau <= 0:
        raise ValueError('Finite energies and a positive finite temperature required')
    logits = -(energy - energy.min(axis=-1, keepdims=True)) / tau
    exponent = np.exp(logits)
    return exponent / exponent.sum(axis=-1, keepdims=True)

def energy_score_distances(particles, truth, scale):
    """RMS Euclidean distance in frozen TRAIN-standardized hidden coordinates."""
    particles, truth, scale = map(lambda x: np.asarray(x, dtype=np.float64), (particles, truth, scale))
    if particles.ndim != 3 or truth.shape != (particles.shape[0], particles.shape[2]):
        raise ValueError('Expected particles[contexts,K,D] and truth[contexts,D]')
    if scale.shape != (particles.shape[2],) or np.any(scale <= 0) or not all(
            np.isfinite(x).all() for x in (particles, truth, scale)):
        raise ValueError('Finite particles/truth and positive frozen scales required')
    point = np.sqrt(np.mean(((particles-truth[:, None])/scale)**2, axis=-1))
    pair = np.sqrt(np.mean(((particles[:, :, None]-particles[:, None, :])/scale)**2, axis=-1))
    return point, pair

def weighted_energy_score(weights, point_distance, pair_distance):
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not np.allclose(weights.sum(-1), 1.):
        raise ValueError('Nonnegative normalized particle weights required')
    return np.sum(weights*point_distance, axis=-1) - .5*np.einsum('bi,bij,bj->b', weights, pair_distance, weights)

def fit_temperature(energies, predicted_hidden, actual_next_hidden, hidden_scale):
    """One bounded log-temperature fit with an explicit exact uniform reference."""
    energies = np.asarray(energies, dtype=np.float64)
    point, pair = energy_score_distances(predicted_hidden, actual_next_hidden, hidden_scale)
    if energies.shape != point.shape or not np.isfinite(energies).all():
        raise ValueError('One finite energy per predictive particle is required')
    centered = energies - energies.mean(axis=1, keepdims=True)
    contrasts = np.sqrt(np.mean(centered**2, axis=1))
    positive = contrasts[contrasts > 0]
    energy_scale = max(float(np.median(positive)), 1e-12) if len(positive) else 1.
    bounds = (math.log(energy_scale)-math.log(1000.), math.log(energy_scale)+math.log(1000.))
    objective = lambda logtau: float(weighted_energy_score(energy_weights(energies, math.exp(logtau)), point, pair).mean())
    optimum = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-5})
    if not optimum.success or not math.isfinite(optimum.fun):
        raise RuntimeError('Bounded temperature calibration did not converge')
    tau = math.exp(float(optimum.x))
    weights = energy_weights(energies, tau)
    uniform = np.full_like(weights, 1/weights.shape[1])
    calibrated_scores = weighted_energy_score(weights, point, pair)
    uniform_scores = weighted_energy_score(uniform, point, pair)
    ess = 1/np.sum(weights**2, axis=1)
    total_variation = .5*np.abs(weights-uniform).sum(axis=1)
    delta = float(calibrated_scores.mean()-uniform_scores.mean())
    near_upper = bounds[1]-float(optimum.x) <= .01*(bounds[1]-bounds[0])
    result = dict(tau=tau, energy_contrast_scale=energy_scale, log_tau_bounds=list(bounds),
        tau_bounds=[math.exp(x) for x in bounds], optimization_evaluations=int(optimum.nfev),
        contexts=len(weights), particles=weights.shape[1], hidden_dimensions=predicted_hidden.shape[-1],
        calibrated_energy_score=float(calibrated_scores.mean()), uniform_energy_score=float(uniform_scores.mean()),
        score_delta_calibrated_minus_uniform=delta,
        relative_score_reduction=-delta/max(float(uniform_scores.mean()), 1e-12),
        mean_effective_sample_size=float(ess.mean()), min_effective_sample_size=float(ess.min()),
        mean_total_variation_from_uniform=float(total_variation.mean()),
        optimum_near_upper_bound=bool(near_upper), near_no_update=bool(near_upper or total_variation.mean() <= .01),
        recommend_update=bool(delta < -1e-10), recommendation_is_advisory=True,
        formula='sum_i w_i RMS((z_i-y)/TRAINscale) - 0.5 sum_ij w_i w_j RMS((z_i-z_j)/TRAINscale)',
        weight_formula='softmax(-energy/tau); subtracting a common energy offset is numerical only',
        objective='Equal mean weighted empirical energy score over all96 CAL roots and histories1/8/24',
        evaluation='Temperature is fitted and scored on the same CAL contexts; its score change is in-sample, not held-out evidence of update benefit',
        interpretation='Generalized-energy temperature; restricted eight-particle predictive ensemble with common process noise, not a normalized likelihood or full-posterior calibration',
        runtime='Finite tau retained for the prespecified VFE arm regardless of advisory recommendation; no arm is disabled')
    return result, dict(weights=weights, uniform_weights=uniform, calibrated_scores=calibrated_scores,
        uniform_scores=uniform_scores, effective_sample_size=ess, total_variation=total_variation,
        point_distance=point, pair_distance=pair)

def _numpy(value):
    return value.detach().cpu().numpy()

def _tensor_history(history, device):
    import torch
    return {k: torch.as_tensor(v, device=device, dtype=torch.bool if k in ('valid','reward_valid') else torch.float32)
            for k, v in history.items() if k != 'mask'}

def diagnostic_root(model, arrays, row, *, selected_passes, root_ordinal):
    """Query one root. The only truth inputs are the explicitly named oracle arm.

    The calibration evidence query has no access to current/future hidden truth;
    actual next hidden truth is used later by fit_temperature only.
    """
    import torch
    device = model.state_mean.device
    cost_scale = float(model.cost_scale.detach().cpu())
    seed = SEED + 104729*root_ordinal
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn((1,30,162), device=device, generator=generator)
    known_noise = torch.randn((1,30,162), device=device, generator=generator)
    evidence_noise = torch.randn((len(EVIDENCE_TIMES),58), device=device, generator=generator)
    current = torch.as_tensor(arrays['privileged'][row:row+1], device=device, dtype=torch.float32)
    actions = torch.as_tensor(arrays['actions'][None,row:row+30], device=device, dtype=torch.float32)
    next_public = torch.as_tensor(arrays['next_observations'][row], device=device, dtype=torch.float32)
    reward = torch.as_tensor(-arrays['costs'][row]/cost_scale, device=device, dtype=torch.float32)
    output = dict(root_row=np.array(row), seed=np.array(seed), actual_current_private=arrays['privileged'][row],
        actual_next_private=arrays['next_privileged'][row:row+30], actual_next_public=arrays['next_observations'][row:row+30],
        actual_actions=arrays['actions'][row:row+30], actual_rewards=-arrays['costs'][row:row+30]/cost_scale,
        initial_noise=_numpy(initial_noise[0]), known_noise=_numpy(known_noise[0]),
        evidence_common_noise=_numpy(evidence_noise), evidence_times=np.asarray(EVIDENCE_TIMES),
        history_lengths=np.asarray(HISTORIES), passes=np.asarray(PASSES), conditions=np.asarray(CONDITIONS))
    means, scales, particles, predictions, rewards, energies, energies_by_time, next_particles = [], [], [], [], [], [], [], []
    with torch.no_grad():
        for h in HISTORIES:
            raw_history = causal_history(arrays, np.array([row]), h, cost_scale)
            history = _tensor_history(raw_history, device)
            for key, value in raw_history.items():
                if key != 'mask':
                    output[f'history_h{h}_{key}'] = value[0]
            # Pair history lengths with the same standardized particle draws;
            # only their learned means/scales and public context differ.
            generator.manual_seed(seed + 19)
            initializer = model.initialize(history, samples=PARTICLES, generator=generator)
            means.append(_numpy(initializer['mean'][0])); scales.append(_numpy(initializer['scale'][0]))
            particles.append(_numpy(initializer['samples'][0]))
            predictions_h, rewards_h = [], []
            for passes in PASSES:
                predictions_p, rewards_p = [], []
                for condition in (current, initializer['mean']):
                    forecast = model.predict_actions(condition, history, actions, passes=passes,
                        initial_noise=initial_noise, known_noise=known_noise)
                    predictions_p.append(_numpy(forecast['next_private'][0]))
                    rewards_p.append(_numpy(forecast['rewards'][0]))
                predictions_h.append(predictions_p); rewards_h.append(rewards_p)
            predictions.append(predictions_h); rewards.append(rewards_h)
            # All hypotheses share pretransition history, observed evidence and corruption.
            particle_history = {k:v.expand(PARTICLES,*v.shape[1:]) for k,v in history.items()}
            root_particles = initializer['samples'][0]
            evidence = model.evidence(root_particles, particle_history, next_public, reward, actions[0,0],
                times=EVIDENCE_TIMES, noise=evidence_noise)
            energies.append(_numpy(evidence['energy'])); energies_by_time.append(_numpy(evidence['energy_by_time']))
            particle_forecast = model.predict_actions(root_particles, particle_history,
                actions[:,:1].expand(PARTICLES,1,2), passes=selected_passes,
                initial_noise=initial_noise.expand(PARTICLES,30,162), known_noise=known_noise.expand(PARTICLES,30,162))
            next_particles.append(_numpy(particle_forecast['next_private'][:,0]))
    output.update(initializer_mean=np.asarray(means), initializer_scale=np.asarray(scales),
        initializer_particles=np.asarray(particles), forecast_next_private=np.asarray(predictions),
        forecast_rewards=np.asarray(rewards), evidence_energy=np.asarray(energies),
        evidence_energy_by_time=np.asarray(energies_by_time), predictive_next_particles=np.asarray(next_particles))
    if not all(np.isfinite(v).all() for v in output.values() if np.asarray(v).dtype.kind in 'fc'):
        raise FloatingPointError('Nonfinite calibration query output; stop without replacing failed samples')
    return output

def root_metrics(values, root, state_scale, public_indices, hidden_indices, cost_scale):
    common = {k: root[k] for k in ('root_id','parent_id','weather','behavior','episode_id','step')}
    initial, forecasts = [], []
    for hi, history in enumerate(HISTORIES):
        delta = (values['initializer_mean'][hi]-values['actual_current_private'])/state_scale
        hidden_delta = delta[hidden_indices]
        initial.append(common | dict(history=history, hidden_normalized_mse=float(np.mean(hidden_delta**2)),
            hidden_coverage90=float(np.mean(np.abs(hidden_delta) <= 1.644853627*
                values['initializer_scale'][hi,hidden_indices]/state_scale[hidden_indices])),
            negative_hidden_mean_fraction=float(np.mean(values['initializer_mean'][hi,hidden_indices] < 0))))
        for pi, passes in enumerate(PASSES):
            for ci, condition in enumerate(CONDITIONS):
                errors = (values['forecast_next_private'][hi,pi,ci]-values['actual_next_private'])/state_scale
                reward_error = values['forecast_rewards'][hi,pi,ci]-values['actual_rewards']
                for lead in LEADS:
                    forecasts.append(common | dict(history=history, passes=passes, condition=condition, lead=lead,
                        public_normalized_mse=float(np.mean(errors[lead-1,public_indices]**2)),
                        hidden_normalized_mse=float(np.mean(errors[lead-1,hidden_indices]**2)),
                        reward_mse=float(reward_error[lead-1]**2),
                        economic_cost_daily_squared_error=float((cost_scale*reward_error[lead-1])**2)))
    return initial, forecasts

def _csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def aggregate(rows, keys, mse_fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group, values in groups.items():
        out = dict(zip(keys,group)); out['roots'] = len(values)
        for field in mse_fields:
            name = field.replace('_mse','_rmse').replace('_squared_error','_rmse')
            out[name] = float(np.sqrt(np.mean([row[field] for row in values])))
        result.append(out)
    return result

def run(data, checkpoint, output, *, passes=20, device='cuda'):
    import torch
    from .joint_diffusion import load_model
    data, checkpoint, output = map(lambda x:Path(x).resolve(), (data, checkpoint, output))
    if passes not in PASSES:
        raise ValueError('Supply the preselected oracle budget10 or20')
    if not str(device).startswith('cuda'):
        raise ValueError('Production CAL diagnostic requires CUDA; pure CPU stubs belong in focused tests')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Fresh calibration output required; no automatic retry or overwrite')
    output.mkdir(parents=True, exist_ok=True)
    (output/'roots').mkdir()
    started = time.time()
    manifest = read(data/'manifest.json')
    arrays = load_arrays(data/'calibration.npz')
    roots = select_roots(arrays, manifest)
    write(output/'manifest.json', dict(status='frozen', data=str(data), source=str(data/'calibration.npz'),
        checkpoint=str(checkpoint), roots=roots, seed=SEED, histories=list(HISTORIES), forecast_passes=list(PASSES),
        forecast_conditions=list(CONDITIONS), horizon=30, leads=list(LEADS), particles=PARTICLES,
        calibration_forecast_passes=passes, selection='Eight equally spaced H30 starts per CAL parent/behavior, including startup',
        calibration_only=True, developmental_or_test_data_used=False,
        history='Same pretransition public history for initializer, forecast and evidence; prior completed rewards only',
        evidence='Actual action, next public55 and completed reward only; common corruption across particles',
        calibration_score='Weighted energy score on actual next104 hidden coordinates, offline evaluator truth only',
        temperature='Single finite pooled logtau fit; exact uniform reference separate; advisory recommendation never disables an arm',
        randomness='seed17+104729*root ordinal; common forecast noise across histories/conditions/passes/particles',
        calibration_parent_note=manifest['calibration_policy'],
        uncertainty='Fixed CAL contexts within three parents are not independent replications; no CI'))
    def progress(completed, phase, status='running'):
        write(output/'progress.json', dict(status=status, phase=phase, completed_roots=completed,
            total_roots=len(roots), elapsed_seconds=time.time()-started, updated_at=time.time()))
    progress(0,'load_model')
    torch.set_num_threads(1)
    model = load_model(checkpoint, device=device)
    from work3_paths import verify_relocated_training_data
    verify_relocated_training_data(model.recipe['data'], data)
    state_scale = _numpy(model.state_scale).astype(np.float64)
    public_indices, hidden_indices = _numpy(model.public_indices), _numpy(model.hidden_indices)
    cost_scale = float(model.cost_scale.detach().cpu())
    np.savez_compressed(output/'normalization.npz', state_scale=state_scale, state_mean=_numpy(model.state_mean),
        public_indices=public_indices, hidden_indices=hidden_indices, cost_scale=np.array(cost_scale))
    all_initial, all_forecasts, energies, particles, truths = [], [], [], [], []
    completed = 0
    try:
        for i, root in enumerate(roots):
            values = diagnostic_root(model, arrays, root['row'], selected_passes=passes, root_ordinal=i)
            np.savez_compressed(output/'roots'/f"{root['root_id']}.npz", **values)
            initial, forecasts = root_metrics(values,root,state_scale,public_indices,hidden_indices,cost_scale)
            all_initial.extend(initial); all_forecasts.extend(forecasts)
            energies.append(values['evidence_energy'])
            particles.append(values['predictive_next_particles'][...,hidden_indices])
            truths.append(np.broadcast_to(values['actual_next_private'][0,hidden_indices], (len(HISTORIES),len(hidden_indices))))
            completed = i+1
            progress(completed,'root_queries')
        energies = np.concatenate(energies); particles = np.concatenate(particles); truths = np.concatenate(truths)
        calibration, scores = fit_temperature(energies, particles, truths, state_scale[hidden_indices])
        np.savez_compressed(output/'temperature_scores.npz', energies=energies,
            predictive_hidden_particles=particles, evaluator_actual_next_hidden=truths,
            context_root_id=np.repeat([r['root_id'] for r in roots],len(HISTORIES)),
            context_history=np.tile(HISTORIES,len(roots)), **scores)
        by_history = []
        for hi, h in enumerate(HISTORIES):
            index=np.arange(hi,len(energies),len(HISTORIES))
            by_history.append(dict(history=h,roots=len(roots),
                calibrated_energy_score=float(scores['calibrated_scores'][index].mean()),
                uniform_energy_score=float(scores['uniform_scores'][index].mean()),
                mean_effective_sample_size=float(scores['effective_sample_size'][index].mean())))
        calibration['by_history'] = by_history
        calibration['forecast_passes'] = passes
        calibration['evidence_sampler_independent'] = True
        write(output/'calibration.json', calibration)
        write(output/'runtime_temperature.json', dict(tau=calibration['tau'],
            checkpoint=str(checkpoint), forecast_passes_used_for_offline_calibration=passes,
            evidence_times=list(EVIDENCE_TIMES), history_lengths=list(HISTORIES),
            applies_to_sampler_passes=list(PASSES), energy_formula='JointDiffusion.evidence energy without extra weighting',
            recommendation_is_advisory=True, recommend_update=calibration['recommend_update'],
            near_no_update=calibration['near_no_update'], truth_inputs_required_at_runtime=False))
        forecast_summary = aggregate(all_forecasts, ('history','passes','condition','lead'),
            ('public_normalized_mse','hidden_normalized_mse','reward_mse','economic_cost_daily_squared_error'))
        initializer_summary = aggregate(all_initial, ('history',), ('hidden_normalized_mse',))
        for row in initializer_summary:
            subset=[r for r in all_initial if r['history']==row['history']]
            for key in ('hidden_coverage90','negative_hidden_mean_fraction'):
                row[key]=float(np.mean([r[key] for r in subset]))
        _csv(output/'initializer_by_root.csv',all_initial); _csv(output/'forecasts_by_root.csv',all_forecasts)
        _csv(output/'initializer_summary.csv',initializer_summary); _csv(output/'forecast_summary.csv',forecast_summary)
        write(output/'completion.json',dict(status='complete',checkpoint=str(checkpoint),
            checkpoint_updates=int(model.checkpoint_updates), roots=len(roots), contexts=len(energies),
            calibration_forecast_passes=passes,tau=calibration['tau'], calibration=calibration,
            initializer_diagnostics=initializer_summary, forecast_diagnostics=forecast_summary,
            elapsed_seconds=time.time()-started, runtime_parameters=str(output/'runtime_temperature.json')))
        note = ('The calibrated weights improve the pooled CAL proper score.' if calibration['recommend_update'] else
                'Exact uniform/no-update weights score at least as well as the finite calibrated weights.')
        (output/'RESULTS.md').write_text(f"# Joint-model CAL diagnostic\n\n{note}\n\n"
            f"Energy score: uniform {calibration['uniform_energy_score']:.6g}; calibrated {calibration['calibrated_energy_score']:.6g}. "
            f"Finite tau {calibration['tau']:.6g}; mean ESS {calibration['mean_effective_sample_size']:.3f}/8; "
            f"near-no-update flag {calibration['near_no_update']}. The finite tau remains assigned to the prespecified VFE arm.\n\n"
            f"The calibration forecast uses {passes} DDIM passes. The evidence energy itself is sampler-independent; the same tau applies to both 10/20-pass comparisons. "
            "Forecast tables separately report oracle current-truth and initializer-mean roots for histories1/8/24 at exact leads1/8/30. "
            "They measure one common-noise sample against recorded outcomes, not predictive likelihood.\n\n"
            "The temperature is fitted and scored on the same CAL contexts; its score change is in-sample, not held-out evidence of update benefit. "
            "The proper score evaluates a restricted eight-particle next-hidden ensemble with shared process noise. It does not establish normalized likelihoods or calibrated full posterior uncertainty. "
            "All96 contexts come from three CAL parents; their within-parent dependence precludes an independent-replication interpretation. "
            "CAL is newly disjoint from TRAIN/modelVAL for this freeze but was historical validation data. Simulator truth is used only for the oracle diagnostic and offline scoring. "
            "No controller was selected, disabled, trained or physically evaluated here.\n")
        progress(len(roots),'complete','complete')
    except BaseException as error:
        write(output/'failure.json',dict(status='failed',error=repr(error),completed_roots=completed,
            elapsed_seconds=time.time()-started, automatic_retry=False))
        raise
    return read(output/'completion.json')
