"""Recompute final dissertation summaries from saved traces, without model/physics calls.

Usage: python -m work2.report --input RUN_DIR [--input OTHER_ARM] --output report.json
Absolute trace paths and previously stored numerical metrics are never trusted.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import episode_metrics

CONFIG=Path(__file__).resolve().parent/'configs'
BITS=(('gravity',4),('friction',2),('joint_range',1))
METHODS=('genaif','history','dadp','oracle')
STUDY_NOTES={
    'benchmark': 'Fixed 120-case, five-family benchmark; all adverse outcomes retained.',
    'preference_a': 'Matched 40-case preference panel A; distinct from panel B.',
    'preference_b': 'Matched 40-case preference panel B; distinct from panel A.',
    'all_conditions': 'Outcome-selected all-condition illustration; not a representative benchmark.',
    'gravity_sweep': 'One reset at each physical gravity; fixed 1x/3x hypothesis bank. Off-bank slots are not ground-truth hypotheses.',
    'adaptation': 'Two target and two retention resets; the declared 1x/5x or 1x/3x bank is retained per case. Updates are separate from control.',
}

def read(path):return json.loads(Path(path).read_text())

def expert_reference(path):
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=8 or {int(r['condition'],2) for r in rows}!=set(range(8)):raise ValueError('All eight unique expert conditions required')
    values={int(r['condition'],2):float(r['mean_step_reward']) for r in rows}
    if not all(np.isfinite(v) and v>0 for v in values.values()):raise ValueError('Finite positive expert references required')
    return values

def scheduled_conditions(case):
    schedule=np.full(int(case['steps']),int(case['initial_condition']),dtype=np.int64)
    for event in sorted(case['events'],key=lambda e:e['step']):
        step=int(event['step']);condition=int(event['condition'])
        if not 0<=step<len(schedule) or not 0<=condition<8:raise ValueError('Invalid scheduled switch')
        schedule[step:]=condition
    return schedule

def preaction_belief(arrays):
    """Decision-prior MAP and separate two-mass marginals; ties choose bit zero."""
    if 'decision_prior' not in arrays:return None
    p=np.asarray(arrays['decision_prior'],dtype=np.float64);truth=np.asarray(arrays['conditions'],dtype=np.int64)
    if p.shape!=(len(truth),8) or not np.isfinite(p).all() or np.any(p<0):raise ValueError('Finite N-by-eight pre-action decision prior required')
    np.testing.assert_allclose(p.sum(1),1.,atol=2e-6,rtol=0)
    n=len(truth);result=dict(decisions=n,map_correct=int(np.sum(p.argmax(1)==truth)),true_mass_sum=float(p[np.arange(n),truth].sum()),alignment='decision_prior[t], before action t, versus actual conditions[t]; never beliefs[t+1]')
    hypotheses=np.arange(8)
    for name,bit in BITS:
        one=(hypotheses&bit)!=0
        marginal=np.stack((p[:,~one].sum(1),p[:,one].sum(1)),axis=1)
        target=((truth&bit)!=0).astype(int)
        result[name+'_correct']=int(np.sum(marginal.argmax(1)==target))
        result[name+'_true_mass_sum']=float(marginal[np.arange(n),target].sum())
    return result

def inversion_readout(arrays):
    """Retain the native indicator; next-observation pitch is a separate check."""
    mask=np.asarray(arrays['inverted'],dtype=bool)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    intervals=[dict(first_transition=int(a),last_transition=int(b-1),
                    first_observed_seconds=float((a+1)*.05),
                    last_observed_seconds=float(b*.05),steps=int(b-a))
               for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    result=dict(recorded_inverted_steps=int(mask.sum()),intervals=intervals)
    if 'observations' in arrays:
        pitch=np.asarray(arrays['observations'])[1:,1]
        if pitch.shape!=mask.shape:raise ValueError('Next-observation pitch alignment differs')
        cosine=np.cos(pitch)
        result.update(negative_next_pitch_cosines=int((cosine<0).sum()),
                      indicator_pitch_disagreements=int(np.sum(mask!=(cosine<0))),
                      scope='Saved inversion indicator governs reported count; pitch disagreement is retained, not corrected')
    return result

def saved_preference_decision(data,controller,case_id,step):
    """Derive the selected Appendix B decision from its saved forecasts only."""
    if len(data['rewards'])<=step:return None
    prior=data['decision_prior'][step].astype(float)
    observation=data['predicted_observations'][step].astype(float)
    reward=data['predicted_rewards'][step].astype(float)
    valid=data['prediction_valid'][step].astype(bool)
    discount=controller['discount']**np.arange(4)
    expected=((reward*valid*discount).sum(-1)*prior).sum(-1)
    penalty=(((1-np.cos(observation[...,1]))*valid*discount).sum(-1)*prior).sum(-1)*controller['posture_coefficient']
    score=expected-penalty
    for recomputed,key in ((expected,'expected_reward'),(penalty,'orientation_penalty'),(score,'selection_score')):
        np.testing.assert_allclose(recomputed,data[key][step],atol=3e-6,rtol=2e-6)
    selected=int(data['selected_index'][step])
    if selected!=int(score.argmax()):raise ValueError('Saved choice differs from recomputed score')
    angle=(observation[:,:,2,1]*prior).sum(-1)
    tilt=np.degrees(abs(np.arctan2(np.sin(angle),np.cos(angle))))
    return dict(case_id=case_id,decision_index=step,decision_seconds=step*.05,
                selected_plan=selected,displayed_pose_lead_seconds=.15,
                plans=[dict(plan=i,expected_reward=float(expected[i]),posture_penalty=float(penalty[i]),
                            score=float(score[i]),mean_pose_tilt_degrees=float(tilt[i])) for i in range(3)],
                scope='Outcome-selected illustrative bank; all scores use four forecast steps and full pre-action belief. Pose tilt uses only the +0.15 s mean pose; alternatives were not executed.')

def preference_ranking(data,case):
    """Native saved-bank eligibility and ordering from decision/select.py.

    Keep one winner per case: the minimum of the per-case minima is exactly
    the minimum over every eligible decision. No candidate bank is executed.
    """
    reward=data['expected_reward'].astype(float)
    penalty=data['orientation_penalty'].astype(float)
    score=data['selection_score'].astype(float)
    selected=data['selected_index'].astype(int)
    native=data['native_selected_index'].astype(int)
    n=len(reward)
    if reward.shape!=(n,3):raise ValueError('Expected three saved candidate rewards per decision')
    np.testing.assert_array_equal(selected,score.argmax(1))
    np.testing.assert_array_equal(native,reward.argmax(1))
    np.testing.assert_allclose(score,reward-penalty,atol=8e-6,rtol=0)
    idx=np.arange(n);bank=data['banks']
    commands=bank[idx,selected,0];native_commands=bank[idx,native,0]
    np.testing.assert_array_equal(commands,data['actions'])
    changed=np.any(commands!=native_commands,axis=1)
    valid=data['prediction_valid'].all(axis=(1,2,3))
    positive=(reward>0).all(axis=1)
    observation=data['observations']
    pitch=np.arctan2(np.sin(observation[:-1,1]),np.cos(observation[:-1,1])).astype(float)
    current_speed=observation[:-1,8].astype(float)
    reduction=penalty[idx,native]-penalty[idx,selected]
    sacrifice=reward[idx,native]-reward[idx,selected]
    gain=score[idx,selected]-score[idx,native]
    eligible=changed & valid & positive & (current_speed>1.) & (np.abs(pitch)<np.pi/2) & (gain>0)
    best=None
    if eligible.any():
        step=min(np.flatnonzero(eligible),key=lambda i:(-float(reduction[i]),-float(gain[i]),case['case_id'],int(i)))
        best=dict(rank=[-float(reduction[step]),-float(gain[step]),case['case_id'],int(step)],
                  case_id=case['case_id'],family=case['family'],transition_index=int(step),
                  decision_time_s=float(step*.05),reward_only_plan=int(native[step]),selected_plan=int(selected[step]),
                  current_speed_m_s=float(current_speed[step]),current_wrapped_pitch_degrees=float(np.rad2deg(pitch[step])),
                  preference_penalty_reduction=float(reduction[step]),expected_reward_sacrifice=float(sacrifice[step]),
                  final_score_advantage=float(gain[step]))
    return dict(population_decisions=n,changed_commands=int(changed.sum()),eligible_decisions=int(eligible.sum()),best=best)

def select_preference_decision(rows):
    """Rank the exact declared population, retaining explicit partial scope."""
    source=[r for r in rows if 'preference_ranking' in r]
    if not source:return None
    expected={c['case_id']:c['steps'] for c in read(CONFIG/'cases.json')['benchmark']}
    complete=({r['case_id'] for r in source}==set(expected) and
              all(r['preference_ranking']['population_decisions']==expected[r['case_id']] for r in source))
    eligible=[r for r in source if r['preference_ranking']['best'] is not None]
    selected=detail=None
    if eligible:
        winner=min(eligible,key=lambda r:r['preference_ranking']['best']['rank'])
        selected=winner['preference_ranking']['best']
        with np.load(Path(winner['source_run'])/winner['trace'],allow_pickle=False) as data:
            detail=saved_preference_decision(data,read(CONFIG/'controller.json'),selected['case_id'],selected['transition_index'])
    return dict(state='selected' if complete and selected is not None else 'partial' if not complete else 'no_eligible_decisions',
                complete_population=complete,cases=len(source),expected_cases=len(expected),
                expected_population_decisions=sum(expected.values()),
                **{key:sum(r['preference_ranking'][key] for r in source) for key in ('population_decisions','changed_commands','eligible_decisions')},
                eligibility=['Selected first command differs exactly from reward-only first command',
                             'All 3x8x4 forecast validity flags are true',
                             'All three full-prior expected discounted rewards are positive',
                             'Current observations[t,8] speed >1 m/s and absolute wrapped observations[t,1] pitch <90 degrees',
                             'Selected plan final-score advantage over reward-only winner is strictly positive'],
                ranking=['Descending preference-penalty reduction','Descending final-score advantage','Ascending case id','Ascending transition index'],
                selected=selected,forecast_table=detail,
                interpretation='Outcome-selected largest eligible saved preference tradeoff; not a typical decision or an executed physical counterfactual. Partial populations cannot reproduce the final population claim.')

def load_run(directory, reference=None):
    directory=Path(directory).resolve();rows=read(directory/'results.json')
    if not isinstance(rows,list):raise ValueError('Expected work2.run results.json list; old campaign reports must be explicitly converted to this schema')
    design=read(directory/'design.json') if (directory/'design.json').exists() else {}
    cases=read(CONFIG/'cases.json');controller=read(CONFIG/'controller.json');out=[]
    for raw in rows:
        if raw.get('state')!='completed':continue
        study=raw['study'];method=raw['method'];case=raw['case'];arm=raw.get('arm',design.get('arm','zero'))
        if method not in METHODS+('reward_only',):raise ValueError('Unknown fixed controller')
        expected={c['case_id']:c for c in cases.get(study,[])}
        if case['case_id'] not in expected or case!=expected[case['case_id']]:raise ValueError(f'Case differs from fixed {study} design: {case["case_id"]}')
        # Reconstructed path is relative to the supplied run directory. A copied
        # run continues to work even if its metadata contains an old absolute path.
        trace=directory/method/f"case_{case['case_id']:04d}"/'trace.npz'
        if not trace.is_file():raise FileNotFoundError(trace)
        with np.load(trace,allow_pickle=False) as data:
            required=('rewards','speed','inverted','conditions','terminated','truncated','decision_s','update_s','simulator_s')
            arrays={k:data[k].copy() for k in required}
            for k in ('decision_prior','batch_size','planning_mask','update_mask'):
                if k in data:arrays[k]=data[k].copy()
            if study.startswith('preference'):arrays['observations']=data['observations'].copy()
            ranking=preference_ranking(data,case) if (study,method)==('benchmark','genaif') else None
        n=len(arrays['rewards']);assigned=int(case['steps'])
        if n<1 or n>assigned:raise ValueError('Observed trace length must be within the assigned episode')
        for key in required:
            if arrays[key].shape!=(n,) or not np.isfinite(arrays[key]).all():raise ValueError(f'Invalid observed transition array: {key}')
        if method in ('genaif','reward_only') and 'decision_prior' not in arrays:raise ValueError('Missing pre-action belief; posterior arrays cannot substitute')
        truth=scheduled_conditions(case)
        np.testing.assert_array_equal(arrays['conditions'],truth[:n])
        metrics=episode_metrics('main',case,arrays,method,controller)
        planning=np.asarray(arrays['decision_s'],dtype=float);mask=np.asarray(arrays.get('planning_mask',np.ones(n,bool)),dtype=bool)
        counts=np.bincount(truth,minlength=8)
        baseline=float(sum(counts[c]*reference[c] for c in range(8))) if reference is not None and study in ('benchmark','preference_a','preference_b','all_conditions') else None
        result=dict(study=study,arm=arm,method=method,case_id=case['case_id'],family=case['family'],case_key=case.get('case_key'),actual_gravity=case.get('actual_gravity'),public_high_gravity=case.get('high_gravity',3),reset_seed=case['reset_seed'],sampling_seed=case['sampling_seed'],source_run=str(directory),trace=f"{method}/case_{case['case_id']:04d}/trace.npz",metrics=metrics,expert_reference_return=baseline,expert_relative_percent=100*(metrics['return']/baseline-1) if baseline is not None else None,planning_seconds=float(planning[mask].sum()),planning_calls=int(mask.sum()),preaction_belief=preaction_belief(arrays) if study!='gravity_sweep' else None,regenerated_checkpoints=design.get('regenerated_checkpoints',False))
        result['inversion_readout']=inversion_readout(arrays)
        if ranking is not None:result['preference_ranking']=ranking
        out.append(result)
    return out

def aggregate(rows,expected_cases):
    observed=sum(r['metrics']['native_steps'] for r in rows);events=sum(r['metrics']['scheduled_events'] for r in rows)
    relative=[r['expert_relative_percent'] for r in rows if r['expert_relative_percent'] is not None]
    calls=sum(r['planning_calls'] for r in rows)
    result=dict(cases=len(rows),case_ids=sorted(r['case_id'] for r in rows),expected_cases=expected_cases,complete_case_coverage=len(rows)==expected_cases,all_episodes_complete=all(r['metrics']['completed_full_duration'] for r in rows),mean_return=float(np.mean([r['metrics']['return'] for r in rows])),mean_expert_relative_percent=float(np.mean(relative)) if relative else None,observed_steps=observed,assigned_steps=sum(r['metrics']['assigned_steps'] for r in rows),planning_ms_per_action=1000*sum(r['planning_seconds'] for r in rows)/observed,planning_ms_per_planning_call=1000*sum(r['planning_seconds'] for r in rows)/calls if calls else None,planning_calls=calls,planning_scope='Pre-action history preparation and action selection; excludes observation update and simulator',scheduled_events=events,reached_events=sum(r['metrics']['reached_events'] for r in rows),confirmed_events=sum(r['metrics']['confirmed_events'] for r in rows),sustained_events=sum(r['metrics']['sustained_events'] for r in rows),early_ends=sum(not r['metrics']['completed_full_duration'] for r in rows),inverted_steps=sum(r['metrics']['inverted_steps'] for r in rows),regenerated_checkpoint_runs=any(r['regenerated_checkpoints'] for r in rows))
    result['sustained_fraction']=result['sustained_events']/events if events else None
    result['episodes_with_inversion']=sum(r['metrics']['inverted_steps']>0 for r in rows)
    b=[r['preaction_belief'] for r in rows if r['preaction_belief'] is not None]
    if b:
        n=sum(r['decisions'] for r in b)
        result['preaction_belief']=dict(decisions=n,cases=len(b),map_accuracy=sum(r['map_correct'] for r in b)/n,true_mass=sum(r['true_mass_sum'] for r in b)/n,**{name+'_marginal_accuracy':sum(r[name+'_correct'] for r in b)/n for name,bit in BITS},scope='Pre-action own-trajectory identification; descriptive, not calibration')
    return result

def paired(candidate,reference,expected):
    a={r['case_id']:r for r in candidate};b={r['case_id']:r for r in reference};keys=sorted(a.keys()&b.keys())
    if not keys:return None
    for key in keys:
        if a[key]['reset_seed']!=b[key]['reset_seed'] or a[key]['metrics']['assigned_steps']!=b[key]['metrics']['assigned_steps']:raise ValueError('Pair physical reset or assigned duration mismatch')
    differences=np.asarray([a[k]['metrics']['return']-b[k]['metrics']['return'] for k in keys])
    return dict(cases=len(keys),case_ids=keys,expected_cases=expected,complete_case_coverage=len(keys)==expected,wins=int((differences>0).sum()),losses=int((differences<0).sum()),ties=int((differences==0).sum()),candidate_mean_return=float(np.mean([a[k]['metrics']['return'] for k in keys])),reference_mean_return=float(np.mean([b[k]['metrics']['return'] for k in keys])),mean_return_difference=float(differences.mean()),all_episodes_complete=all(a[k]['metrics']['completed_full_duration'] and b[k]['metrics']['completed_full_duration'] for k in keys),differences=[dict(case_id=k,return_difference=float(d)) for k,d in zip(keys,differences)],scope='Exact observed-case intersection; incomplete ordered cohorts are labelled and not representative')

def summarize(rows):
    configs=read(CONFIG/'cases.json');keys=[(r['study'],r['arm'],r['method'],r['case_id']) for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate study/arm/method/case; supply each run once')
    groups=[];pairs=[];arms=[]
    for study,arm,method in sorted(set(k[:3] for k in keys)):
        subset=[r for r in rows if (r['study'],r['arm'],r['method'])==(study,arm,method)]
        for family in ['all']+sorted({r['family'] for r in subset}):
            selected=subset if family=='all' else [r for r in subset if r['family']==family]
            expected=len(configs[study]) if family=='all' else sum(c['family']==family for c in configs[study])
            groups.append(dict(study=study,arm=arm,method=method,family=family,**aggregate(selected,expected)))
    for study,arm in sorted(set(k[:2] for k in keys)):
        a=[r for r in rows if (r['study'],r['arm'],r['method'])==(study,arm,'genaif')]
        for reference in ('history','dadp','oracle','reward_only'):
            b=[r for r in rows if (r['study'],r['arm'],r['method'])==(study,arm,reference)]
            for family in ['all']+sorted({r['family'] for r in a}):
                aa=a if family=='all' else [r for r in a if r['family']==family];bb=b if family=='all' else [r for r in b if r['family']==family]
                expected=len(configs[study]) if family=='all' else sum(c['family']==family for c in configs[study]);p=paired(aa,bb,expected)
                if p:pairs.append(dict(study=study,arm=arm,candidate='genaif',reference=reference,family=family,**p))
    for method in METHODS:
        base=[r for r in rows if (r['study'],r['arm'],r['method'])==('adaptation','zero',method)]
        for arm in ('replay_only','expert_1000'):
            target=[r for r in rows if (r['study'],r['arm'],r['method'])==('adaptation',arm,method)]
            for cohort in ('all','target','retention'):
                take=lambda x:x if cohort=='all' else [r for r in x if str(r['case_key']).startswith(cohort)]
                expected=len(configs['adaptation']) if cohort=='all' else sum(str(c.get('case_key','')).startswith(cohort) for c in configs['adaptation']);p=paired(take(target),take(base),expected)
                if p:arms.append(dict(method=method,arm=arm,reference_arm='zero',cohort=cohort,**p))
    coverage=[]
    for study in sorted({r['study'] for r in rows}):
        expected_methods=['genaif'] if study=='gravity_sweep' else ['genaif','reward_only'] if study.startswith('preference') else list(METHODS)
        expected_arms=['zero','replay_only','expert_1000'] if study=='adaptation' else ['zero']
        for arm in expected_arms:
            present={g['method']:g for g in groups if g['study']==study and g['arm']==arm and g['family']=='all'}
            coverage.append(dict(study=study,arm=arm,expected_methods=expected_methods,missing_methods=sorted(set(expected_methods)-present.keys()),complete=all(m in present and present[m]['complete_case_coverage'] for m in expected_methods)))
    return dict(status='complete' if all(c['complete'] for c in coverage) else 'partial',source='Recomputed from saved native traces; stored row metrics ignored',expert_reference='Eight exact stationary expert training means; assigned scheduled duration weighting, then case-level percentages averaged. Not applied to different physical gravity banks.',coverage=coverage,study_notes={s:STUDY_NOTES[s] for s in sorted({r['study'] for r in rows})},groups=groups,paired_genaif=pairs,adaptation_arm_comparisons=arms,preference_decision_selection=select_preference_decision(rows),cases=rows,model_queries=0,physical_transitions=0,optimizer_updates=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expert-reference',type=Path,help='CSV from work2.support.expert_reference; omitted expert-relative percentages are null')
    a=p.parse_args();reference=expert_reference(a.expert_reference) if a.expert_reference else None
    rows=[r for directory in a.input for r in load_run(directory, reference)]
    if not rows:raise SystemExit('No completed saved episodes')
    value=summarize(rows)
    value['expert_reference_input']=str(a.expert_reference.resolve()) if a.expert_reference else None
    if reference is None:value['expert_reference']='Not supplied; expert-relative percentages are omitted.'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(f'Recomputed {len(rows)} episodes; saved {a.output}')
if __name__=='__main__':main()
