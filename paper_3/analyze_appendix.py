"""Derive Appendix C readouts from generated physical traces and saved plan banks.

No model or simulator query is executed. The saved-bank alternatives are never
executed, and their target scores are not claims about physical tracking.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from bsm1_offline_benchmark.stock_shared_data import PUBLIC_INDICES
from studies.monitor_readout import flags, first_hit, recovery
from work3_paths import ROOT

HIDDEN=np.setdiff1d(np.arange(159),PUBLIC_INDICES)

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def stats(values):
    return dict(median=float(np.median(values)),q95=float(np.quantile(values,.95)))

def settling_readout(actual,belief,nominal,*,restore=None,scores=None):
    """Decision-current hidden states and next-boundary physical comparisons."""
    scale=belief['train_state_scale'];idx=PUBLIC_INDICES[:41]
    current=np.einsum('nk,nkd->nd',belief['weights'],belief['particles'])
    forecast=np.einsum('nk,nkd->nd',belief['weights'],belief['forecast'])
    if scores is None:scores=np.sqrt(np.mean(((forecast[:,idx]-actual['privileged'][1:,idx])/scale[idx])**2,axis=1))
    hidden=np.sqrt(np.mean(((current[:,HIDDEN]-actual['privileged'][:-1,HIDDEN])/scale[HIDDEN])**2,axis=1))
    difference=np.sqrt(np.mean(((actual['privileged'][1:,idx]-nominal['privileged'][1:,idx])/scale[idx])**2,axis=1))
    _,alarm,_,_=flags(scores)
    result=dict(final_day_forecast=stats(scores[-96:]),final_day_hidden104=stats(hidden[-96:]),final_day_physical_plant41=stats(difference[-96:]),first_alarm_after_day7=first_hit(alarm,672),postchange_flagged=int(alarm[672:].sum()))
    if restore is not None:result['recovery']=recovery(alarm,restore,np.asarray(scores))
    return result

def rescore_record(record,targets=(1.5,2.,2.5)):
    oxygen=np.asarray(record['candidate_do5_forecasts'],dtype=np.float64)
    rewards=np.asarray(record['candidate_reward_forecasts'],dtype=np.float64)
    plans=np.asarray(record['candidate_actions'],dtype=np.float64)
    weights=np.asarray(record['prior_weights'],dtype=np.float64)
    discounts=np.asarray(record['forecast_discounts'],dtype=np.float64)
    info=np.asarray(record['information_l1'],dtype=np.float64)
    if oxygen.shape!=(3,8,30) or rewards.shape!=oxygen.shape or plans.shape!=(3,30,2):raise ValueError('Complete saved native 3x8x30 bank required')
    np.testing.assert_allclose(weights.sum(),1.,atol=1e-12,rtol=0)
    np.testing.assert_allclose(discounts,.99**np.arange(30),atol=1e-12,rtol=0)
    returns=np.einsum('ckh,k,h->c',rewards,weights,discounts)
    native=-returns-.215947119414*info
    base=int(native.argmin())
    penalties=np.asarray([np.einsum('ckh,k,h->c',.5*((oxygen-target)/.5)**2,weights,discounts) for target in targets])
    winners=(native[None,:]+.01*penalties).argmin(axis=1)
    means=np.einsum('ckh,k,h->c',oxygen,weights,discounts)/discounts.sum()
    rms=np.sqrt(penalties*2*.5**2/discounts.sum())
    return [dict(target=float(t),selected_candidate=int(w),native_candidate=base,predicted_oxygen_mean=float(means[w]),predicted_return_change=float(returns[w]-returns[base]),predicted_target_rms=float(rms[i,w]),predicted_target_rms_reduction=float(rms[i,base]-rms[i,w]),first_command_changed=bool(np.max(np.abs(plans[w,0]-plans[base,0]))>1e-7),executed=False) for i,(t,w) in enumerate(zip(targets,winners))]

def preference_bank_readout(records):
    """Count exact selected first-command changes on the executed preference banks."""
    changed=0
    for r in records:
        returns=np.asarray(r['predicted_returns']);info=np.asarray(r['information_l1']);penalty=np.asarray(r['preference_penalties']);commands=np.asarray(r['candidate_first_actions'])
        native=-returns-.215947119414*info
        selected=int((native+.01*penalty).argmin());without=int(native.argmin())
        if selected!=int(r['selected_candidate']):raise ValueError('Saved executed preference winner does not match its score')
        changed+=int(np.max(np.abs(commands[selected]-commands[without]))>1e-7)
    return dict(changed_first_commands=changed,decisions=len(records),comparison='Same recorded preference-controller bank with and without oxygen penalty; alternative is unexecuted')

def summarize(root):
    result={}
    nominal_path=root/'settling/validation_p000__nominal/arrays.npz'
    if nominal_path.exists():
        nominal=arrays(nominal_path);table={}
        for method,restore in [('nominal',None),('settling_6h',696),('settling_24h',768),('settling_permanent',None)]:
            folder=root/'settling'/f'validation_p000__{method}'
            if not (folder/'belief_forecast.npz').exists():continue
            actual=arrays(folder/'arrays.npz')
            if len(actual['costs'])!=1344:continue
            belief=arrays(folder/'belief_forecast.npz');monitor=folder/'monitor.json';scores=None
            if monitor.exists():scores=np.asarray([r['public41_residual_rmse'] for r in json.loads(monitor.read_text())])
            table[method]=settling_readout(actual,belief,nominal,restore=restore,scores=scores)
        result['settling_final_day_and_clearing']=table
    prefs={}
    for index in range(3):
        path=root/'preference'/f'validation_p00{index}__oxygen_preference/controller_diagnostics.json'
        if path.exists():
            records=json.loads(path.read_text())
            if len(records)==1344:prefs[f'validation_p00{index}']=preference_bank_readout(records)
    if prefs:
        result['preference_bank_changes']=dict(cases=prefs,changed_first_commands=sum(v['changed_first_commands'] for v in prefs.values()),decisions=sum(v['decisions'] for v in prefs.values()),complete_cases=len(prefs)==3)
    rain=root/'preference/validation_p001__reward_reference/controller_diagnostics.json'
    if rain.exists():
        records=json.loads(rain.read_text())
        if len(records)==1344:result['rain_saved_decision557']=dict(step=557,selection='Previously selected illustration; no reselection on outcomes',alternatives=rescore_record(records[557]))
    monitoring=[]
    for study in ('settling','oxygen-transfer','sensor-offset'):
        for path in sorted((root/study).glob('*/monitor.json')):
            records=json.loads(path.read_text())
            if len(records)!=1344:continue
            score=np.asarray([r['public41_residual_rmse'] for r in records])
            counts,alarm,onsets,clears=flags(score)
            channel=np.asarray([r['channel_squared_contributions'] for r in records])
            denom=float(channel[672:].sum())
            row=dict(study=study,job=path.parent.name,post_day7_flags=int(alarm[672:].sum()),post_day7_observations=672,first_alarm_after_day7=first_hit(alarm,672),settler_tss_fraction_of_postchange_squared_residual=float(channel[672:,25:35].sum()/denom) if denom else None)
            if study=='sensor-offset':
                row.update(biased_packet_observation_boundaries=[672,767],biased_packet_flags=int(alarm[671:767].sum()),biased_packet_observations=96)
            monitoring.append(row)
    if monitoring:result['monitor_limits']=monitoring
    choices=[]
    for i in range(6):
        name=f'stock_test_p00{i}__genaif';path=root/'benchmark'/name/'controller_diagnostics.json'
        if not path.exists():path=root/'correction'/name/'controller_diagnostics.json'
        if not path.exists():continue
        records=json.loads(path.read_text())
        if len(records)!=1344:continue
        refined=information_changes=0
        for r in records:
            commands=np.asarray(r['candidate_first_actions']);returns=np.asarray(r['predicted_returns']);info=np.asarray(r['information_l1'])
            selected=int((-returns-.215947119414*info).argmin());reward=int(returns.argmax())
            refined+=int(selected!=0)
            information_changes+=int(np.max(np.abs(commands[selected]-commands[reward]))>1e-7)
        choices.append(dict(case=name,decisions=len(records),refinement_selected=refined,information_changed_first_command=information_changes))
    if choices:result['final_candidate_selection']=dict(cases=choices,decisions=sum(r['decisions'] for r in choices),refinement_selected=sum(r['refinement_selected'] for r in choices),information_changed_first_command=sum(r['information_changed_first_command'] for r in choices),complete_cases=len(choices)==6)
    # Same-boundary prior and corrected means from all six final controller traces.
    corrections=[]
    for i in range(6):
        name=f'stock_test_p00{i}__genaif';folder=root/'correction'/name
        if not (folder/'controller_diagnostics.json').exists():folder=root/'benchmark'/name
        if not (folder/'controller_diagnostics.json').exists():continue
        rec=json.loads((folder/'controller_diagnostics.json').read_text())
        if len(rec)!=1344 or 'prior_mean' not in rec[0]:continue
        truth=arrays(folder/'arrays.npz')['privileged'][:-1,[59,21]]
        before=np.asarray([r['prior_mean'] for r in rec]);after=np.asarray([r['post_mean'] for r in rec])
        corrections.append(dict(case=name,decisions=len(rec),incoming_sse=((before-truth)**2).sum(0).tolist(),corrected_sse=((after-truth)**2).sum(0).tolist()))
    if corrections:
        count=sum(r['decisions'] for r in corrections)
        result['correction_rmse']=dict(coordinates=['oxygen5','nitrate2'],cases=corrections,decisions=count,incoming_rmse=np.sqrt(np.sum([r['incoming_sse'] for r in corrections],axis=0)/count).tolist(),corrected_rmse=np.sqrt(np.sum([r['corrected_sse'] for r in corrections],axis=0)/count).tolist(),complete_cases=len(corrections)==6)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'outputs');a=p.parse_args()
    result=summarize(a.root)
    if not result:raise SystemExit('No complete included episode/bank records found')
    (a.root/'appendix_readouts.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
