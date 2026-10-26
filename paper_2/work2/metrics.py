"""Observed-only recovery and assigned-window reward; no missing speed fabrication."""
import numpy as np
from work2.native.reduced_benchmark_metrics import recovery, timing_metrics


def window(arrays,start,stop,reference):
    rewards=np.asarray(arrays['rewards'],float)[start:stop]
    speed=np.asarray(arrays['speed'],float)[start:stop]
    n=len(speed);assigned=stop-start
    return dict(start_step=start,stop_step_exclusive=stop,start_seconds=start*.05,stop_seconds=stop*.05,
                assigned_steps=assigned,observed_steps=n,missing_steps=assigned-n,reached=n>0,complete=n==assigned,
                reward_sum=float(rewards.sum()),reward_per_assigned_step=float(rewards.sum()/assigned),
                observed_mean_speed_m_s=float(speed.mean()) if n else None,
                reference_speed_m_s=reference,threshold_speed_m_s=.9*reference)


def episode_metrics(suite,case,arrays,method,plan):
    n=len(arrays['rewards']);refs=plan['reference_speeds'];events=[]
    for event in case['events']:
        start=event['step'];ref=refs[str(event['condition'])]
        events.append(dict(**event,**window(arrays,start,start+150,ref),
                           recovery=recovery(arrays['speed'][start:start+150],ref)))
    row=dict(suite=suite,method=method,case_id=case['case_id'],family=case['family'],
             initial_condition=case['initial_condition'],native_steps=n,assigned_steps=case['steps'],
             native_seconds=n*.05,completed_full_duration=n==case['steps'],
             native_termination=bool(arrays['terminated'][-1]),native_truncation=bool(arrays['truncated'][-1]),
             full_run_return=float(np.asarray(arrays['rewards'],float).sum()),
             full_run_observed_mean_speed_m_s=float(np.asarray(arrays['speed'],float).mean()),
             inverted_steps=int(np.asarray(arrays['inverted'],bool).sum()),
             inverted_fraction=float(np.asarray(arrays['inverted'],bool).mean()),
             events=events,scheduled_events=len(events),reached_events=sum(e['reached'] for e in events),
             unreached_switches=sum(not e['reached'] for e in events),
             confirmed_events=sum(e['recovery']['recovered'] for e in events),
             sustained_events=sum(e['recovery']['sustained'] for e in events),
             timing=timing_metrics(arrays,n,.05))
    row['return']=row['full_run_return']
    row.update(event_reward_150=float(np.mean([e['reward_sum'] for e in events])) if events else None,
               nonrecovery_rate=float(np.mean([not e['recovery']['recovered'] for e in events])) if events else None,
               sustained_recovery_rate=float(np.mean([e['recovery']['sustained'] for e in events])) if events else None,
               recovery_time_steps=float(np.mean([e['recovery']['capped_delay_steps'] for e in events])) if events else None,
               recovery_time_seconds=float(np.mean([e['recovery']['capped_delay_seconds'] for e in events])) if events else None)
    if 'beliefs' in arrays:
        belief=np.asarray(arrays['beliefs'],float)[1:];truth=np.asarray(arrays['conditions'],int)
        row['belief']=dict(alignment='beliefs[i+1] after observing transition i, compared with conditions[i]',
                           posterior_map_accuracy=float((belief.argmax(-1)==truth).mean()),
                           posterior_true_mass=float(belief[np.arange(n),truth].mean()),
                           observed_steps=n,surrogate_not_calibrated_probability=True)
    return row
