"""Exact retained alarm-boundary and clearing readouts from the transient study."""

import numpy as np

from studies.monitor import THRESHOLD

def flags(scores):
    counts=np.zeros(len(scores),dtype=np.int64)
    for i,x in enumerate(scores):counts[i]=(counts[i-1] if i else 0)+1 if x>THRESHOLD else 0
    flag=counts>=4
    onset=flag & ~np.r_[False,flag[:-1]]
    clear=~flag & np.r_[False,flag[:-1]]
    return counts,flag,onset,clear

def alarm_episodes(flag):
    changes=np.diff(np.r_[False,flag,False].astype(int))
    return [dict(first_step=int(a),last_step=int(b-1),first_available_boundary=int(a+1),
        last_flagged_boundary=int(b),clear_boundary=int(b+1) if b<len(flag) else None,
        first_available_day=(a+1)/96.,clear_day=(b+1)/96. if b<len(flag) else None,
        flagged_observations=int(b-a),right_censored=bool(b==len(flag)))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]

def first_hit(mask,start):
    hits=np.flatnonzero(np.asarray(mask)&(np.arange(len(mask))>=start))
    return dict(found=True,step=int(hits[0]),available_boundary=int(hits[0]+1),
        delay_hours=float((hits[0]+1-start)/4.)) if len(hits) else dict(found=False,right_censored=True,
            delay_hours=None,censor_hours=float((len(mask)-start)/4.))

def quiet_observations(flag,restore,count=97):
    """A fixed number of post-restoration clear observations; no detector reset."""
    streak=0
    for i in range(restore,len(flag)):
        streak=0 if flag[i] else streak+1
        if streak>=count:
            first=i-count+1
            return dict(found=True,observations=count,first_step=int(first),last_step=int(i),
                interval_start_boundary=int(first),interval_start_day=float(first/96.),
                first_clear_available_boundary=int(first+1),confirmation_boundary=int(i+1),
                first_available_day=float((first+1)/96.),confirmation_day=float((i+1)/96.),
                interval_start_delay_hours=float((first-restore)/4.),
                first_clear_delay_hours=float((first+1-restore)/4.),confirmation_delay_hours=float((i+1-restore)/4.),
                available_endpoint_span_hours=float((count-1)/4.),observation_interval_hours=float(count/4.))
    return dict(found=False,observations=count,right_censored=True,censor_hours=float((len(flag)-restore)/4.))

def recovery(flag,restore,scores):
    quiet=quiet_observations(flag,restore,97)
    if quiet['found']:
        selected=slice(quiet['first_step'],quiet['last_step']+1)
        quiet['threshold_exceedances']=int(np.sum(scores[selected]>THRESHOLD))
        quiet['maximum_score']=float(np.max(scores[selected]))
    low=quiet_observations(np.asarray(scores)>THRESHOLD,restore,4)
    onset=np.asarray(flag)&~np.r_[False,np.asarray(flag)[:-1]]
    clear=~np.asarray(flag)&np.r_[False,np.asarray(flag)[:-1]]
    active=bool(flag[restore-1])
    return dict(active_at_restoration=active,
        first_alarm_clearing=first_hit(clear,restore) if active else dict(already_clear_at_restoration=True,delay_hours=0.),
        first_unflagged_observation=first_hit(~np.asarray(flag),restore),
        four_low_score_observations=low,
        sampled_24h_quiet=quiet,
        post_restoration_onsets=first_hit(onset,restore),
        post_restoration_onset_count=int(onset[restore:].sum()),
        post_restoration_flagged_observations=int(np.sum(flag[restore:])),
        post_restoration_observations=len(flag)-restore,
        final_day_flagged_observations=int(np.sum(flag[-96:])),
        final_day_onsets=int(onset[-96:].sum()),
        all_alarm_episodes=alarm_episodes(flag))
