"""Summarize full included episodes; implementation smokes never enter results."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from work3_paths import ROOT

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'outputs');a=p.parse_args();rows=[]
    for path in sorted(a.root.glob('*/*/result.json')):
        r=json.loads(path.read_text())
        if r.get('implementation_smoke') or r.get('completed_decisions')!=1344:continue
        with np.load(path.parent/'arrays.npz',allow_pickle=False) as d:
            row=dict(archived_inputs=r.get('archived_inputs',True),study=r['study'],case=r['case_id'],method=r['method'],cost=float(d['costs'].mean()),oxygen_rmse_to2=float(np.sqrt(np.mean((d['privileged'][1:,59]-2.)**2))),mean_decision_seconds=float(d['decision_seconds'].mean()))
            row['cost_plus_preference']=float(np.mean(d['costs']/12220.5947265625+.01*.5*((d['privileged'][1:,59]-2.)/.5)**2))
            trace=path.parent/'belief_forecast.npz'
            if trace.exists():
                with np.load(trace,allow_pickle=False) as b:
                    from bsm1_offline_benchmark.stock_shared_data import PUBLIC_INDICES
                    prediction=np.einsum('nk,nkd->nd',b['weights'],b['forecast']);observed=d['privileged'][1:];idx=PUBLIC_INDICES[:41]
                    row['forecast_public41_rmse']=float(np.sqrt(np.mean(((prediction[:,idx]-observed[:,idx])/b['train_state_scale'][idx])**2)))
                    row['persistence_public41_rmse']=float(np.sqrt(np.mean(((d['privileged'][:-1,idx]-observed[:,idx])/b['train_state_scale'][idx])**2)))
                    corrected=np.einsum('nk,nkd->nd',b['weights'],b['particles'])
                    for label,index in [('oxygen5',59),('nitrate2',21)]:row[label+'_corrected_rmse']=float(np.sqrt(np.mean((corrected[:,index]-d['privileged'][:-1,index])**2)))
            monitor=path.parent/'monitor.json'
            if monitor.exists():
                records=json.loads(monitor.read_text());flags=np.array([v['persistent_prediction_mismatch'] for v in records]);row['alarms_after_day7']=int(flags[672:].sum())
            rows.append(row)
    if not rows:raise SystemExit('No complete scientific episodes found')
    fields=list(dict.fromkeys(k for row in rows for k in row));a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    paired=[]
    for study,on,off in [('benchmark','genaif','public_pi'),('correction','genaif','uncorrected'),('forecast-selection','scoring_on','scoring_off')]:
        by={(r['case'],r['method']):r for r in rows if r['study']==study}
        effects=[dict(case=c,percent=100*(r['cost']/by[c,off]['cost']-1)) for (c,m),r in by.items() if m==on and (c,off) in by]
        if effects:paired.append(dict(study=study,pairs=effects,equal_case_mean_percent=float(np.mean([v['percent'] for v in effects])),expected_cases=6,complete=len(effects)==6))
    (a.root/'paired_effects.json').write_text(json.dumps(paired,indent=2)+'\n')
    from analyze_appendix import summarize
    (a.root/'appendix_readouts.json').write_text(json.dumps(summarize(a.root),indent=2)+'\n')
    print(json.dumps(paired,indent=2))
if __name__=='__main__':main()
