"""Recompute information-gain contrasts from generated case_metrics.json records."""
import argparse
from pathlib import Path
import numpy as np
from work2.io import read, save
from work2.native.analysis import paired_case_contrast

FAMILIES = ('stationary', 'single_factor', 'repeated_factor')


def revised_contrasts(rows):
    """Selected pair of the original policy_preservation_metrics bootstrap, same RNG/order."""
    methods = ('revised_efe','revised_vfe')
    indexed = {(r['method'],r['case_id']):r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != {(m,i) for m in methods for i in range(24)}:
        raise ValueError('Exactly 24 complete uniquely paired revised scenarios are required')
    cases = [indexed[methods[0],i] for i in range(24)]
    for case in cases:
        other=indexed[methods[1],case['case_id']]
        if any(case[k] != other[k] for k in ('family','initial_condition')):
            raise ValueError('Paired scenario identity differs')
    rng=np.random.default_rng(13)
    columns=[]
    for family in FAMILIES:
        ids=np.array([c['case_id'] for c in cases if c['family']==family])
        if len(ids)!=8: raise ValueError('Revised study requires eight cases per family')
        columns.append(ids[rng.integers(len(ids),size=(5000,len(ids)))])
    draws=np.concatenate(columns,axis=1)
    column_families=np.repeat(FAMILIES,8)
    result=[]
    for metric in ('return','event_reward_150'):
        allowed=[f for f in FAMILIES if not (metric=='event_reward_150' and f=='stationary')]
        ids=[c['case_id'] for c in cases if c['family'] in allowed]
        delta=np.zeros(24,np.float64)
        for cid in ids:
            a,b=(indexed[m,cid][metric] for m in methods)
            if a is None or b is None or not np.isfinite([a,b]).all(): raise ValueError('Missing finite paired metric')
            delta[cid]=a-b
        selected=draws[:,np.isin(column_families,allowed)]
        result.append(dict(first=methods[0],second=methods[1],metric=metric,
            estimate=float(delta[ids].mean()),ci95=np.quantile(delta[selected].mean(1),[.025,.975]).tolist(),
            cases=len(ids),strata=len(allowed),draws=5000,seed=13,unit='paired whole scenario',
            scope='Case variation conditional on one trained seed-13 model set'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='case_metrics.json from work2.support.information_run')
    p.add_argument('--study',choices=('earlier','revised'),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); source=read(a.input)
    if not isinstance(source,list):raise ValueError('Expected a case_metrics.json list')
    if a.study=='earlier':
        cases=read(Path(__file__).resolve().parents[1]/'configs/information_protocols.json')['earlier']['cases']
        expected={(method,case['case_id']) for method in ('efe_full','vfe_full') for case in cases}
        if len(source)!=240 or {(r['method'],r['case_id']) for r in source}!=expected:
            raise ValueError('Earlier comparison requires both complete uniquely paired 120-case methods')
        contrasts=[paired_case_contrast(source,'efe_full','vfe_full','event_reward_150')]
    else:contrasts=revised_contrasts(source)
    result=dict(study=a.study,contrasts=contrasts,
        scope='Recomputed case summaries; bootstrap does not rerun controllers or estimate training-seed variation',
        physical_transitions=0,model_queries=0)
    if a.output.exists(): raise FileExistsError(a.output)
    save(a.output,result)
    print(f'Saved support analysis to {a.output}')

if __name__ == '__main__': main()
