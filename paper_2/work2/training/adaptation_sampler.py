"""Parent-isolated CPU windows; importing reads no data and starts no science.

Physical strata are (gravity_scale,condition), never just the reused integer
condition slot. Native eligibility is preserved at every future action. Target
prefix budgets limit observations, reward-to-go and every training window.
"""
from __future__ import annotations
from collections import Counter
import copy
import numpy as np

GROUPS = ('old1','new3','expert','random','weak')


def arm_rows(pools, arm, split='train'):
    source=pools['source_train' if split=='train' else 'source_dev']
    groups={'old1':[dict(r) for r in source if float(r['gravity_scale'])==1],
            'new3':[dict(r) for r in source if float(r['gravity_scale'])==3]}
    targets=pools['target_train' if split=='train' else 'target_dev']
    for kind in ('expert','random','weak'):
        rows=[dict(r) for r in targets[kind]]
        if split=='train':
            budget=arm[kind+'_steps']; assert budget%2==0
            if budget == 0:
                rows=[]
            else:
                assert len(rows)==2
            rows=[dict(r,included_length=min(int(r['length']),budget//2)) for r in rows]
            rows=[r for r in rows if r['included_length']>0]
        groups[kind]=rows
    return groups


class Windows:
    def __init__(self, groups, normalization, seed=13):
        import h5py
        self.rng=np.random.default_rng(seed); self.normalization=normalization
        self.episodes=[]; self.group_indices={g:[] for g in GROUPS}; self.exposure=Counter()
        self.requested=Counter(); self.fallback=Counter(); handles={}
        try:
            for group,rows in groups.items():
                for row in rows:
                    if row['path'] not in handles: handles[row['path']]=h5py.File(row['path'],'r')
                    h=handles[row['path']][row['episode']]
                    n=int(row.get('included_length',row['length'])); assert 0<n<=int(row['length'])
                    assert len(h['actions'])==int(row['length'])
                    assert str(h.attrs['split'])==row['split'] and str(h.attrs['parent_id'])==str(row['parent_id'])
                    assert int(h.attrs['condition'])==int(row['condition'])
                    assert float(h.attrs['gravity_scale'])==float(row['gravity_scale'])
                    a={k:h[k][:n] for k in ('actions','rewards','terminated','proposal_eligible')}
                    a['observations']=h['observations'][:n+1]
                    assert a['observations'].shape==(n+1,17) and a['actions'].shape==(n,6)
                    assert all(np.isfinite(v).all() for v in a.values()) and abs(a['actions']).max()<=1.000001
                    if group=='random': assert not a['proposal_eligible'].any()
                    if group in ('old1','new3'): assert row['kind']=='expert'
                    rtg=np.zeros(n,np.float32); value=0.
                    for t in range(n-1,-1,-1): value=float(a['rewards'][t])+.997*value;rtg[t]=value
                    idx=len(self.episodes)
                    self.episodes.append(dict(row=row,group=group,n=n,arrays=a,values=rtg))
                    self.group_indices[group].append(idx)
        finally:
            for h in handles.values(): h.close()
        self.pools={}
        for group in GROUPS:
            self.pools[group]={k:{} for k in ('prediction','proposal','representation','policy')}
            for idx in self.group_indices[group]:
                ep=self.episodes[idx]; n=ep['n']; eligible=ep['arrays']['proposal_eligible'];r=ep['row']
                stratum=(float(r['gravity_scale']),int(r['condition']))
                kinds={'prediction':np.arange(n),'proposal':np.flatnonzero(eligible),
                       'representation':np.arange(16,max(16,n-1)),
                       'policy':np.asarray([t for t in range(max(0,n-3)) if eligible[t:t+4].all()],int)}
                for kind,times in kinds.items():
                    self.pools[group][kind].setdefault(stratum,[]).extend((idx,int(t)) for t in times)
            for kind,bycell in self.pools[group].items():
                self.pools[group][kind]={s:v for s,v in bycell.items() if v}
        for group in ('old1','new3'):
            for kind in ('prediction','proposal','representation','policy'):
                assert len(self.pools[group][kind])==4,(group,kind,'missing source cell')

    def count(self,group,kind): return sum(map(len,self.pools[group][kind].values()))

    def draw(self,group,kind,n,*,stratum=None):
        cells=self.pools[group][kind]
        if stratum is not None: cells={stratum:cells[stratum]}
        assert cells and n>0,(group,kind,n)
        # Source balances cells; target behavior weights proportional to included anchors.
        keys=list(cells); chosen=[keys[i%len(keys)] for i in range(n)];self.rng.shuffle(chosen)
        return [cells[key][int(self.rng.integers(len(cells[key])))] for key in chosen]

    def select(self,kind,n,target=True,group=None):
        if group is not None: return self.draw(group,kind,n)
        specs=[('target',n//2),('old1',n//4),('new3',n//4)] if target else [('old1',n//2),('new3',n//2)]
        assert sum(k for _,k in specs)==n
        result=[]
        for name,k in specs:
            self.requested[f'{kind}:{name}']+=k
            if name!='target': result.extend(self.draw(name,kind,k));continue
            candidates=[g for g in ('expert','random','weak') if self.count(g,kind)]
            if not candidates:
                self.fallback[kind]+=k
                result.extend(self.draw('old1',kind,k//2));result.extend(self.draw('new3',kind,k-k//2));continue
            sizes=np.asarray([self.count(g,kind) for g in candidates],float); sizes/=sizes.sum()
            selected=self.rng.choice(candidates,k,p=sizes)
            for g in candidates:
                count=int((selected==g).sum())
                if count:result.extend(self.draw(g,kind,count))
        self.rng.shuffle(result)
        return result

    def history(self,idx,t,h):
        ep=self.episodes[idx];a=ep['arrays']; obs=np.zeros((h,17),np.float32)
        actions=np.zeros((h-1,6),np.float32);rewards=np.zeros(h-1,np.float32);mask=np.zeros(h,np.float32)
        first=max(0,t-h+1);dest=h-1-(t-first)
        obs[dest:]=a['observations'][first:t+1];mask[dest:]=1
        if t>first:
            actions[dest:]=a['actions'][first:t];rewards[dest:]=a['rewards'][first:t]
        return dict(observations=obs,actions=actions,rewards=rewards,mask=mask)

    def diffusion(self,n=256,*,target=True,group=None,augment=True):
        assert n%32==0
        support=group is None or self.count(group,'proposal')>0
        roles=np.repeat(np.arange(3),[n//2,n//4,n//4]) if support else np.repeat([1,2],[n//2,n//2])
        anchors=[]
        for role in np.unique(roles): anchors.extend(self.select('proposal' if role==0 else 'prediction',int((roles==role).sum()),target,group))
        horizon=self.rng.choice([4,8,16],n);horizon[roles==1]=1
        batch=dict(history={k:[] for k in ('observations','actions','rewards','mask')},condition=[],physical_features=[],
                   role=roles,horizon=horizon,quality=(roles==0).astype(np.int64),actions=[],future_observations=[],rewards=[],terminated=[],valid=[],proposal_eligible=[])
        for j,(idx,t) in enumerate(anchors):
            ep=self.episodes[idx];a=ep['arrays'];r=ep['row'];h=self.history(idx,t,8)
            for k,v in h.items():batch['history'][k].append(v)
            c=int(r['condition']);batch['condition'].append(c);batch['physical_features'].append([(float(r['gravity_scale'])-1)/2,(c>>1)&1,c&1])
            count=min(16,ep['n']-t)
            for key,shape in [('actions',(16,6)),('future_observations',(16,17)),('rewards',(16,)),('terminated',(16,)),('valid',(16,)),('proposal_eligible',(16,))]:
                v=np.zeros(shape,np.float32)
                if key=='future_observations':v[:count]=a['observations'][t+1:t+count+1]
                elif key=='valid':v[:count]=1
                else:v[:count]=a[key][t:t+count]
                batch[key].append(v)
            self.exposure[f'diffusion_role{roles[j]}:{ep["group"]}']+=1
        batch={k:{a:np.asarray(b) for a,b in v.items()} if isinstance(v,dict) else np.asarray(v) for k,v in batch.items()}
        batch['physical_features']=batch['physical_features'].astype(np.float32)
        if augment:
            short=self.rng.random(n)<.2;length=self.rng.choice([1,2,4],n)
            batch['history']['mask']*=~short[:,None]|(np.arange(8)[None]>=8-length[:,None])
            batch['quality'][self.rng.random(n)<.2]=0
        order=self.rng.permutation(n)
        return {k:{a:b[order] for a,b in v.items()} if isinstance(v,dict) else v[order] for k,v in batch.items()}

    def representation(self,n=256,*,target=True,group=None):
        first=self.select('representation',n,target,group); states=[];actions=[];factors=[]
        mean=np.asarray(self.normalization['observation_mean'],np.float32);std=np.asarray(self.normalization['observation_std'],np.float32)
        for idx,t in first:
            ep=self.episodes[idx];r=ep['row'];stratum=(float(r['gravity_scale']),int(r['condition']))
            second=self.draw(ep['group'],'representation',1,stratum=stratum)[0]; pair=[];act=[]
            for i,s in [(idx,t),second]:
                a=self.episodes[i]['arrays'];indices=s+np.arange(-16,3)
                # Native paired-window second future observation may hit n exactly;
                # its unused final action is zero-padded as in WindowSampler.
                pair.append((a['observations'][indices]-mean)/std)
                aa=np.zeros((19,6),np.float32);valid=indices<self.episodes[i]['n'];aa[valid]=a['actions'][indices[valid]];act.append(aa)
            c=int(r['condition']); factors.append([[((c>>2)&1),((c>>1)&1),c&1]]*2)
            states.append(pair);actions.append(act);self.exposure[f'representation_pair:{ep["group"]}']+=1
        return dict(obs={'state':np.asarray(states,np.float32)},act=np.asarray(actions,np.float32),task_list=np.asarray(factors,np.float32))

    def policy(self,n=256,*,target=True,group=None,value_bounds):
        self.value_bounds=list(value_bounds)
        anchors=self.select('policy',n,target,group);out=[];weights=[]
        mean=np.asarray(self.normalization['observation_mean'],np.float32);std=np.asarray(self.normalization['observation_std'],np.float32)
        lo,hi=map(float,value_bounds); assert hi>lo
        for idx,t in anchors:
            ep=self.episodes[idx];a=ep['arrays'];indices=t+np.arange(-16,4)
            obs=a['observations'][np.maximum(indices,0)];act=np.zeros((20,6),np.float32);valid=indices>=0;act[valid]=a['actions'][indices[valid]]
            out.append(np.concatenate(((obs-mean)/std,act),-1))
            weight=np.exp(4*(float(ep['values'][t])-hi)/(hi-lo));assert np.isfinite(weight)
            weights.append([weight]);self.exposure[f'policy:{ep["group"]}']+=1
        return dict(trajectory=np.asarray(out,np.float32),weight=np.asarray(weights,np.float32))

    def weight_diagnostics(self):
        if not hasattr(self,'value_bounds'): return None
        lo,hi=self.value_bounds; result={}
        for group in GROUPS:
            anchors=[a for pool in self.pools[group]['policy'].values() for a in pool]
            values=np.asarray([self.episodes[i]['values'][t] for i,t in anchors],float)
            weights=np.exp(4*(values-hi)/(hi-lo))
            result[group]=dict(eligible_policy_windows=len(values),
                rtg_min=float(values.min()) if len(values) else None,rtg_max=float(values.max()) if len(values) else None,
                weight_min=float(weights.min()) if len(values) else None,weight_max=float(weights.max()) if len(values) else None,
                below_frozen_bound=int((values<lo).sum()),above_frozen_bound=int((values>hi).sum()))
        return dict(frozen_value_bounds=list(self.value_bounds),groups=result,clipping=False,normalization=False)

    def metadata(self):
        return dict(weight_diagnostics=self.weight_diagnostics(),rows=[dict(e['row'],included_length=e['n']) for e in self.episodes],
                    pools={g:{k:{str(c):len(v) for c,v in cells.items()} for k,cells in kinds.items()} for g,kinds in self.pools.items()},
                    exposures=dict(self.exposure),requested=dict(self.requested),unsupported_target_fallback=dict(self.fallback),
                    rtg='gamma=.997, finite included parent endpoint; no lookahead beyond selected prefix',
                    representation_factors='Native binary auxiliary factor targets; detached factor head. Pairing separately keys actual gravity and condition.')
