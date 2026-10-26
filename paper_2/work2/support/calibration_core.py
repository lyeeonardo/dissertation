"""Original fixed 256-CAL query with original common noise schedule."""
QUERY_SEED_BASE=398_000_000

def query_calibration(engine,references,save_batch):
    import numpy as np
    import torch
    from work2.native.cpu_denoising_data import read_batch
    from work2.native.optimized_proposal_control import _noise
    parts=[]
    for start in range(0,len(references),32):
        refs=references[start:start+32]
        if any(r['role']!='calibration' for r in refs):
            raise ValueError('Model calibration query attempted outside CAL')
        batch=read_batch(refs)
        history={k:torch.as_tensor(v,device=engine.device) for k,v in batch['history'].items()}
        actions=torch.as_tensor(batch['actions'],device=engine.device)
        outcomes=torch.as_tensor(batch['next_observations'],device=engine.device)
        rewards=torch.as_tensor(batch['rewards'],device=engine.device,dtype=torch.float32)
        seeds=[QUERY_SEED_BASE+10000*(start+i) for i in range(len(refs))]
        noise=torch.stack([_noise((4,2,18),seed,'actual',engine.device) for seed in seeds])
        errors=engine.denoising_errors(history,actions,outcomes,rewards,noise=noise).detach().cpu().numpy()
        labels=np.asarray(batch['labels_evaluator_only'],np.int64)
        if errors.shape!=(len(refs),8) or not np.isfinite(errors).all() or np.any(errors<0):
            raise ValueError('Invalid native CAL denoising errors')
        arrays=dict(errors=errors,labels=labels,query_seeds=np.asarray(seeds,np.int64))
        save_batch(start,arrays,refs)
        parts.append(arrays)
    return {k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
