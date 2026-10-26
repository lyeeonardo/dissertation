"""Original eight-condition SAC teacher protocol used by the supplied 1×/1.5× R corpus."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--condition',type=int,choices=range(8),required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    recipe=dict(output=a.output.resolve(),condition=a.condition,env_id='HalfCheetah-v5',steps=1000000,max_steps=2000000,
                workers=4,warmup=10000,eval_interval=100000,eval_episodes=5,qualification_episodes=10)
    if not a.execute: print({**recipe,'execution':'metadata only; --execute starts native SAC fitting'});return
    from work2.native.experts import train
    destination=recipe['output']/'experts'/'HalfCheetah-v5'/f'condition_{a.condition}'
    if destination.exists(): raise FileExistsError('Use a fresh output; historical resumable trainer is not automatically resumed')
    train(SimpleNamespace(**recipe))
    from work2.io import read, save, identity
    receipt = read(destination/'result.json')
    receipt.update(condition=a.condition, env_id=recipe['env_id'],
                   selected_policy_identity=identity(destination/'selected.zip'))
    save(destination/'result.json', receipt)
    if receipt.get('qualification', {}).get('qualified') is not True:
        raise RuntimeError('Teacher qualification failed; preserve this result and do not use it for collection')

if __name__ == '__main__': main()
