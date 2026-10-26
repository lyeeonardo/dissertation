"""Recompute the exact same-root 101-versus-111 friction probe (73/110 saved roots)."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import h5py
from work2.io import read, save
from .friction_core import branch, identify, equal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-data', type=Path, required=True)
    p.add_argument('--refreshed-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    before_anchors=read(a.original_data/'splits/paired_anchors.json')['anchors']
    after_anchors=read(a.refreshed_data/'splits/paired_anchors.json')['anchors']
    if before_anchors != after_anchors: raise ValueError('Exact anchor identities differ')
    rows=[]
    with ExitStack() as stack:
        before=[stack.enter_context(h5py.File(a.original_data/'data/HalfCheetah-v5'/f'cell_{c}.h5','r')) for c in range(8)]
        after=[stack.enter_context(h5py.File(a.refreshed_data/'data/HalfCheetah-v5'/f'cell_{c}.h5','r')) for c in range(8)]
        for anchor in before_anchors:
            name=f"paired/{anchor['anchor_id']:04d}_3"
            old=[branch(f[name]) for f in before]; new=[branch(f[name]) for f in after]
            matched, checks=identify(old,new)
            rows.append(dict(anchor_id=anchor['anchor_id'],exact_root_prefix_action_match=matched,
                next_observation_identical=equal(new[5]['next_observation'],new[7]['next_observation']) if matched else None,
                checks=checks))
    matched=[r for r in rows if r['exact_root_prefix_action_match']]
    if not matched: raise ValueError('No exactly matched probe roots')
    if a.output.exists(): raise FileExistsError(a.output)
    save(a.output,dict(conditions=['101','111'],probe_chunk=3,matched_roots=len(matched),
        identical_next_observations=sum(r['next_observation_identical'] for r in matched),
        scope='Exact shape/dtype/value equality, 17 public next-observation coordinates, reward excluded',
        physical_transitions=0,model_queries=0,rows=rows))

if __name__ == '__main__': main()
