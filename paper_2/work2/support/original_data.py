"""Collect/finalize the original R corpus for the earlier evidence/IG checkpoint; no excluded E-arm dataset."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('trunks','branches','finalize'))
    p.add_argument('--root',type=Path,required=True,help='Work directory containing qualified experts/HalfCheetah-v5/condition_{0..7}')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute: print(dict(stage=a.stage,root=str(a.root),corpus='original1x/1.5x R',execution='metadata only'));return
    from work2.native.data import collect_trunks,collect_branches,finalize_dataset
    {'trunks':collect_trunks,'branches':collect_branches,'finalize':finalize_dataset}[a.stage](a.root.resolve())

if __name__ == '__main__': main()
