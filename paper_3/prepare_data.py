"""Rebuild the retained training bank from official inflows, or process supplied source arrays."""
from pathlib import Path
import argparse
import json
from work3_paths import ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('collect-original','prepare-shared','collect-shared','freeze','calibrate'))
    parser.add_argument('--root',type=Path,default=ROOT/'outputs')
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--data',type=Path,help='Explicit frozen data directory for calibration')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();root=args.root.resolve();collection=root/'collection'
    if args.stage=='collect-original':
        from bsm1_offline_benchmark.paper_protocol import collect_paper_dataset
        collect_paper_dataset(collection/'original/data',workers=args.workers,seed=17)
    elif args.stage=='prepare-shared':
        from bsm1_offline_benchmark.paper_shared_dataset import prepare
        parent_root=collection/'parent_protocol';(parent_root/'data').mkdir(parents=True,exist_ok=True)
        (parent_root/'data/manifest.json').write_text((ROOT/'configs/collection_parents.json').read_text())
        prepare(collection/'shared',collection/'original',parent_root,seed=17)
    elif args.stage=='collect-shared':
        from bsm1_offline_benchmark.paper_shared_dataset import collect
        collect(collection/'shared',workers=args.workers)
    elif args.stage=='freeze':
        from bsm1_offline_benchmark.stock_shared_data import freeze
        freeze(root/'prepared',collection/'shared/data')
    else:
        from bsm1_offline_benchmark.stock_joint_diagnostics import run
        if not args.checkpoint:parser.error('calibrate requires --checkpoint')
        run(args.data.resolve() if args.data else root/'prepared/data',args.checkpoint,root/'calibration',passes=10,device=args.device)
if __name__=='__main__':main()
