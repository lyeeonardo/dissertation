"""Collect the fixed one-seed Han paper-style data protocol into ROOT/data."""

import argparse

from pathlib import Path

from .paper_protocol import collect_paper_dataset

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    collect_paper_dataset(args.root / "data", workers=args.workers, seed=args.seed)

if __name__ == "__main__":
    main()
