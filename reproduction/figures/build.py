#!/usr/bin/env python3
"""Draw numerical figures from explicitly supplied experiment outputs."""

import argparse
import importlib
from pathlib import Path


PLOTTERS = {
    "work1": "work1",
    "work2-adaptation": "work2_adaptation",
    "work2-gravity": "work2_gravity",
    "work3-benchmark": "work3_benchmark",
    "work3-forecast": "work3_forecast",
    "work3-belief": "work3_belief",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=PLOTTERS)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing the numerical inputs described in README.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for generated figures; must differ from the input directory",
    )
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if (
        source == destination
        or destination in source.parents
        or source in destination.parents
    ):
        parser.error(
            "Input and output directories must be separate, non-nested locations"
        )
    destination.mkdir(parents=True, exist_ok=True)
    plotter = importlib.import_module("plotters." + PLOTTERS[args.figure])
    plotter.render(source, destination)
    print(f"Wrote {args.figure} figures to {destination}")


if __name__ == "__main__":
    main()
