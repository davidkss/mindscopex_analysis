"""Analyze the Seed 1 steering stability sweep using the existing CRT pipeline."""
import argparse
from pathlib import Path

from analyze_crt_open_ended import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('results/runs'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    protocol = 'stability_smoke' if args.smoke else 'stability_sweep'
    prefix = 'study_steering_' + protocol
    output = args.output or Path('results/analysis/crt_steering_' + protocol + '_seed1')
    try:
        run(args.input_root, output, seeds=(1,), run_prefix=prefix, protocol=protocol)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Analysis failed: {exc}\n')


if __name__ == '__main__':
    main()
