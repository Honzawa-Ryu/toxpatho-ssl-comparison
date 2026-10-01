"""Analyze all checkpoints on fixed images; no GPU needed for diagnostics."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from lib.analysis.dino_dense_analysis import analyze_dense, compare_precision, write_dense_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-dir',type=Path)
    parser.add_argument('--fp32-output',type=Path,action='append')
    parser.add_argument('--precision-only',action='store_true')
    args=parser.parse_args()
    if not args.precision_only:
        if args.training_dir is None:
            parser.error('--training-dir is required')
        summary=analyze_dense(args.output,args.training_dir)
        print(f"Complete: {summary['n_checkpoints']} checkpoints.")
    if args.fp32_output is not None:
        rows=compare_precision(args.output,args.fp32_output)
        print(f'Compared precision at {len(rows)} checkpoints.')
    elif args.precision_only:
        parser.error('--precision-only requires --fp32-output')
    write_dense_report(args.output)


if __name__=='__main__':
    main()
