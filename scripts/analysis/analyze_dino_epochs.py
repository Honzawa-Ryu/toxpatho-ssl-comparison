"""Extract exp 0028 features and run exploratory CKA/geometry on local images."""

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.analysis.dino_epoch_analysis import analyze, extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/evaluate/dino_0028.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--scan-patches", type=int, default=5000)
    parser.add_argument("--per-slide", type=int, default=20)
    parser.add_argument("--slides", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    for name in ("scan_patches", "per_slide", "slides", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    config = yaml.safe_load(args.config.read_text())
    epochs = config["epochs"]
    if not epochs or any(type(epoch) is not int or epoch <= 0 for epoch in epochs) or len(set(epochs)) != len(epochs):
        parser.error("epochs must be unique positive integers")
    if not args.analyze_only:
        if args.shard is None or args.checkpoint_root is None:
            parser.error("--shard and --checkpoint-root are required for extraction")
        extract(args.shard, args.checkpoint_root, epochs, args.output,
                args.scan_patches, args.per_slide, args.slides,
                args.batch_size, args.device)
    summary = analyze(args.output, args.report_dir)
    print(f"Complete: {summary['n_patches']} patches; report={args.report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
