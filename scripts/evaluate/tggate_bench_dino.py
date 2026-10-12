"""DINO 0028 on tggate-bench: extract embeddings, run its evaluations, summarize.

  extract              GPU; all epochs in one pass over the tiles (resumable)
  eval --task-index i  one tggate-bench run (Exp 0011 anomaly or Exp 0002 ridge)
  list-tasks           print the eval tasks (index, experiment, args)
  summarize            study-mean AUROC with bootstrap CIs, figure, paired differences

Run with tggate-bench's Python (.venv), which has torch/timm/sklearn.
"""

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lib.analysis import tggate_bench as tb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("extract", "eval", "list-tasks", "summarize"))
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/evaluate/dino_0028_tggate.yaml")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    c = yaml.safe_load(args.config.read_text())
    tasks = tb.eval_tasks(c["dataset"], c["epochs"])

    if args.command == "extract":
        tb.extract(c["bench_root"], c["overlay"], c["dataset"], c["checkpoint_dir"], c["epochs"],
                   batch_size=args.batch_size, workers=args.workers, device_name=args.device,
                   precision=args.precision)
    elif args.command == "list-tasks":
        for i, (exp, task_args) in enumerate(tasks):
            print(i, exp, " ".join(task_args))
    elif args.command == "eval":
        if args.task_index is None or not 0 <= args.task_index < len(tasks):
            parser.error(f"--task-index must be in [0, {len(tasks) - 1}]")
        tb.run_bench_experiment(c["bench_root"], c["overlay"], *tasks[args.task_index])
    else:
        table, diffs = tb.summarize(c["bench_root"], c["overlay"], c["dataset"], c["epochs"], c["summary_dir"])
        print(table.to_string(index=False))
        print(diffs.to_string(index=False))


if __name__ == "__main__":
    main()
