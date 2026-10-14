import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

# --- Deep learning (uncomment if needed) ---
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# --- Visualization (uncomment if needed) ---
# import matplotlib.pyplot as plt
# import seaborn as sns


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(config_path: Path) -> dict:
    """Load a config YAML file.

    The path comes from --config, which run_slurm.sh passes as a path relative
    to the experiment directory (so `--config config.yml` means the config.yml
    sitting next to this script). Resolving it here rather than hardcoding
    exp_dir/config.yml is what makes it possible to feed the same
    experiment.py a different config.
    """
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    # Config file, resolved relative to this experiment directory.
    # run_slurm.sh's RUN_COMMAND passes `--config config.yml` by default.
    parser.add_argument(
        "--config",
        default="config.yml",
        help="config YAML (experiment ディレクトリからの相対パス、または絶対パス)",
    )
    # Add one argument per swept dimension (required=True).
    # These must match GRID_ARGS entries in run_slurm.sh.
    # Example:
    #   parser.add_argument("--model", required=True)
    #   parser.add_argument("--seed",  required=True)
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # Build variant_key from all variable dimensions so each config gets its own directory.
    # Include every arg that changes the result (model, seed, prompt, ...).
    # Example:
    #   model_short = args.model.replace("/", "-")
    #   variant_key = f"{model_short}__{args.seed}"
    variant_key = "default"

    # Initialize output directory (exits immediately if already completed).
    # Written on OUTPUT_ROOT (scratch) when set; scripts/slurm_entry.sh
    # rsyncs it back to project_root/outputs/ at job end.
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)
    seed: int = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    from lib.analysis import compare, embeddings

    methods_config = project_root / config["methods_config"]
    # 学習側(lib/trainer/data.py)と同じく DATASET_DIR 配下を見る。
    # USE_LOCAL_SSD_INPUT=1 のときノードローカルSSDのコピーを読むため。
    data_dir = dataset_dir / "ssl_patches"
    logger.info(f"methods:     {methods_config}")
    logger.info(f"data_dir:    {data_dir}")

    # (1) 全チェックポイント共通の固定パッチ集合を埋め込む
    done = embeddings.run(
        str(methods_config),
        data_dir=str(data_dir),
        per_slide=config["per_slide"],
        fold_idx=config["fold_idx"],
        num_folds=config["num_folds"],
        batch_size=config["batch_size"],
        output_dir=str(run_dir),
    )
    # checkpoint が見つからない手法は embeddings.run が skip するだけなので、
    # 全部 skip されても黙って進んでしまう。旧 methods_paper.yaml が別クラスタの
    # 絶対パスを指していた事故がこれなので、1つも埋め込めなければここで落とす。
    if not done:
        raise RuntimeError(
            f"埋め込みが1つも作られなかった。{methods_config} の model_path を確認すること。"
        )
    logger.info(f"embedded:    {list(done)}")

    # (2) CKA / k-means の ARI・NMI / クラスタ別プロトタイプパッチ / t-SNE
    report_path = compare.run(
        str(run_dir),
        k=config["kmeans_k"],
        rbf=config["rbf_cka"],
    )

    results: dict = {"embeddings": done, "report": str(report_path)}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()