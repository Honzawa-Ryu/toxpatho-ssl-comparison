import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

# --- Deep learning ---
import torch


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
    # 汎用パイプライン疎通確認（SSL学習ロジックは含まない）:
    #   create_exp -> run_slurm.sh -> runx投入 -> slurm_entry.sh(apptainer実行)
    #   -> このスクリプト -> 出力回収、が一通り正常に動くかだけを確認する。
    results: dict = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "project_root": str(project_root),
        "dataset_dir_exists": dataset_dir.exists(),
        "env": {
            k: os.environ.get(k)
            for k in ["PROJECT_ROOT", "EXP_NAME", "DATASET_DIR", "OUTPUT_ROOT", "APPTAINER_CONTAINER"]
        },
    }
    logger.info(f"python:      {results['python_version']}")
    logger.info(f"torch:       {results['torch_version']}")
    logger.info(f"hostname:    {results['hostname']}")

    # nvidia-smi (コンテナ内から --nv 経由でGPUが見えているかの確認)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        results["nvidia_smi_output"] = smi.stdout.strip()
        logger.info(f"nvidia-smi:  {smi.stdout.strip() or smi.stderr.strip()}")
    except FileNotFoundError:
        results["nvidia_smi_output"] = None
        logger.warning("nvidia-smi not found")

    # torch側のCUDA認識確認
    cuda_available = torch.cuda.is_available()
    results["cuda_available"] = cuda_available
    results["cuda_device_count"] = torch.cuda.device_count() if cuda_available else 0
    logger.info(f"cuda_available: {cuda_available}, device_count: {results['cuda_device_count']}")

    if cuda_available:
        device = torch.device("cuda:0")
        results["cuda_device_name"] = torch.cuda.get_device_name(device)
        # 実際にGPU上でtensor演算を回し、apptainer --nv 経由でのCUDA実行そのものを確認する
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        torch.cuda.synchronize(device)
        c = a @ b
        torch.cuda.synchronize(device)
        results["matmul_check_sum"] = float(c.sum().item())
        logger.info(f"GPU matmul check OK (device={results['cuda_device_name']})")
    else:
        logger.warning("CUDA not available; skipping GPU matmul check (--gres=gpu:1 requested)")

    results["status"] = "ok"

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()