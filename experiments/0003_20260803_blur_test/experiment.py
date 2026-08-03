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
    # experiments/0002_20260803_data_preprocess と同じロジック。1枚のWSIのみ対象に
    # した動作確認用(config.yml の blur.wsi_ids で絞り込み)。
    from lib.data_preprocess.calculate_blur import append_blur_scores_from_wsi

    WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".vms", ".vmu", ".scn", ".mrxs", ".bif"}

    blur_params = config.get("blur", {})
    # wsi_dir: 読み取り専用の生スライド入力。USE_LOCAL_SSD_INPUT=1 + DATA_SUBDIRS で
    # data/raw_slide を scratch にコピーした場合、dataset_dir 経由でそちらを読む
    # （dataset_dir.parent は USE_LOCAL_SSD_INPUT=0 なら project_root と同じ値になる
    # ので、無効時の挙動は変わらない）。
    wsi_dir = dataset_dir.parent / blur_params["wsi_dir"]
    # coords_dir: append_blur_scores_from_wsi がこの下のh5に直接スコアを追記する
    # "その場書き換え" 対象なので、常に project_root（NFS）を指す。DATASET_DIR
    # 経由にすると、scratch 側に書いた追記結果が NFS へ同期されず消える
    # （scripts/slurm_entry.sh が起動時に同期するのは OUTPUT_DIR のみ）。
    coords_dir = project_root / blur_params["coords_dir"]
    patch_size = int(blur_params["patch_size"])
    wsi_ids = blur_params.get("wsi_ids")

    wsi_paths = sorted(p for p in wsi_dir.iterdir() if p.suffix.lower() in WSI_EXTENSIONS)
    if wsi_ids:
        wanted = set(wsi_ids)
        wsi_paths = [p for p in wsi_paths if p.stem in wanted]

    logger.info(f"Found {len(wsi_paths)} WSIs to process")

    results: list = []
    for wsi_path in wsi_paths:
        wsi_id = wsi_path.stem
        coords_h5_path = coords_dir / f"{wsi_id}_patches.h5"
        if not coords_h5_path.exists():
            logger.info(f"[skip] {wsi_id}: coords h5 not found at {coords_h5_path}")
            continue

        logger.info(f"Processing: {wsi_id}")
        scores = append_blur_scores_from_wsi(str(wsi_path), coords_h5_path, patch_size=patch_size)
        results.append({
            "wsi_id": wsi_id,
            "n_patches": len(scores),
            "blur_min": float(scores.min()) if len(scores) else None,
            "blur_median": float(np.median(scores)) if len(scores) else None,
            "blur_max": float(scores.max()) if len(scores) else None,
        })
        logger.info(f"  -> {len(scores)} patches scored, appended to {coords_h5_path}")

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()