import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import h5py
import numpy as np
import pandas as pd


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
    # 全WSI(coords_dir配下の{wsi_id}_patches.h5に既にblur_score追記済み)から、
    # 全体統一のぼやけ閾値(threshold)を通過したパッチをWSIごとにn_patches_per_wsi枚
    # ランダムサンプリングし、全WSI共有の単一memmapへ書き込む(Goal.md参照)。
    # 巨大ジョブなので、index.csvに既に記録済みのwsi_idはスキップしてresumeする。
    from lib.analysis.blur_qc import build_offset_table, init_shared_memmap, sample_patches_to_shared_memmap

    sample_params = config["sample"]
    coords_dir = project_root / sample_params["coords_dir"]
    wsi_dir = project_root / sample_params["wsi_dir"]
    patch_size = int(sample_params["patch_size"])
    n_patches_per_wsi = int(sample_params["n_patches_per_wsi"])
    threshold = float(sample_params["threshold"])
    output_dir = project_root / sample_params["output_dir"]

    h5_paths = sorted(coords_dir.glob("*_patches.h5"))
    wsi_ids = [p.stem.removesuffix("_patches") for p in h5_paths]
    total_n_patches = len(wsi_ids) * n_patches_per_wsi
    logger.info(f"Found {len(wsi_ids)} WSIs under {coords_dir}")
    logger.info(f"threshold={threshold}  n_patches_per_wsi={n_patches_per_wsi}  total_n_patches={total_n_patches}")

    output_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = output_dir / "patches.memmap"
    index_path = output_dir / "index.csv"

    init_shared_memmap(str(memmap_path), total_n_patches, patch_size)
    offset_table = build_offset_table(wsi_ids, n_patches_per_wsi)

    processed_wsi_ids: set[str] = set()
    if index_path.exists():
        existing_df = pd.read_csv(index_path, dtype={"wsi_id": str})
        processed_wsi_ids = set(existing_df["wsi_id"].unique())
        logger.info(f"Resuming: {len(processed_wsi_ids)} WSIs already in {index_path}")

    write_header = not index_path.exists()
    n_newly_processed = 0
    with open(index_path, "a", newline="") as index_f:
        writer = csv.writer(index_f)
        if write_header:
            writer.writerow(["row", "wsi_id", "x", "y", "blur_score"])

        for h5_path, wsi_id in zip(h5_paths, wsi_ids):
            if wsi_id in processed_wsi_ids:
                continue

            wsi_path = wsi_dir / f"{wsi_id}.svs"
            if not wsi_path.exists():
                raise FileNotFoundError(f"{wsi_id}: WSI本体が見つかりません: {wsi_path}")

            with h5py.File(h5_path, "r") as f:
                if "blur_score" not in f:
                    raise KeyError(f"{h5_path} に blur_score データセットがありません")
                coords = f["coords"][:]
                scores = f["blur_score"][:].astype(np.float64)

            rows = sample_patches_to_shared_memmap(
                wsi_path=str(wsi_path),
                coords=coords,
                scores=scores,
                threshold=threshold,
                n_patches=n_patches_per_wsi,
                patch_size=patch_size,
                memmap_path=str(memmap_path),
                total_n_patches=total_n_patches,
                offset=offset_table[wsi_id],
                seed=seed,
            )
            for row in rows:
                writer.writerow([row["row"], row["wsi_id"], row["x"], row["y"], row["blur_score"]])
            index_f.flush()
            n_newly_processed += 1
            offset = offset_table[wsi_id]
            logger.info(
                f"[{n_newly_processed}] {wsi_id}: sampled {len(rows)} patches "
                f"-> rows [{offset}, {offset + n_patches_per_wsi})"
            )

    results = {
        "n_wsi_total": len(wsi_ids),
        "n_wsi_newly_processed": n_newly_processed,
        "n_wsi_already_done_before_this_run": len(processed_wsi_ids),
        "total_n_patches": total_n_patches,
        "patch_size": patch_size,
        "n_patches_per_wsi": n_patches_per_wsi,
        "threshold": threshold,
        "seed": seed,
        "memmap_path": str(memmap_path),
        "index_path": str(index_path),
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()