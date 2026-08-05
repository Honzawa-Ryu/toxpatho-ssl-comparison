import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import h5py

# --- Visualization ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    parser.add_argument(
        "--config",
        default="config.yml",
        help="config YAML (experiment ディレクトリからの相対パス、または絶対パス)",
    )
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    variant_key = "default"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)
    seed: int = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir: {run_dir}")
    logger.info(f"seed:    {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    # coords h5 の blur_score をパッチグリッド位置に並べ直し、スライドごとの空間
    # ヒートマップを描く(0006の集計でmedian blur_scoreが極端だった20枚が対象)。
    # coordsはlevel0ピクセル座標・0px overlapのグリッドなので、patch_sizeで割って
    # そのまま行列インデックスにできる(再切り出し・WSI本体アクセスは不要)。
    hm_params = config["blur_heatmap"]
    coords_dir = project_root / hm_params["coords_dir"]
    patch_size = int(hm_params["patch_size"])
    wsi_ids = hm_params["wsi_ids"]

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    results: list = []
    for wsi_id in wsi_ids:
        h5_path = coords_dir / f"{wsi_id}_patches.h5"
        if not h5_path.exists():
            logger.info(f"[skip] {wsi_id}: coords h5 not found at {h5_path}")
            continue

        with h5py.File(h5_path, "r") as f:
            if "blur_score" not in f:
                logger.info(f"[skip] {wsi_id}: blur_score dataset not found")
                continue
            coords = f["coords"][:]
            scores = f["blur_score"][:].astype(np.float64)

        cols = coords[:, 0] // patch_size
        rows = coords[:, 1] // patch_size

        grid = np.full((rows.max() + 1, cols.max() + 1), np.nan, dtype=np.float64)
        grid[rows, cols] = scores

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(grid, cmap="viridis")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("blur_score (Laplacian variance)")
        ax.set_title(f"{wsi_id}  (median={np.median(scores):.0f}, n={len(scores)})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()

        out_path = plots_dir / f"{wsi_id}_blur_heatmap.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        logger.info(f"Wrote heatmap: {out_path}")
        results.append({
            "wsi_id": wsi_id,
            "n_patches": int(len(scores)),
            "median": float(np.median(scores)),
            "plot": str(out_path.relative_to(run_dir)),
        })

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
