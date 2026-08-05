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
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
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
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    # 0002/0003/0005 で既に coords h5 の blur_score データセットに書き込み済みの
    # ラプラシアン分散を全WSI分読み、閾値方針(global vs per_slide)を決めるための
    # 分布・統計量を出す(WSI/パッチの再読み込みはしない、h5のスコアを読むだけの軽量ジョブ)。
    dist_params = config.get("blur_dist", {})
    coords_dir = project_root / dist_params["coords_dir"]
    candidate_percentiles = dist_params.get("candidate_percentiles", [1, 5, 10, 20])
    n_target_patches = int(dist_params.get("n_target_patches", 1000))

    h5_paths = sorted(coords_dir.glob("*_patches.h5"))
    logger.info(f"Found {len(h5_paths)} coords h5 files under {coords_dir}")

    PCTS = sorted(set([1, 5, 10, 25, 50, 75, 90, 99] + list(candidate_percentiles)))

    per_slide_rows = []
    all_scores_list = []
    missing_blur = []

    for h5_path in h5_paths:
        wsi_id = h5_path.stem.removesuffix("_patches")
        with h5py.File(h5_path, "r") as f:
            if "blur_score" not in f:
                missing_blur.append(wsi_id)
                continue
            scores = f["blur_score"][:].astype(np.float64)
        if len(scores) == 0:
            missing_blur.append(wsi_id)
            continue

        row = {
            "wsi_id": wsi_id,
            "n_patches": len(scores),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }
        for p in PCTS:
            row[f"p{p}"] = float(np.percentile(scores, p))
        per_slide_rows.append(row)
        all_scores_list.append(scores)

    if missing_blur:
        preview = missing_blur[:20]
        suffix = "..." if len(missing_blur) > 20 else ""
        logger.warning(f"blur_score not found/empty for {len(missing_blur)} WSIs: {preview}{suffix}")

    per_slide_df = pd.DataFrame(per_slide_rows).sort_values("wsi_id").reset_index(drop=True)
    per_slide_csv = run_dir / "per_slide_stats.csv"
    per_slide_df.to_csv(per_slide_csv, index=False)
    logger.info(f"Wrote per-slide stats: {per_slide_csv} ({len(per_slide_df)} slides)")

    all_scores = np.concatenate(all_scores_list)
    n_total = len(all_scores)
    logger.info(f"Pooled patch count: {n_total}")

    pooled_percentiles = {f"p{p}": float(np.percentile(all_scores, p)) for p in PCTS}
    pooled_percentiles = dict(sorted(pooled_percentiles.items(), key=lambda kv: int(kv[0][1:])))

    # スライドあたりのパッチ数が少なく、閾値フィルタ前から n_target_patches に届かない
    # WSIをリストアップ(サンプリング段階で「あるだけ全部採用」になる対象の目安)。
    n_patches_arr = per_slide_df["n_patches"].to_numpy()
    n_slides_below_target = int((n_patches_arr < n_target_patches).sum())
    slides_below_target = per_slide_df.loc[
        per_slide_df["n_patches"] < n_target_patches, "wsi_id"
    ].tolist()

    # per-slide の median のばらつき(スライド間で「標準的なシャープさ」がどれだけ違うか)。
    slide_medians = per_slide_df["p50"].to_numpy()
    slide_median_cv = float(slide_medians.std() / slide_medians.mean())

    pooled_summary = {
        "n_wsi": len(per_slide_df),
        "n_wsi_missing_blur_score": len(missing_blur),
        "n_patches_total": int(n_total),
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        "min": float(all_scores.min()),
        "max": float(all_scores.max()),
        "percentiles": pooled_percentiles,
        "n_target_patches": n_target_patches,
        "n_slides_below_target": n_slides_below_target,
        "slides_below_target": slides_below_target,
        "slide_median_mean": float(slide_medians.mean()),
        "slide_median_std": float(slide_medians.std()),
        "slide_median_cv": slide_median_cv,
    }

    # 候補percentileごとに、global閾値を各スライドに当てはめた場合の除外率のばらつきを見る。
    # (per_slideスコープでは定義上どのスライドでも除外率=percentileそのものになるため、
    #  「global閾値がスライドごとにどれだけ違う扱いになるか」だけを見ればよい)
    threshold_rows = []
    for p in candidate_percentiles:
        global_threshold = float(np.percentile(all_scores, p))
        exclusion_rates = np.array([
            float((scores < global_threshold).mean()) for scores in all_scores_list
        ])
        threshold_rows.append({
            "percentile": p,
            "global_threshold": global_threshold,
            "excl_rate_mean": float(exclusion_rates.mean()),
            "excl_rate_std": float(exclusion_rates.std()),
            "excl_rate_min": float(exclusion_rates.min()),
            "excl_rate_max": float(exclusion_rates.max()),
            "n_slides_excl_over_50pct": int((exclusion_rates > 0.5).sum()),
            "n_slides_excl_zero": int((exclusion_rates == 0.0).sum()),
        })
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_csv = run_dir / "global_threshold_exclusion_rates.csv"
    threshold_df.to_csv(threshold_csv, index=False)
    logger.info(f"Wrote global-threshold exclusion-rate comparison: {threshold_csv}")

    summary_path = run_dir / "pooled_summary.json"
    summary_path.write_text(json.dumps(pooled_summary, indent=2, ensure_ascii=False))
    logger.info(f"Wrote pooled summary: {summary_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # (a) pooled histogram (log-x、ラプラシアン分散は右に裾が長いため)
    fig, ax = plt.subplots(figsize=(8, 5))
    positive = all_scores[all_scores > 0]
    ax.hist(np.log10(positive), bins=100, color="#4C72B0")
    for p in candidate_percentiles:
        v = np.percentile(all_scores, p)
        if v > 0:
            ax.axvline(np.log10(v), color="red", linestyle="--", linewidth=1)
            ax.text(np.log10(v), ax.get_ylim()[1] * 0.95, f"p{p}", rotation=90, va="top", fontsize=8, color="red")
    ax.set_xlabel("log10(blur_score)")
    ax.set_ylabel("count")
    ax.set_title(f"Pooled blur score distribution (n={n_total} patches, {len(per_slide_df)} WSIs)")
    fig.tight_layout()
    fig.savefig(plots_dir / "pooled_hist.png", dpi=150)
    plt.close(fig)

    # (b) per-slide median の分布(スライド間のばらつき)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(slide_medians, bins=50, color="#55A868")
    ax.set_xlabel("per-slide median blur_score")
    ax.set_ylabel("n WSIs")
    ax.set_title(f"Per-slide median blur score (CV={slide_median_cv:.2f})")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_slide_median_hist.png", dpi=150)
    plt.close(fig)

    # (c) global閾値を全スライドに当てはめた場合の、スライドごとの除外率のばらつき
    n_cand = len(candidate_percentiles)
    fig, axes = plt.subplots(1, n_cand, figsize=(4 * n_cand, 4), sharey=True)
    if n_cand == 1:
        axes = [axes]
    for ax, p in zip(axes, candidate_percentiles):
        global_threshold = float(np.percentile(all_scores, p))
        exclusion_rates = np.array([
            float((scores < global_threshold).mean()) for scores in all_scores_list
        ]) * 100
        ax.hist(exclusion_rates, bins=30, color="#C44E52")
        ax.axvline(p, color="black", linestyle="--", linewidth=1, label=f"per_slide baseline ({p}%)")
        ax.set_title(f"global p{p} threshold")
        ax.set_xlabel("per-slide exclusion rate (%)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("n WSIs")
    fig.suptitle("Per-slide exclusion rate when applying a single global threshold")
    fig.tight_layout()
    fig.savefig(plots_dir / "global_threshold_exclusion_rate_by_slide.png", dpi=150)
    plt.close(fig)

    logger.info(f"Slide median CV: {slide_median_cv:.4f}")
    logger.info(f"Slides with n_patches < {n_target_patches}: {n_slides_below_target}")
    for row in threshold_rows:
        logger.info(
            f"  p{row['percentile']}: global_threshold={row['global_threshold']:.2f}, "
            f"excl_rate mean={row['excl_rate_mean']*100:.2f}% std={row['excl_rate_std']*100:.2f}% "
            f"range=[{row['excl_rate_min']*100:.2f}%, {row['excl_rate_max']*100:.2f}%], "
            f"n_slides>50%excluded={row['n_slides_excl_over_50pct']}"
        )

    results = {
        "n_wsi": len(per_slide_df),
        "n_wsi_missing_blur_score": len(missing_blur),
        "missing_blur_wsi_ids": missing_blur,
        "n_patches_total": int(n_total),
        "n_target_patches": n_target_patches,
        "n_slides_below_target": n_slides_below_target,
        "slides_below_target": slides_below_target,
        "slide_median_cv": slide_median_cv,
        "candidate_percentiles": threshold_rows,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
