"""blur_qc Step 2 (sample_patches_memmap) を全WSI(~998枚)に対して
arrayジョブで実行する。

score_blur(experiments/0008_.../)で計算済みの `data/blur_qc/scores/*_blur.h5`
一覧を config.yml の analysis.num_chunks 個に分割し(ストライプ分割)、
chunk_index番目の担当分だけをこのタスクが処理する。実処理は
lib/analysis/blur_qc.sample_patches_to_memmap() を直接呼ぶ
(REFACTOR_PLAN.md §5-0)。CLIシム単体で動作確認したい場合は
experiments/0007_20260801_blur_qc_pipeline/sample_patches_memmap.py を使うこと
(本ファイルはそのarray実行版)。

threshold_scope="global" のときは、全chunkが独立に
`data/blur_qc/scores/*_blur.h5` の blur_score をプールしてパーセンタイル
閾値を算出する(Goal.md §5.1の「scores_summary.csvから算出」は生の分布を
持たないため不正確。個別h5を直接プールする方が正確 — 詳細は
experiments/0007_20260801_blur_qc_pipeline/sample_patches_memmap.py の
NOTE参照)。スコア配列自体は小さいため、chunkごとに再計算しても軽量。

投入前の注意(Goal.md §7-8):
    - config.yml の percentile が Goal.md §5.1 に従いヒストグラム確認後の
      値に設定されていること(null のままだと起動時に例外で止まる)。
    - data/blur_qc/patches/ の想定容量(約140GiB)を利用者に確認済みである
      こと。
    - num_chunks を変更したら run_slurm.sh の GRID_VALUES と
      #SBATCH --array=0-N を必ず合わせて変更する。
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".vms", ".vmu", ".scn", ".mrxs", ".bif"}


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_index", type=int, required=True)
    return parser.parse_args()


def build_variant_key(args) -> str:
    return f"chunk{args.chunk_index}"


def _find_wsi_path(wsi_dir: Path, wsi_id: str) -> Path:
    for ext in WSI_EXTENSIONS:
        candidate = wsi_dir / f"{wsi_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{wsi_id} のWSIファイルが {wsi_dir} に見つかりません")


def _compute_global_threshold(scores_dir: Path, percentile: float) -> float:
    all_scores = []
    for h5_path in sorted(scores_dir.glob("*_blur.h5")):
        with h5py.File(h5_path, "r") as f:
            all_scores.append(f["blur_score"][:])
    if not all_scores:
        raise FileNotFoundError(f"{scores_dir} に *_blur.h5 が見つかりません")
    return float(np.percentile(np.concatenate(all_scores), percentile))


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.analysis.blur_qc import sample_patches_to_memmap
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = build_variant_key(args)
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)

    config = load_config(Path(__file__).parent)
    params = config.get("analysis", {})
    num_chunks = int(params["num_chunks"])
    threshold_scope = params["threshold_scope"]
    percentile = params["percentile"]

    if threshold_scope in ("per_slide", "global") and percentile is None:
        raise ValueError(
            "config.yml analysis.percentile が未設定です。Goal.md §5.1の通り、"
            "スコア分布のヒストグラムを確認してから設定してください。"
        )

    scores_dir = project_root / params["scores_dir"]
    wsi_dir = project_root / params["wsi_dir"]
    output_dir = project_root / params["output_dir"]
    n_patches = int(params["n_patches"])
    seed = int(params["seed"])
    patch_size = int(params["patch_size"])
    output_dir.mkdir(parents=True, exist_ok=True)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    score_h5_paths = sorted(scores_dir.glob("*_blur.h5"))
    if not score_h5_paths:
        raise FileNotFoundError(
            f"{scores_dir} に *_blur.h5 が見つかりません。先に score_blur "
            "(experiments/0008_.../) を完了させてください。"
        )
    my_h5_paths = score_h5_paths[args.chunk_index::num_chunks]
    print(f"chunk {args.chunk_index}/{num_chunks}: {len(my_h5_paths)} WSIs")

    global_threshold = None
    if threshold_scope == "global":
        global_threshold = _compute_global_threshold(scores_dir, percentile)
        print(f"global threshold (percentile={percentile}): {global_threshold}")

    for h5_path in my_h5_paths:
        wsi_id = h5_path.stem[: -len("_blur")]
        wsi_path = _find_wsi_path(wsi_dir, wsi_id)

        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            scores = f["blur_score"][:]

        print(f"Processing: {wsi_id} ({len(scores)} candidate patches)")
        meta = sample_patches_to_memmap(
            wsi_path=str(wsi_path),
            coords=coords,
            scores=scores,
            threshold_scope=threshold_scope,
            percentile=percentile,
            n_patches=n_patches,
            patch_size=patch_size,
            out_dir=str(output_dir),
            seed=seed,
            global_threshold=global_threshold,
        )
        print(f"  -> sampled {meta['shape'][0]} patches")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
