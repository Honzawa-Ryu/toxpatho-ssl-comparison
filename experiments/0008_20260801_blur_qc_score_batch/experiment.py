"""blur_qc Step 1 (score_blur) を全WSI(~998枚)に対してarrayジョブで実行する。

WSI一覧を config.yml の analysis.num_chunks 個に分割し(ストライプ分割)、
chunk_index番目の担当分だけをこのタスクが処理する。実処理は
lib/analysis/blur_qc.compute_blur_scores() を直接呼ぶ(REFACTOR_PLAN.md §5-0)。
CLIシム単体で動作確認したい場合は
experiments/0007_20260801_blur_qc_pipeline/score_blur.py を使うこと
(本ファイルはそのarray実行版)。

出力:
    data/blur_qc/scores/{wsi_id}_blur.h5          … WSIごと
    data/blur_qc/scores_summary_chunk{N}.csv      … chunkごとのサマリ

全chunk完了後、`scores_summary_chunk*.csv` を1つの `scores_summary.csv` に
まとめること(ヘッダー行の重複に注意)。

投入前の注意(Goal.md §7):
    - experiments/0007_20260801_blur_qc_pipeline/ の検証結果で動作確認済みで
      あることを確認してから投入する。
    - num_chunks を変更したら run_slurm.sh の GRID_VALUES と
      #SBATCH --array=0-N を必ず合わせて変更する。
"""
import csv
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


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.analysis.blur_qc import compute_blur_scores
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = build_variant_key(args)
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)

    config = load_config(Path(__file__).parent)
    params = config.get("analysis", {})
    num_chunks = int(params["num_chunks"])

    wsi_dir = project_root / params["wsi_dir"]
    coords_dir = project_root / params["coords_dir"]
    output_dir = project_root / params["output_dir"]
    patch_size = params["patch_size"]
    output_dir.mkdir(parents=True, exist_ok=True)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    wsi_ids = sorted(
        p.stem for p in wsi_dir.iterdir()
        if p.suffix.lower() in WSI_EXTENSIONS and (coords_dir / f"{p.stem}_patches.h5").exists()
    )
    my_wsi_ids = wsi_ids[args.chunk_index::num_chunks]
    print(f"chunk {args.chunk_index}/{num_chunks}: {len(my_wsi_ids)} WSIs")

    summary_path = output_dir.parent / f"scores_summary_chunk{args.chunk_index}.csv"
    with open(summary_path, "w", newline="") as summary_f:
        writer = csv.writer(summary_f)
        writer.writerow(["wsi_id", "n_patches", "blur_min", "blur_median", "blur_max"])

        for wsi_id in my_wsi_ids:
            candidates = list(wsi_dir.glob(f"{wsi_id}.*"))
            if not candidates:
                print(f"[skip] {wsi_id}: WSI file not found")
                continue
            wsi_path = candidates[0]

            coords_h5_path = coords_dir / f"{wsi_id}_patches.h5"
            out_h5 = output_dir / f"{wsi_id}_blur.h5"
            if out_h5.exists():
                print(f"[skip] {wsi_id}: already scored")
                continue

            print(f"Processing: {wsi_id}")
            coords, scores = compute_blur_scores(str(wsi_path), str(coords_h5_path), patch_size)

            with h5py.File(str(out_h5), "w") as f:
                f.create_dataset("coords", data=coords, compression="gzip")
                f.create_dataset("blur_score", data=scores, compression="gzip")

            writer.writerow([
                wsi_id,
                len(scores),
                float(scores.min()) if len(scores) else "",
                float(np.median(scores)) if len(scores) else "",
                float(scores.max()) if len(scores) else "",
            ])
            summary_f.flush()

    complete_run(run_dir)


if __name__ == "__main__":
    main()
