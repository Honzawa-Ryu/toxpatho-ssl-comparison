# -*- coding: utf-8 -*-
"""
WSIパッチ単位のぼやけ(blur)スコアリング CLIシム(Goal.md Step 1)。

実処理は `lib/analysis/blur_qc.compute_blur_scores()` にある。このファイルは
WSI一覧をイテレートして呼ぶだけ(scripts/evaluate/extract_features_pathobench.py
と同じ構成)。

NOTE: 本来の配置場所は `scripts/analysis/score_blur.py`(Goal.md §4.3)。
campaign モードのhookが新規Pythonファイルを `lib/` または
`experiments/{id}_.../` 配下にしか作成させないため、暫定的にこの実験ディレクトリに
置いている。レビュー後、`git mv` で `scripts/analysis/score_blur.py` へ移動すること
(import パスはそのままで動く)。

Usage:
    python experiments/0007_20260801_blur_qc_pipeline/score_blur.py \
        --wsi_dir data/moo_collected_tggate_wsi \
        --coords_dir data/trident_processed/20x_224px_0px_overlap/patches \
        --output_dir data/blur_qc/scores
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

from lib.analysis.blur_qc import compute_blur_scores  # noqa: E402

WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".vms", ".vmu", ".scn", ".mrxs", ".bif"}


def parse_args():
    ap = argparse.ArgumentParser(description="Score patch-level blur (Laplacian variance) for WSIs")
    ap.add_argument("--wsi_dir", required=True, help="WSI本体(.svs等)を含むディレクトリ")
    ap.add_argument("--coords_dir", required=True, help="TRIDENT座標h5({wsi_id}_patches.h5)を含むディレクトリ")
    ap.add_argument("--output_dir", default="data/blur_qc/scores", help="{wsi_id}_blur.h5 の出力先")
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--wsi_id", default="", help="指定した1WSIのみ処理する(動作確認用)")
    ap.add_argument("--resume", action="store_true", help="既に出力済みのWSIをスキップする")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir.parent / "scores_summary.csv"

    wsi_paths = [
        p for p in sorted(Path(args.wsi_dir).iterdir())
        if p.suffix.lower() in WSI_EXTENSIONS
    ]
    if args.wsi_id:
        wsi_paths = [p for p in wsi_paths if p.stem == args.wsi_id]
        if not wsi_paths:
            raise FileNotFoundError(f"--wsi_id={args.wsi_id} に一致するWSIが {args.wsi_dir} に見つかりません")

    print(f"Found {len(wsi_paths)} WSIs to process")

    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as summary_f:
        writer = csv.writer(summary_f)
        if write_header:
            writer.writerow(["wsi_id", "n_patches", "blur_min", "blur_median", "blur_max"])

        for wsi_path in wsi_paths:
            wsi_id = wsi_path.stem
            out_h5 = out_dir / f"{wsi_id}_blur.h5"
            if args.resume and out_h5.exists():
                print(f"[skip] {wsi_id}")
                continue

            coords_h5_path = Path(args.coords_dir) / f"{wsi_id}_patches.h5"
            if not coords_h5_path.exists():
                print(f"[skip] {wsi_id}: coords h5 not found at {coords_h5_path}")
                continue

            print(f"Processing: {wsi_id}")
            coords, scores = compute_blur_scores(str(wsi_path), str(coords_h5_path), args.patch_size)

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
            print(f"  -> {len(scores)} patches, saved to {out_h5}")

    print("Done.")


if __name__ == "__main__":
    main()
