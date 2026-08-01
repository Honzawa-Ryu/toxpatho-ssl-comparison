# -*- coding: utf-8 -*-
"""
ぼやけスコアによる足切り + ランダムサンプリング + memmap保存 CLIシム(Goal.md Step 2)。

実処理は `lib/analysis/blur_qc.sample_patches_to_memmap()` にある。
`--threshold_scope global` のときのみ、`--scores_dir` 配下の `*_blur.h5` の
`blur_score` を全WSI分プールしてパーセンタイル閾値を1回だけ算出し、以降の
WSIループへ渡す。

NOTE: Goal.md §5.1 は「global モードは scores_summary.csv から算出する」と
書いているが、そのCSVは wsi_id ごとの min/median/max のみで生の分布を
持たないため、正確なパーセンタイルを再現できない。同じ scores_dir に
既にある個別h5(`{wsi_id}_blur.h5` の `blur_score`)を直接プールする方が
プール全体の分布を正確に反映できるため、本実装ではそちらを採用している。
scores_summary.csv は当初の用途(ヒストグラム確認用の簡易サマリ)のまま出力する。

NOTE: 本来の配置場所は `scripts/analysis/sample_patches_memmap.py`(Goal.md §5.4)。
campaign モードのhookが新規Pythonファイルを `lib/` または
`experiments/{id}_.../` 配下にしか作成させないため、暫定的にこの実験ディレクトリに
置いている。レビュー後、`git mv` で `scripts/analysis/sample_patches_memmap.py` へ
移動すること(import パスはそのままで動く)。

Usage:
    python experiments/0007_20260801_blur_qc_pipeline/sample_patches_memmap.py \
        --scores_dir data/blur_qc/scores \
        --wsi_dir data/moo_collected_tggate_wsi \
        --threshold_scope per_slide --percentile 25 \
        --n_patches 1000 --seed 42 \
        --output_dir data/blur_qc/patches
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
sys.path.insert(0, PROJECT_ROOT)

from lib.analysis.blur_qc import sample_patches_to_memmap  # noqa: E402

WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".vms", ".vmu", ".scn", ".mrxs", ".bif"}


def parse_args():
    ap = argparse.ArgumentParser(description="Threshold + random sample + memmap patches")
    ap.add_argument("--scores_dir", default="data/blur_qc/scores", help="score_blur.py の出力ディレクトリ")
    ap.add_argument("--wsi_dir", required=True, help="WSI本体(.svs等)を含むディレクトリ")
    ap.add_argument("--threshold_scope", choices=["none", "per_slide", "global"], required=True)
    ap.add_argument("--percentile", type=float, default=None, help="下位何%を除外するか(per_slide/globalで必須)")
    ap.add_argument("--n_patches", type=int, default=1000)
    ap.add_argument("--seed", type=int, required=True, help="再現性のための基準シード")
    ap.add_argument("--output_dir", default="data/blur_qc/patches")
    ap.add_argument("--patch_size", type=int, default=224)
    ap.add_argument("--wsi_id", default="", help="指定した1WSIのみ処理する(動作確認用)")
    args = ap.parse_args()

    if args.threshold_scope in ("per_slide", "global") and args.percentile is None:
        ap.error("--threshold_scope per_slide/global には --percentile が必須です")
    return args


def _find_wsi_path(wsi_dir: str, wsi_id: str) -> str:
    for ext in WSI_EXTENSIONS:
        candidate = Path(wsi_dir) / f"{wsi_id}{ext}"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"{wsi_id} のWSIファイルが {wsi_dir} に見つかりません")


def _compute_global_threshold(scores_dir: str, percentile: float) -> float:
    """全WSIの blur_score をプールしてパーセンタイル閾値を1回だけ算出する。"""
    all_scores = []
    for h5_path in sorted(Path(scores_dir).glob("*_blur.h5")):
        with h5py.File(h5_path, "r") as f:
            all_scores.append(f["blur_score"][:])
    if not all_scores:
        raise FileNotFoundError(f"{scores_dir} に *_blur.h5 が見つかりません")
    return float(np.percentile(np.concatenate(all_scores), percentile))


def main():
    args = parse_args()

    score_h5_paths = sorted(Path(args.scores_dir).glob("*_blur.h5"))
    if args.wsi_id:
        score_h5_paths = [p for p in score_h5_paths if p.stem == f"{args.wsi_id}_blur"]
        if not score_h5_paths:
            raise FileNotFoundError(f"--wsi_id={args.wsi_id} の blur h5 が {args.scores_dir} に見つかりません")

    print(f"Found {len(score_h5_paths)} scored WSIs to process")

    global_threshold = None
    if args.threshold_scope == "global":
        global_threshold = _compute_global_threshold(args.scores_dir, args.percentile)
        print(f"global threshold (percentile={args.percentile}): {global_threshold}")

    for h5_path in score_h5_paths:
        wsi_id = h5_path.stem[: -len("_blur")]
        wsi_path = _find_wsi_path(args.wsi_dir, wsi_id)

        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            scores = f["blur_score"][:]

        print(f"Processing: {wsi_id} ({len(scores)} candidate patches)")
        meta = sample_patches_to_memmap(
            wsi_path=wsi_path,
            coords=coords,
            scores=scores,
            threshold_scope=args.threshold_scope,
            percentile=args.percentile,
            n_patches=args.n_patches,
            patch_size=args.patch_size,
            out_dir=args.output_dir,
            seed=args.seed,
            global_threshold=global_threshold,
        )
        print(f"  -> sampled {meta['shape'][0]} patches")

    print("Done.")


if __name__ == "__main__":
    main()
