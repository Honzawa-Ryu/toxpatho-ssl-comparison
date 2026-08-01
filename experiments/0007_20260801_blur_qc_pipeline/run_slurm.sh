#!/bin/bash
#SBATCH --job-name=0007_20260801_blur_qc_pipeline
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/logs/0007_20260801_blur_qc_pipeline/%j_0007_20260801_blur_qc_pipeline.out
#SBATCH --error=/workspace/logs/0007_20260801_blur_qc_pipeline/%j_0007_20260801_blur_qc_pipeline.out
#SBATCH --signal=B:USR1@30
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# GPU不要のCPU処理(OpenSlideデコード + OpenCV)のためgpus指定なし(Goal.md §7)。

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の --output/--error の2行と直後の --array を
# 以下の3行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/logs/0007_20260801_blur_qc_pipeline/%A_%a_0007_20260801_blur_qc_pipeline.out
# #SBATCH --error=/workspace/logs/0007_20260801_blur_qc_pipeline/%A_%a_0007_20260801_blur_qc_pipeline.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gpus/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace"
export EXP_NAME="0007_20260801_blur_qc_pipeline"

# =====================================================
# Storage
# /workspace はNFS（遅い）、ノード付属のSSD（.env の LOCAL_SSD_DIR、
# 未定義なら /scratch/${USER}）は速い。デフォルトで有効。
# NFS越しに直接読み書きしたい場合のみ0にする
# （例: 出力を実行中に /workspace 側から監視したい等）。
# =====================================================

# 対象は2 WSIのみで data/ 全量(WSI 998枚 + trident_processed)の転送は
# 不要かつ無駄なため、このジョブでは無効化してNFSを直接読み書きする。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

# data/ 丸ごとではなく必要なサブパスだけをSSDへ転送したい場合に列挙する
# （空白区切り、data/ からの相対パス）。未指定なら data/ 全量を転送する。
# TG-GATE 規模（224² × 100万枚の uint memmap ≒ 150GB）では毎ジョブの全量
# 転送が効いてくるので、必要な fold だけに絞ること（REFACTOR_PLAN.md §7-2）。
# 転送時間は logs/{exp}/{job_id}/run_metadata.yaml の
# input_staging_seconds / input_staging_bytes に記録される。
#
# export LOCAL_SSD_INPUT_PATHS="shards/fold0 shards/fold1"

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH}"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
# 4. experiment.py の parse_args() と build_variant_key() に同じ次元を足す
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--ssl_name" "--model_name")
#   GRID_VALUES=("barlowtwins simsiam" "ResNet50 ViTB16")
#   → --ssl_name barlowtwins --model_name ResNet50 / ... の4通り
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--ssl_name"
#     "--model_name"
# )
# GRID_VALUES=(
#     "barlowtwins simsiam mae dino swav"
#     "ViTB16"
# )

# =====================================================
# Seq run にしたい場合（1ジョブ内でGRIDを順次実行）
#
# 上と同様に RUN_MODE="seq" にし、BASE_COMMAND/GRID_ARGS/GRID_VALUES を設定する。
# こちらは #SBATCH --array は不要（1ジョブでループするため）。
# =====================================================

# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--ssl_name"
# )
# GRID_VALUES=(
#     "barlowtwins simsiam"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
