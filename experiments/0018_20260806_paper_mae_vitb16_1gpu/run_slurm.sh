#!/bin/bash
#SBATCH --job-name=0018_20260806_paper_mae_vitb16_1gpu
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0018_20260806_paper_mae_vitb16_1gpu/%j_0018_20260806_paper_mae_vitb16_1gpu.out
#SBATCH --error=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0018_20260806_paper_mae_vitb16_1gpu/%j_0018_20260806_paper_mae_vitb16_1gpu.out
#SBATCH --signal=B:USR1@7056
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110g
#SBATCH --time=196:00:00

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0018_20260806_paper_mae_vitb16_1gpu/%A_%a_0018_20260806_paper_mae_vitb16_1gpu.out
# #SBATCH --error=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0018_20260806_paper_mae_vitb16_1gpu/%A_%a_0018_20260806_paper_mae_vitb16_1gpu.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison"
export EXP_NAME="0018_20260806_paper_mae_vitb16_1gpu"

# =====================================================
# 0013(0013_20260805_paper_mae_vitb16, 2GPU DDP版)からの切り替え。
# job 8286で初回collective(ALLGATHER)が、job 8287で1時間以上正常稼働した後
# 勾配のALLREDUCEが、いずれもNCCL 10分watchdog timeoutでハング(SIGABRT)。
# NCCL_P2P_DISABLE=1で初回は回避できたが後者は再発 → このノードのNCCL経路が
# 断続的に不安定と判断。スパコン側(Miyabi)はもともとGPU1台・DDP無しの方針
# (0016_20260806_paper_mae_vitb16_pbs)なので、ローカルでもDDPの安定性を
# 追いかける実益は薄いと判断し、こちらは単一GPUで新規に切った実験。
# batch_size/lrは0016と同じ(single-GPUなのでworld_size=1、effective batch=768で
# 変わらないためlrも同じ4.5e-4)。
# =====================================================

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage
# /workspace はNFS（遅い）、/scratch はノード付属のm.2 SSD（速い）。
#
# USE_LOCAL_SSD_INPUT はデフォルト0（NFSを直読み）。data/ 全体をコピーすると
# 実験に不要なデータまで毎回転送して起動が遅くなるため、有効化する場合は
# 必ず DATA_SUBDIRS で実際に読むサブディレクトリだけを列挙すること。
# 有効化した場合、実験コード（experiment.py）側は project_root ではなく
# 環境変数 DATASET_DIR 経由でデータを読むこと（でないとコピーが無駄になる）。
#
# ⚠️ /scratch 側（SCRATCH_DIR）はジョブ終了時に自動削除されない
#    （rm -rf の誤削除リスクを避けるため）。出力は自動で /workspace/outputs/
#    へ回収されるが、SCRATCH_DIR自体は残るので、ディスクを圧迫してきたら
#    slurm.out に出る警告に従って手動で消すこと（詳細はdocs/USAGE.md 3-2節）。
# =====================================================

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=1

# USE_LOCAL_SSD_INPUT=1 にする場合のみ、コピー対象を列挙する
# （data/ からの相対パス。空のままだと data/ 全体をコピーする後方互換動作になる）。
DATA_SUBDIRS=(
    "ssl_patches"
)

# =====================================================
# python path
#
# experiments/${EXP_NAME}/experiment.py ではなく、本番の学習エントリポイント
# (scripts/train/train_tggate.py、実体は lib/trainer/entry.py)を直接叩く。
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/scripts/train/train_tggate.py"

# =====================================================
# Single run（デフォルト）
#
# GPU1台・DDP無し(torchrunではなくpython単体で起動)。
# batch_size/lrは0016_20260806_paper_mae_vitb16_pbsと同じ
# (0013のvram_probe.pyでA6000 48GB上でbs=768実測済み、peak 42.10GB=83%)。
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_mae_vitb16_1gpu --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name mae \
    --optimizer adamw --beta2 0.95 --lr 4.5e-4 --weight_decay 0.05 \
    --batch_size 768 \
    --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
    --num_epoch 100 --warmup_t 40 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--model" "--dataset")
#   GRID_VALUES=("bert roberta" "pubmed pmc")
#   → --model bert --dataset pubmed / --model bert --dataset pmc / ...
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it"
#     "BC5CDR BIORED"
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
#     "--model"
# )
# GRID_VALUES=(
#     "bert roberta"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
