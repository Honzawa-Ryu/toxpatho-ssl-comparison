#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — スパコン(Miyabi等のPBS Pro環境)向け。
# GPU 1台(96GB想定)・DDP無し(単一プロセス python 実行)。
#
# サイト固有の確認事項(#PBS -q/-P/-l select、apptainer利用可否、SIF_PATH/.env準備)
# は experiments/0014_20260806_paper_simsiam_vitb16/run_slurm.sh 冒頭のコメントを参照
# (4手法とも同じ注意点)。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0015_20260806_paper_barlowtwins_vitb16`
# してから `qsub experiments/0015_20260806_paper_barlowtwins_vitb16/run_slurm.sh`。
# =====================================================
#PBS -N 0015_20260806_paper_barlowtwins_vitb16
#PBS -q <FIXME: queue name>
#PBS -P <FIXME: project/account code>
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=196:00:00
#PBS -j oe
#PBS -o logs/0015_20260806_paper_barlowtwins_vitb16/
#PBS -e logs/0015_20260806_paper_barlowtwins_vitb16/

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0015_20260806_paper_barlowtwins_vitb16"

# export SIF_PATH="${PROJECT_ROOT}/env/env.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage (理由はEXP14冒頭コメント参照。data/ssl_patchesのNFSランダム読みは遅い)
# =====================================================

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=1
DATA_SUBDIRS=(
    "ssl_patches"
)

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/scripts/train/train_tggate.py"

# =====================================================
# Single run（デフォルト）
#
# wsi-ad experiments/20260714_paper_barlowtwins_vitb16/run_slurm.sh のpaper-faithful
# なCLI引数(Barlow Twins, ViT-B/16)をそのまま踏襲。GPU 1台なのでDDP無し。
#
# batch_size 256: wsi-adの単一GPU(A6000 48GB)実績値。96GBならもっと大きく出来る
# はずだが未検証。Barlow Twinsのcross-correlation lossはバッチ統計に依存するため、
# batch_sizeを変えるなら以下2点に注意:
#   - lr(weights)はlinear scaling rule (base 0.2 @ bs256) で再計算
#     (lr_biasも同様、base 0.0048 @ bs256)
#   - 損失の性質自体がbatch_size依存(単純に大きくすると論文の意図から外れる方向にも
#     働きうる)なので、上げるなら論文のAppendix(batch size ablation)を確認推奨
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_barlowtwins_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name barlowtwins \
    --optimizer lars --lr 0.2 --lr_bias 0.0048 --lars_exclude_bias_bn \
    --weight_decay 1.5e-6 \
    --batch_size 256 \
    --proj_dim 8192 --bt_lambda 5e-3 \
    --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
