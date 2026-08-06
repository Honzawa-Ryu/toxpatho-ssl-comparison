#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — スパコン(Miyabi等のPBS Pro環境)向け。
# GPU 1台(96GB想定)・DDP無し(単一プロセス python 実行)。
#
# サイト固有の確認事項(#PBS -q/-P/-l select、apptainer利用可否、SIF_PATH/.env準備)
# は experiments/0014_20260806_paper_simsiam_vitb16/run_slurm.sh 冒頭のコメントを参照。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0017_20260806_paper_dino_vitb16`
# してから `qsub experiments/0017_20260806_paper_dino_vitb16/run_slurm.sh`。
# =====================================================
#PBS -N 0017_20260806_paper_dino_vitb16
#PBS -q <FIXME: queue name>
#PBS -P <FIXME: project/account code>
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=196:00:00
#PBS -j oe
#PBS -o logs/0017_20260806_paper_dino_vitb16/
#PBS -e logs/0017_20260806_paper_dino_vitb16/

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0017_20260806_paper_dino_vitb16"

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
# wsi-ad experiments/20260714_paper_dino_vitb16/run_slurm.sh のpaper-faithfulな
# CLI引数(DINO, Caron et al. 2021, ViT-B/16)を踏襲。GPU 1台なのでDDP無し。
#
# batch_size 128: wsi-adの単一GPU(A6000 48GB)実績値。4手法中もっとも小さいのは
# multicrop(2x224 + 8x96 = 1サンプルにつき10 views)で単位サンプルあたりの
# メモリ消費が大きいため。96GBならもっと大きく出来る可能性はあるが未検証
# (このプロジェクトのexperiments/0013.../vram_probe.pyと同じ手法で
# Miyabi上で実測してから増やすのが安全)。batch_sizeを変える場合、lrはlinear
# scaling rule (base_lr 2.5e-4 相当 @ bs128) で再計算すること。
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_dino_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 2.5e-4 \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --batch_size 128 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 100 --warmup_t 10 --lr_min 1e-6 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
