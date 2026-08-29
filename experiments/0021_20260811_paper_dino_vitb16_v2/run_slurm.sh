#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。0017の再実行(v2)。
#
# 0017(job 2522081, 100 epoch完走)との違いはコード側のみ:
#   1. teacher温度warmup(0.04→0.07, 最初30epoch線形)・teacher momentumの
#      cosineスケジュール(0.996→1.0)を実装(以前は両方とも固定値で、
#      0017ではepoch13-30/57-58で崩壊→自然回復が2回起きていた。固定値が
#      主因の一つと推測)。
#   2. lr/weight decayのスケジュール粒度をepoch単位(階段状)からstep単位
#      (滑らかなcosine/linear補間、元のDINO実装と同じ粒度)に変更。
#   3. 全SSL手法・全backboneでImageNet事前学習済み重みを使わずスクラッチ
#      初期化に統一(DINOは元々スクラッチだったので実質的な影響はない)。
# 詳細は lib/sslmodel/models/dino.py, lib/trainer/{model,loop}.py の
# 2026-08-11のコメント参照。投入設定(ノード数・batch_size・lr等)は0017と
# 同一(既に実機で無事完走している構成なので変更しない)。
#
# outputs/0017_.../model_ssl.pt が既に存在するため、0017をそのまま
# --resume 付きで再投入すると「完了済みなので何もしない」で即終了する
# (lib/trainer/entry.py:152)。新しいコードで実際に学習し直すには新しい
# 実験ディレクトリ(=新しいdir_result)が必要なため、0021として作成した。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0021_20260811_paper_dino_vitb16_v2`
# してから
#   `qsub experiments/0021_20260811_paper_dino_vitb16_v2/run_slurm.sh`
# =====================================================
#PBS -N 0021_paper_dino_vitb16_v2
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0021_20260811_paper_dino_vitb16_v2/
#PBS -e logs/0021_20260811_paper_dino_vitb16_v2/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0021_20260811_paper_dino_vitb16_v2"

export SCRATCH_ROOT="/local"
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage
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
# Multi-node DDP (4 nodes) — 0017と同一設定
# momentum_start/momentum_end/teacher_temp_start/teacher_temp_end/
# teacher_temp_warmup_epochsは全て新しいCLI引数のデフォルト値(論文準拠:
# 0.996/1.0/0.04/0.07/30)をそのまま使うため、明示指定していない。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_dino_vitb16_v2 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --batch_size 256 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 100 --warmup_t 10 --lr_min 1e-6 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
