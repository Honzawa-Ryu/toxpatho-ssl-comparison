#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。
#
# 【診断B】DINOの恒久崩壊要因の切り分け: teacher温度 軸だけを論文値に寄せる。
#
# 背景と全体の枠組みは診断A
# (experiments/0023_20260829_dino_diag_momentum_paper/run_slurm.sh の冒頭コメント)
# を参照。AとBは「0021で同時に変えた2つの軸」を1つずつ分離したペアで、
# それ以外の設定は0017と完全に同一に揃えてある。
#
# 本実験(B)で論文へ寄せるのは teacher温度だけ:
#   --dino_teacher_temp_end 0.07 --dino_teacher_temp_warmup_epochs 30
#     論文(Caron et al. 2021)の線形warmup 0.04 -> 0.07(最初30epoch)、以降0.07固定。
#     teacher温度が高い = sharpening(崩壊回避力)が弱い方向なので、0.04固定は
#     意図的なanti-collapse設定。0.07へ上げることが崩壊の引き金かを見る。
#   momentumは指定しない = 0.9995固定(0017と同一, anti-collapse側)。
#   lr/wdのスケジュール粒度もepoch単位のまま(0021の3つ目の変更は持ち込まない)。
#
# ⚠️ --num_epoch を480のままにしてある理由、および480の根拠は診断Aのコメント参照。
#   なお温度warmupは「30 epoch」という絶対値指定なので地平線には依存しないが、
#   lr cosineは地平線依存なので、Aと条件を揃える意味でも480のままにしてある。
#
# ⚠️ 観察には最低30 epoch必要。温度は30epochかけて0.04→0.07へ上がりきるため、
#   0021が崩壊した epoch15 時点ではまだ0.055までしか上がっていない。
#   walltime 5h ≈ 45 epoch なので warmup 完了後も約15 epoch観察できる。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0024_20260829_dino_diag_teacher_temp_paper`
# してから
#   `qsub experiments/0024_20260829_dino_diag_teacher_temp_paper/run_slurm.sh`
# 両方同時に投げる場合は計8ノード必要になる点に注意。
# =====================================================
#PBS -N 0024_dino_diag_teacher_temp_paper
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -o logs/0024_20260829_dino_diag_teacher_temp_paper/
#PBS -e logs/0024_20260829_dino_diag_teacher_temp_paper/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0024_20260829_dino_diag_teacher_temp_paper"

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
# Multi-node DDP (4 nodes) — 投入設定・ハイパラは0017と同一。
# 0017との差分は --dino_teacher_temp_end / --dino_teacher_temp_warmup_epochs /
# --num_epoch / --collapse_early_stop の4点のみ。
# lr の考え方は診断Aのコメント参照(--ddp_linear_scale_lr で 5e-4 × 4 = 2e-3)。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note dino_diag_teacher_temp_paper --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --batch_size 256 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 480 --warmup_t 10 --lr_min 1e-6 \
    --dino_teacher_temp_end 0.07 --dino_teacher_temp_warmup_epochs 30 \
    --collapse_early_stop \
    --rank_monitor_interval 5 --save_interval 5"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
