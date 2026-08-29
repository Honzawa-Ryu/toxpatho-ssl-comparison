#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 2ノード(GH200 120GB×2)マルチノードDDP。batch_size 256/GPU × 2 = 実質512
# (論文 SimSiam, Chen & He 2021 のデフォルトbatch 512と一致)。
#
# このファイルは元々サイト未確定のFIXMEテンプレート(汎用PBS Pro想定)だった。
# experiments/0015_20260806_paper_barlowtwins_vitb16/run_slurm.sh でMiyabi実機の
# qsub投入まで検証済みのサイト固有値(キュー/課金コード/select句/apptainer/
# SIF_PATH/SCRATCH_ROOT)にここで合わせ、あわせてBSZ/LRも論文値に揃える。
# マルチノードDDP機構自体は experiments/0017_20260806_paper_dino_vitb16/
# (4ノード、batch 256/GPU、job 2522081で100 epoch完走)で実機検証済み。
#
# 要点(0015/0017と共通、詳細根拠は experiments/0019_20260806_pbs_test/
# run_slurm.sh 参照):
#   - キュー名は実行キュー(small-g等)ではなく親のルーティングキュー `regular-g`。
#   - 課金/グループは `-P` ではなく `-W group_list=gd43`。
#   - select句は `select=2`(2ノード、各ノードGPU1台・ngpusは無効resource)。
#   - SCRATCH_ROOT: ノードローカルNVMe SSDの実マウント点は `/local`。
#   - apptainer: ジョブスクリプトはmodule環境を引き継がないため明示的に
#     `module load apptainer/1.3.5` が必要。
#   - SIF_PATH: このリポジトリ専用の env/env.sif は未ビルドのため、暫定で
#     共有コンテナ pytorch-ngc-26.06.sif を使う。
#   - walltime上限: `regular-g` は48時間が上限。RUN_COMMAND側に
#     `--save_interval 5 --resume` があるので、打ち切られても同じ `qsub` で
#     直近チェックポイントから再開できる想定(実際の再開動作は未検証)。
#
# lrについて: base_lr 0.05 はbatch_size=256基準(SimSiam公式のlinear scaling
# rule: lr = base_lr × batch_size/256)。今回batch_size 256/GPU × 2ノード = 512
# なので --lr = 0.05 × 512/256 = 0.1 に変更(旧: 単一ノードbatch=256のときは
# ratio=1で --lr 0.05 のままでよかった)。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0014_20260806_paper_simsiam_vitb16`
# してから `qsub experiments/0014_20260806_paper_simsiam_vitb16/run_slurm.sh`。
# 48h経過でジョブが打ち切られたら、同じコマンドで再度 `qsub` して再開する。
# =====================================================
#PBS -N 0014_20260806_paper_simsiam_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0014_20260806_paper_simsiam_vitb16/
#PBS -e logs/0014_20260806_paper_simsiam_vitb16/

# ジョブスクリプトはmodule環境を引き継がないため明示的にロードする。
module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0014_20260806_paper_simsiam_vitb16"

# ノードローカルSSDの実際のマウント点(既定の/scratchはMiyabiに存在しない)。
export SCRATCH_ROOT="/local"

# 共有コンテナを暫定使用(このリポジトリ専用の env/env.sif は未ビルド)。
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage (理由はEXP14旧版/EXP15冒頭コメント参照。data/ssl_patchesのNFS
# ランダム読みは遅いため、ノードローカルSSDへ一度rsyncしてから読む)
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
# Multi-node DDP (2 nodes) — batch_size/lrを論文値に揃える
#
# wsi-ad experiments/20260714_paper_simsiam_vitb16/run_slurm.sh のpaper-faithful
# なCLI引数(SimSiam, Chen & He 2021, ViT-B/16)を踏襲しつつ、BSZ/LRは論文の
# 実値(batch_size=512)にここで揃える。
# =====================================================

NNODES=2
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_simsiam_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name simsiam \
    --optimizer sgd --lr 0.1 --fix_pred_lr --weight_decay 1e-4 \
    --batch_size 256 \
    --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
