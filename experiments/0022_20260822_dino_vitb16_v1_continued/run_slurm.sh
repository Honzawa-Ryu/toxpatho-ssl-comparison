#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。0017の継続学習(fresh cosineで再加速)。
#
# 背景: 0017(job 2522081)は100epoch完走時点でtrain_loss 4.32とまだ下降中
# だった。0021(paper-faithfulなteacher momentum/温度スケジュールへの変更)は
# epoch15で恒久collapse(未解決、別途調査中)。その結論を待たず、まずは
# 素直に学習が伸びていた0017を続きから伸ばす。
#
# DINOのteacher momentum/teacher温度/lr・wdスケジュール粒度は0017と同一の
# 固定値運用のまま(0021で試した可変スケジュール差分は今回使わない。恒久崩壊の
# 主要容疑がかかっているため、原因調査が終わるまでは切り離す。差分は
# `git stash list` に退避済み: "0021 paper-faithful DINO schedule ...")。
# 0017とのコード差分は以下のCLI引数のみ:
#   1. --model_path で 0017 の model_ssl.pt から重みだけをウォームスタート
#      (student/teacher双方。lib/trainer/model.py の --model_path 読み込みを
#      今回有効化した)。--resume(state.pt引き継ぎ)はしない — timmの
#      CosineLRScheduler は cycle_limit=1 固定で t_initial を超えるとlr_minに
#      張り付くため、延長分は新しいcosineスケジュールで走らせる。
#   2. --lr 1e-4 (0017の5e-4から1/5に低下)。収束済みモデルに対して元と
#      同規模のwarmup→再加速をかけると再collapseの引き金になりかねないため、
#      延長フェーズのピークlrを抑えたもの。
#   3. --early_stop --patience 25 --delta 0 (停滞したら止める。デフォルト
#      patience=7より大幅に広く取り、まだ下がっている間は止めない)。
#   4. --collapse_early_stop (新規実装。effective_rankがrank_threshold=5.0を
#      patience=2回=10epoch連続で下回ったら打ち切る。0021の実ログで検証済み:
#      0021(恒久崩壊)はepoch40で検知・0017(一時的低下から回復)は誤検知なし)。
# 詳細は lib/trainer/model.py の --model_path 読み込み・
# lib/sslmodel/utils.py の CollapseMonitor・lib/trainer/loop.py の
# collapse監視フック(2026-08-22)を参照。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0022_20260822_dino_vitb16_v1_continued`
# してから
#   `qsub experiments/0022_20260822_dino_vitb16_v1_continued/run_slurm.sh`
# =====================================================
#PBS -N 0022_dino_vitb16_v1_continued
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0022_20260822_dino_vitb16_v1_continued/
#PBS -e logs/0022_20260822_dino_vitb16_v1_continued/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0022_20260822_dino_vitb16_v1_continued"

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
# Multi-node DDP (4 nodes) — ノード数・batch_size・その他ハイパラは0017と同一。
# --model_path は永続領域(outputs/、ローカルSSDへ退避されない)上の0017の
# 最終チェックポイントを直接指す。DINOのmomentum/teacher温度はコード側の
# 固定値(0.9995/0.04、0017と同一)を使うため、CLI引数では指定しない
# (0021用のmomentum_start等の可変スケジュールCLI引数は今回未使用)。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note dino_vitb16_v1_continued --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --model_path ${PROJECT_ROOT}/outputs/0017_20260806_paper_dino_vitb16/model_ssl.pt \
    --optimizer adamw --lr 1e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --batch_size 256 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 100 --warmup_t 10 --lr_min 1e-6 \
    --early_stop --patience 25 --delta 0 \
    --collapse_early_stop \
    --rank_monitor_interval 5 --save_interval 5"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
