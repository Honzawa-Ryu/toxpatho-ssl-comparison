#!/bin/bash
# =====================================================
# exp 0047 — BT 0035 の保存済みスナップショット(32 本、scripts/analysis/bt_epochs.yaml)で val 損失を学習と同じ条件で測り直す
#            (debug-g・1 ノード・30 分。学習はしない)。
#
# 0035 の学習中の val_loss(9,000〜15,000、上昇傾向)は、eval モードの BN(running 統計)と実効 N=256 で測った値で、
# 学習の良し悪しを表していなかった(PROJECT_STATUS 2026-10-16)。scripts/analysis/bt_val_recompute.py が各スナップショットを
# val fold の固定バッチ(2048 枚 × 10 = 学習の global batch と同じ N)で、
#   batch 統計の BN(running 統計は更新しない)・bf16 … 学習と同じ条件(本命)
#   eval モードの BN・bf16                             … 学習中の val と同じ条件(参照)
# の 2 通りで測り、損失と内訳(on_diag / λ·off_diag)を outputs/0047_.../bt_val_recompute.jsonl に書く。
# 読み方: batch-BN の損失が epoch とともに train_loss(145.7 @ep1600)に近い値で下がっていれば汎化しており、
#         上がっていれば本当に val で冗長性が増えている(過学習)。eval-BN の値は学習中の val_loss の再現。
# 勾配は取らないので 1 GPU に batch 2048 が載る見込み(学習は batch 256 + 勾配で 35GB)。
#
# 準備と投入:
#   bash experiments/0047_20261016_bt_val_recompute/preflight.sh
#   qsub experiments/0047_20261016_bt_val_recompute/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0047_btval
#PBS -q debug-g
#PBS -W group_list=gd43
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/0047_20261016_bt_val_recompute/
#PBS -e logs/0047_20261016_bt_val_recompute/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0047_20261016_bt_val_recompute"

export SCRATCH_ROOT="/local"
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=0
DATA_SUBDIRS=(
    "ssl_patches"
)

RUN_MODE="single"
RUN_COMMAND="python ${PROJECT_ROOT}/scripts/analysis/bt_val_recompute.py \
    --exp 0035_20261011_bt_vitb16_lrbias_quarter_clip1 --methods ${PROJECT_ROOT}/scripts/analysis/bt_epochs.yaml \
    --out ${PROJECT_ROOT}/outputs/${EXP_NAME}/bt_val_recompute.jsonl --batches 10 --batch 2048"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
