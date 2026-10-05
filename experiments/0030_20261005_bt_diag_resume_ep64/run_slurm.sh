#!/bin/bash
# =====================================================
# exp 0030 — 0029 の NaN(ep65)を再現して観察する**診断ラン**。短時間・8ノード。
#
# 0029 は ep64 まで損失・grad_norm とも単調に下がっていたのに、ep65 の epoch 平均が突然 NaN
# になった。ログが epoch 平均だけで原因を切り分けられなかった。ep64 の健全な state.pt
# (outputs/0029_.../state.pt)から再開し、step 単位のテレメトリ(--bt_step_log)を付けて
# NaN を再現・観察する。
#
# ■ 仮説の現状（2026-10-05）
#   H1(bf16 の標準化の桁落ち): ep64 の重みで実測した結果、前提が成り立っていない。z の
#       |mean|/std は中央値 0.11・最大 0.63(bf16 で std が 0 になるのは数百以上)、bf16 に丸めても
#       std==0 の次元は 0/8192、損失差 0.4%。**定常状態の桁落ちでは説明できない**
#       (B=32・実パッチ・順伝播のみの測定。ep65 の 1 epoch の間に急変した可能性は残る)。
#   H2(LARS・ピーク lr 1.6 の不安定): 未検証。前兆が epoch 平均に出ていない点は不利。
#   他: 特定の壊れたバッチ、backbone 側の bf16 の溢れ、など。step ログで切り分ける。
#
# ■ 結果の読み方
#   - NaN が再現した: bt_steps_ep*.jsonl の最初の非有限 step の直前を見る。z_std_min / n_zero_std /
#       z_ratio_max が先に壊れていれば H1、grad_norm が先に跳ねていれば H2。
#   - 再現しなかった(10〜15 epoch 走って健全): 確率的な事象。その場合の対処は別途検討する。
#
# ■ コスト: walltime 1h × 8ノード = 最大 8 ノード時間(予約 8.0)。NaN が出れば即停止する。
#   ⚠️ 起動時の 141GB コピーの所要時間は未計測。長ければ走れる epoch が減る。
#
# 準備と投入:
#   bash experiments/0030_20261005_bt_diag_resume_ep64/preflight.sh   # 0029 の state.pt を新ディレクトリへコピーして点検(qsub はしない)
#   qsub experiments/0030_20261005_bt_diag_resume_ep64/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0030_bt_diag
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/0030_20261005_bt_diag_resume_ep64/
#PBS -e logs/0030_20261005_bt_diag_resume_ep64/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0030_20261005_bt_diag_resume_ep64"

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
# Multi-node DDP (8 nodes) — 0029 と同一(batch 256 x 8 = 2048, LARS lr 1.6 / lr_bias 0.0384,
# wd 1.5e-6, 1600 epoch, warmup 16)。**スケジュールを変えないこと**: num_epoch が違うと
# cosine の位置が変わり、ep65 の lr が 1.6 でなくなって再現にならない。
#
# 0029 からの変更は次の3点だけ:
#   --bt_step_log                  step 単位の損失・grad_norm・z 統計を bt_steps_ep*.jsonl に記録
#   --collapse_ignore_uniformity   外す(BT の uniformity は実測 -2.9〜-3.4 で健全。有効に戻す)
#   --dir_result                   新しいディレクトリ(0029 の成果物を上書きしない)
# 損失は bf16 のまま(--bt_fp32_head は付けない): まず 0029 の NaN を**そのまま再現**して
# step 単位で観察するため。fp32 にして再現しなくなっても、それは原因の特定にならない。
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note bt_diag_resume_ep64 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name barlowtwins \
    --optimizer lars --lr 1.6 --lr_bias 0.0384 --lars_exclude_bias_bn \
    --weight_decay 1.5e-6 \
    --batch_size 256 \
    --proj_dim 8192 --bt_lambda 5e-3 \
    --num_epoch 1600 --warmup_t 16 --lr_min 0.0 \
    --collapse_early_stop --collapse_loss_rebound 2.0 \
    --bt_step_log \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"

