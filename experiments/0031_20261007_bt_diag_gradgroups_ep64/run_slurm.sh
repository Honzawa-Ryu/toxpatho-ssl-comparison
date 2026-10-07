#!/bin/bash
# =====================================================
# exp 0031 — 0030 と同じ診断ラン(ep64 から再開・bf16・step ログ)に**グループ別 grad_norm** を足して再現する。
#
# 0030 で起点は「勾配の急増(grad_norm 17→29→99→639→2219, 約4 step)」まで絞れた(H1 の bf16 桁落ちは否定)。
# ただし合計ノルムしか無く、**どのグループが先に跳ねたか**が分からなかった。
#   H2' : bias/LayerNorm(ndim<=1, --lars_exclude_bias_bn で LARS 適応なし)は生の勾配 × lr_bias 0.0384 で
#         更新されるので、勾配の急増がそのまま更新の急増になり正のフィードバックで数 step で破綻する。
#         → gn_raw が gn_lars より先に(または同時に)跳ね、top1 が bias/LN なら支持。
#   代替: 重み側(gn_lars)・特に projection_head の最終層が先なら、LARS の trust ratio か頭の問題。
# テレメトリの内容は lib/trainer/bt_telemetry.py GradGroupProbe(gn_lars / gn_raw / gn_backbone / gn_head /
# 上位3テンソル)。読み方は scripts/analysis/bt_step_report.py。
#
# ■ 0030 からの変更: --dir_result と名前だけ。コードは GradGroupProbe の追加分のみ(学習には影響しない)。
#   0030 と同じ ep64 の state.pt(0029 由来)から再開する。NaN が出る step は確率的(0029: ep65, 0030: ep66)。
#
# ■ 結果の読み方
#   python3 scripts/analysis/bt_step_report.py outputs/0031_20261007_bt_diag_gradgroups_ep64 --epoch <NaN の epoch>
#   「グループ別 grad_norm」の表で、基準の 2 倍を初めて超えた step がグループごとに出る。
#
# ■ コスト: 0030 の実績は 4.5 分(ep66 で NaN)。walltime 1h × 8ノード = 最大 8 ノード時間、実質 1〜2。
#
# 準備と投入:
#   bash experiments/0031_20261007_bt_diag_gradgroups_ep64/preflight.sh   # 0029 の state.pt をコピーして点検(qsub はしない)
#   qsub experiments/0031_20261007_bt_diag_gradgroups_ep64/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0031_bt_diag
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/0031_20261007_bt_diag_gradgroups_ep64/
#PBS -e logs/0031_20261007_bt_diag_gradgroups_ep64/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0031_20261007_bt_diag_gradgroups_ep64"

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
# 0029 からの変更は次の3点だけ(0030 と同一):
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
    --note bt_diag_gradgroups_ep64 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
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

