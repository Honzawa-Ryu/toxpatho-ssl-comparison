#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 8ノード(GH200 120GB×8)マルチノードDDP。batch 256/GPU × 8 = 2048。
#
# 【Barlow Twins ViT-B/16 — 元論文の step 数にそろえた初回ラン】
# 0015(未投入)を置き換える。0015 は 100 epoch・--collapse_early_stop 無しだった。
#
# ■ 論文/公式コードに対する立ち位置（「論文準拠」と一語で書かないこと）
#   そろえた点: optimizer=LARS, lr 0.2/0.0048 × bs/256 (=1.6/0.0384 @2048),
#     wd 1.5e-6 (bias/BN は wd も LARS 適応も除外), λ=5e-3, 投影ヘッド 8192-8192-8192,
#     batch 2048, 総 step 数, warmup の step 数。
#   逸脱した点:
#     1. backbone が ViT-B/16 (論文は ResNet-50)。LARS が ViT で安定かは未検証。
#     2. epoch 数: 論文 1000 epoch × 625 step = 625,000 step。本データは train 800,000 パッチ
#        (1M 中 200k=val fold)なので 390 step/epoch。DINO(0017/0028)が論文の step 数に
#        そろえて 480 epoch にしたのと同じ考え方で、**1600 epoch = 624,000 step**(論文の 99.8%)。
#        ⚠️ サンプル総数は論文の 1.28B に対し 1.28B(=0.8M×1600)でほぼ一致するが、
#        画像の多様性は ImageNet より大幅に低い(1スライド内のパッチは互いに似る)。
#     3. warmup: 論文 10 epoch = 6,250 step → 16 epoch(=6,240 step)。
#     4. lr 終端: 論文 end_lr_ratio 0.001、ここは --lr_min 0.0 (差はピーク比 0.1% 以下)。
#     5. 精度: 論文は AMP(fp16)、ここは bf16 autocast(損失も bf16 で返る)。
#
# ■ 崩壊したら止める（Goal.yaml 方針3, 2026-09-30）
#   --collapse_early_stop。NaN/inf の train_loss は即停止(CollapseMonitor.check_loss_finite)、
#   それ以外は rank_monitor_interval=5 × patience=2 の uniformity / eff_rank 判定。
#   DINO の ln(out_dim) 相当の損失天井判定は入れていない(BT の損失は batch 依存が強く、
#   B=2048 の実測なしに閾値を決められないため)。初回ランの損失曲線を見て決める。
#   止まったら原因を切り分けて修正し、**新しい実験番号**で再実行する。
#   修正候補の第一は optimizer (LARS -> AdamW)。ただし逸脱になるので必ず記録する。
#
# ■ コスト見積り（DINO 0028 の実測 6.9分/epoch@4ノード からの外挿。BT の実測は無い）
#   BT は DINO の約 0.48 倍の計算量 => 0.22〜0.35 ノード時間/epoch
#   1600 epoch = 約 350〜560 ノード時間、8ノードで約 44〜69 時間 => 48h では終わらず
#   --resume で 2 本目が必要(resume.sh)。
#
# ■ 点検済み/未検証
#   - 検証済み: qsub/PBS/SIF/local SSD 等のサイト固有値は 0028 と同一(実機で完走)。
#     CPU スモーク(ログインノード, B=8)でヘッド 768->8192->8192->8192・損失・勾配が有限。
#   - 未検証: 8ノード同時の 141GB コピー時間(課金される)、毎 step 8192^2 fp32(約270MB)の
#     all_reduce、batch 256/GPU のメモリ(DINO は 256×(2+8小)で収まっている)、
#     BT の実測スループット。
#   - ⚠️ ログインノードでプロジェクト直下の `uv run` / `uv sync` を叩かないこと(--no-project)。
#     .venv が壊れて全ノード即死する(env/CONTAINER.md)。
#
# 投入前提: `mkdir -p logs/0029_20260930_bt_vitb16_step_matched` のうえで
#   bash experiments/0029_20260930_bt_vitb16_step_matched/preflight.sh   # 点検のみ(.venv 等)
#   qsub experiments/0029_20260930_bt_vitb16_step_matched/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0029_bt_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0029_20260930_bt_vitb16_step_matched/
#PBS -e logs/0029_20260930_bt_vitb16_step_matched/


module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0029_20260930_bt_vitb16_step_matched"

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
# Multi-node DDP (8 nodes)
#
# lr: 本リポジトリの --lr / --lr_bias は「最終lr」をそのまま渡す実装で、--ddp_linear_scale_lr は
#   --lr しか world_size 倍しない(lr_bias 側に同等のフラグが無い)ため使わず、両方とも
#   計算済みの最終値を渡す。公式の lr = base_lr × batch/256 に従うと
#     --lr      = 0.2    × 2048/256 = 1.6
#     --lr_bias = 0.0048 × 2048/256 = 0.0384
#
# epoch: 1600 (= 624,000 step)。steps/epoch = floor(800000 / 2048) = 390 (drop_last)。
# warmup: 16 epoch (= 6,240 step)。
# 保存: 5 epoch ごと(約0.9GB/本 × 320本 ≈ 290GB。group quota 20T に対し現在 1.46T)。
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note bt_vitb16_step_matched --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name barlowtwins \
    --optimizer lars --lr 1.6 --lr_bias 0.0384 --lars_exclude_bias_bn \
    --weight_decay 1.5e-6 \
    --batch_size 256 \
    --proj_dim 8192 --bt_lambda 5e-3 \
    --num_epoch 1600 --warmup_t 16 --lr_min 0.0 \
    --collapse_early_stop \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
