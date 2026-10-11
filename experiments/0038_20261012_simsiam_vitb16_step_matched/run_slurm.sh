#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 2ノード(GH200 120GB×2)マルチノードDDP。batch 256/GPU × 2 = 512(論文・公式の既定 batch と同じ)。
#
# 【SimSiam ViT-B/16 — 元論文の step 数にそろえた本番】0014(100 epoch、未投入)を置き換える。
#
# ■ step 数の合わせ方(DINO 0028 / BT 0029 / MAE 0037 と同じ考え方)
#   SimSiam 論文(Chen & He 2021)の既定は ResNet-50・batch 512・100 epoch。
#     論文: 1,281,167 / 512 = 2502 step/epoch × 100 = 250,200 step
#     本データ: 800,000 / 512 = 1562 step/epoch(drop_last) → 250,200 / 1562 = **160 epoch**(249,920 step)
#
# ■ 公式(main_simsiam.py, README の推奨コマンド)にそろえた点
#   SGD momentum 0.9, lr = 0.05 × 512/256 = 0.1(最終 lr を直接渡す), wd 1e-4, cosine,
#   --fix_pred_lr(predictor は lr 固定), projector 3 層 + BN(隠れ層幅 = backbone 次元。公式は ResNet の 2048、
#   ここは ViT の 768。出力 2048・出力 BN affine なし)/ predictor 2048→512→2048,
#   SyncBN(world_size>1 で distributed.wrap が変換)。
# ■ 逸脱(「論文準拠」と一語で書かない)
#   1. backbone が ViT-B/16(論文は ResNet-50)。**SimSiam の ViT 用公式レシピは無い**。ViT+SGD は不安定という
#      報告がある(MoCo v3, Chen et al. 2021)。
#   2. weight decay を bias/LN から外す(公式 ResNet レシピは全パラメータ。ViT では MoCo v3 の慣行に合わせて除外。
#      外さないと lr 0.1 × wd 1e-4 × 25 万 step で LN ゲインが約 e^-2.5 まで削られる = DINO 崩壊と同じ経路)。
#   3. warmup 16 epoch(論文は batch 512 では warmup なし。論文の大バッチ設定(LARS, batch 4096)の
#      「10/100 epoch」と同じ比率)。ViT のスクラッチ学習の安定化のため。
#   4. augmentation は他手法と共通の病理向け既定(色 0.8 / blur 0.4 / grayscale 0.05)。公式(MoCo v2 aug:
#      grayscale 0.2, blur 0.5)とは違う。手法間で aug を揃えるため。
#   5. 精度: 公式は fp32、ここは bf16 autocast(他手法と同じ)。
#   6. 崩壊検知: NaN 即停止 + eff_rank<5(SimSiam の崩壊は表現が定数に潰れるので eff_rank で見える)。
#      uniformity 判定は外す(健全な ViT 特徴でも 0 付近になりうる。DINO で実測)。損失の跳ね返り判定は
#      使わない(SimSiam の損失は負なので定義できない。check_loss_rebound は min<=0 で無効)。
#
# ■ コスト(BT 0033 の実測からの外挿): BT は 8 ノードで 2.27 分/epoch(2 ビュー × フル ViT-B、256/GPU)。
#   SimSiam は 1 サンプルあたりほぼ同じ計算(投影ヘッドが小さい分わずかに軽い)で、GPU が 1/4 →
#   約 9 分/epoch × 160 epoch ≈ 24h ≈ **48 ノード時間**。48h 枠に収まる見込み。
#
# 投入前提:
#   bash experiments/0038_20261012_simsiam_vitb16_step_matched/preflight.sh   # 点検のみ
#   qsub experiments/0038_20261012_simsiam_vitb16_step_matched/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0038_simsiam_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0038_20261012_simsiam_vitb16_step_matched/
#PBS -e logs/0038_20261012_simsiam_vitb16_step_matched/


module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0038_20261012_simsiam_vitb16_step_matched"

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
# Multi-node DDP (2 nodes) — batch 256/GPU × 2 = 512
# =====================================================

NNODES=2
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note simsiam_vitb16_step_matched --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name simsiam \
    --optimizer sgd --lr 0.1 --fix_pred_lr --weight_decay 1e-4 \
    --batch_size 256 \
    --num_epoch 160 --warmup_t 16 --lr_min 0.0 \
    --collapse_early_stop --collapse_ignore_uniformity \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
