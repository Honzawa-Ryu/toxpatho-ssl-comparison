#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 2ノード(GH200 120GB×2)マルチノードDDP。batch 256/GPU × 2 = 512(論文・公式の既定 batch と同じ)。
#
# 【SimSiam ViT-B/16 — 0046 から SGD の lr をさらに半分(0.025)にした本番(0049 と並列)】
# ■ 0046 からの変更は **lr 0.05 → 0.025** の 1 点だけ(公式 0.05 × bs/256 = 0.1 の 1/4。逸脱)。
#   0.1 → 0.05 で部分崩壊が弱まった(90% 成分数 4〜7 → 23)ので、その延長。predictor の固定 lr も 0.025 になる。
#
# ■ 0046 の結果(2026-10-16): 160 epoch 完走。学習中の監視(⚠️ 3 スライドの条件)で 90% 成分数 23・画像間 cos 0.993。
#   0038(SGD lr 0.1: 4〜7)より多くの方向が残るが部分崩壊は解消せず。0045(AdamW)は悪化(ep61 で停止)。
#   → lr を下げる方向は効く。AdamW は効かない。augmentation は 4 手法で揃えているので変えない(最後の手段)。
# ■ 判定: 学習中の評価(5 epoch ごと、多スライドのバッチに修正済み)の n_comp_90 / pc1_ratio / mean_cos を 0046 と比べる。
#   崩壊は ep10〜20 で見えるので 2〜3 時間で判定できる。
#
# 以下は 0046 の説明をそのまま引き継ぐ。
# 【SimSiam ViT-B/16 — 0038 から SGD の lr を半分にした対照(0045 の AdamW と並列)】
# ■ 0038 からの変更は **lr 0.1 → 0.05** の 1 点だけ(SGD のまま)。0045(AdamW)と並べて、0038 の崩壊が
#   「lr が大きすぎる」問題か「SGD という種類」の問題かを切り分ける対照。公式(0.05 × bs/256)からの逸脱。
#   predictor の固定 lr(fix_pred_lr)も 0.05 になる。高速化の --compile_blocks も付ける(学習の計算は同じ)。
#
# ■ 0038 の診断(andre01、固定タイル 4,080 枚の backbone CLS 特徴、2026-10-15): **部分的な次元崩壊**。
#   分散の 90% が 4〜7 方向に集まり(DINO 0028 は 68〜371)、第 1 主成分が 39〜72%、別々の画像の特徴の cos 平均 0.87〜0.98。
#   完全崩壊ではない(実効ランク 27〜98、止まった次元なし)。**ep10(総学習量の 6%)から既にこの状態**で、残った方向は
#   スライド/試験の違いを多く含む(η²_slide 0.69〜0.82、DINO は 0.40〜0.65)。終盤にヘッド(z・p)だけ回復したのは、
#   backbone の lr がほぼ 0 なのに predictor の lr が固定(fix_pred_lr)だったためと見られる(未検証)。
# ■ 判定: n_comp_90 / pc1_ratio / mean_cos を 5 epoch ごとの評価でログと WandB に出す(2026-10-15 追加)。崩壊は ep10〜20 で
#   既に見えるので、そこまで(2〜3 時間)で 0038 と比べられる。
#
# 以下は 0038 の説明をそのまま引き継ぐ。
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
#   7. augmentation に 0〜180 度の連続回転(毎回)が入る(全手法共通の ssl_transform)。
#
# ■ ログ(--step_log、学習には影響しない): steps_ep*.jsonl に step ごとの損失・grad_norm・グループ別 grad_norm・
#   上位テンソルと、SimSiam の崩壊統計(z_erank / p_erank = 実効ランク、cos_zz = 2 ビューの z の一致度、
#   z_nstd_sqrtd = 論文 Fig.2 の指標)。⚠️ 出力 BN が次元ごとの std を 1 に揃えるので、std 系の指標は
#   ランク崩壊を見逃しうる。崩壊は z_erank の急落と cos_zz → 1 で見る。backbone の実効ランクは 5 epoch ごとの eff_rank。
#
# ■ コスト(BT 0033 の実測からの外挿): BT は 8 ノードで 2.27 分/epoch(2 ビュー × フル ViT-B、256/GPU)。
#   SimSiam は 1 サンプルあたりほぼ同じ計算(投影ヘッドが小さい分わずかに軽い)で、GPU が 1/4 →
#   約 9 分/epoch × 160 epoch ≈ 24h ≈ **48 ノード時間**。48h 枠に収まる見込み。
#
# 投入前提:
#   bash experiments/0048_20261016_simsiam_vitb16_sgd_lr_quarter/preflight.sh   # 点検のみ
#   qsub experiments/0048_20261016_simsiam_vitb16_sgd_lr_quarter/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0048_simsiam_sg
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=2
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0048_20261016_simsiam_vitb16_sgd_lr_quarter/
#PBS -e logs/0048_20261016_simsiam_vitb16_sgd_lr_quarter/


module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0048_20261016_simsiam_vitb16_sgd_lr_quarter"

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
    --note simsiam_vitb16_sgd_lr_quarter --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name simsiam \
    --optimizer sgd --lr 0.025 --fix_pred_lr --weight_decay 1e-4 \
    --batch_size 256 \
    --num_epoch 160 --warmup_t 16 --lr_min 0.0 \
    --collapse_early_stop --collapse_ignore_uniformity \
    --step_log --compile_blocks \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
