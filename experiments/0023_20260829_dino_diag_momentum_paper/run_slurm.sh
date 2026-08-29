#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。
#
# 【診断A】DINOの恒久崩壊要因の切り分け: teacher momentum 軸だけを論文値に寄せる。
#
# 背景:
#   0017(固定値 momentum 0.9995 / teacher_temp 0.04)は100epoch完走したが、
#   0021(両方を論文スケジュールへ変更 + lr/wdのstep粒度化)はepoch15で恒久崩壊した。
#   0021は3つを同時に変えたため原因が切り分けられていない。さらに 0021 の実装は
#   スケジュールを適用する側のコード(entry.pyのCLI引数・loop.pyの毎step更新)が
#   失われており、「本当にスケジュールが効いていたのか」自体が事後検証できない
#   (config.jsonには引数が記録されているが、更新呼び出しの痕跡がツリーに無い)。
#   そこで 2026-08-29 にスケジュールをopt-inで実装し直し(commit c9e39bb)、
#   1軸ずつ検証する。
#
# 本実験(A)で論文へ寄せるのは teacher momentum だけ:
#   --dino_momentum_start 0.996 --dino_momentum_end 1.0
#     論文(Caron et al. 2021)の cosine 0.996 -> 1.0。0017の0.9995固定に対し、
#     開始値0.996はteacherがstudentを約12.5倍速く追従する = 崩壊しやすい側。
#     0021もこの0.996を使っており、最有力容疑。
#   teacher温度は指定しない = 0.04固定(0017と同一, anti-collapse側)。
#   lr/wdのスケジュール粒度もepoch単位のまま(0021の3つ目の変更は持ち込まない)。
#
# ⚠️ --num_epoch は本番想定の 480 にしてある(短縮しない)。
#   momentumのcosineは地平線に対する相対進度で決まるため、診断用に
#   --num_epoch 40 などと縮めると同じepoch15でも momentum が 0.9972 まで
#   1.0側へ寄ってしまい(480地平線なら0.9960)、本番より崩壊しにくい甘い条件に
#   なってしまう。lr cosineについても同様。よって地平線は本番と同じにしたうえで、
#   walltimeを5hに絞ってその範囲(≈45 epoch)だけ観察する。
#   崩壊すれば --collapse_early_stop がそれより早く打ち切る(0021は epoch15 で崩壊)。
#
# 480 epoch の根拠(0017の100epochから変更した理由):
#   論文の300 epochは ImageNet-1k(1,281,167枚)基準。本プロジェクトの学習データは
#   800,000枚(1M中200k=200WSIはvalidation fold)なので、同じ batch 1024 では
#   781 steps/epoch にしかならない。論文300epoch = 375,300 steps に揃えるには
#   375,300 / 781 ≈ 480 epoch が必要。
#   (0017の100 epoch は 78,100 steps = ImageNet換算で約62 epoch相当しかなく、
#    大幅に未収束だった。実際100epoch時点でtrain_loss 4.32とまだ下降中。)
#
# 判定の見方:
#   - 毎epochのログに teacher_momentum / teacher_temp の実測値が出る(commit c9e39bb)。
#     まずこれでスケジュールが実際に効いていることを確認すること
#     (0021で疑われている「有効にしたつもりで固定のまま」の再発防止)。
#   - eff_rank が 5.0 を下回り続ける / train_loss が 9.0109 (= ln(8192)、出力が
#     一様分布に潰れた値) に張り付く / grad_norm が 0 付近に落ちる、が崩壊の兆候。
#   - 0017 も epoch25 で eff_rank 6.04 まで落ちてから回復しているので、
#     一時的な低下だけで崩壊と即断しないこと。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0023_20260829_dino_diag_momentum_paper`
# してから
#   `qsub experiments/0023_20260829_dino_diag_momentum_paper/run_slurm.sh`
# 診断Bは experiments/0024_20260829_dino_diag_teacher_temp_paper/。
# 両方同時に投げる場合は計8ノード必要になる点に注意。
# =====================================================
#PBS -N 0023_dino_diag_momentum_paper
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -o logs/0023_20260829_dino_diag_momentum_paper/
#PBS -e logs/0023_20260829_dino_diag_momentum_paper/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0023_20260829_dino_diag_momentum_paper"

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
# 0017との差分は --dino_momentum_start / --dino_momentum_end / --num_epoch /
# --collapse_early_stop の4点のみ。
#
# lr について(0017から変更なし): --ddp_linear_scale_lr は入力--lrをworld_size倍
# するだけの実装なので、論文のlinear scaling rule
# (lr = 0.0005 × global_batch/256 = 0.0005 × 1024/256 = 2e-3)を満たすには
# 入力を 2e-3 / 4 = 5e-4 にする必要がある。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note dino_diag_momentum_paper --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --batch_size 256 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 480 --warmup_t 10 --lr_min 1e-6 \
    --dino_momentum_start 0.996 --dino_momentum_end 1.0 \
    --collapse_early_stop \
    --rank_monitor_interval 5 --save_interval 5"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
