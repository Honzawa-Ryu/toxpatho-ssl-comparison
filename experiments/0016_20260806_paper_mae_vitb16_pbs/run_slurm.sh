#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 8ノード(GH200 120GB×8)マルチノードDDP。batch_size 512/GPU × 8 = 実質4096
# (論文 MAE, He et al. 2022 のデフォルトbatch 4096と一致)。
#
# このファイルは元々サイト未確定のFIXMEテンプレート(汎用PBS Pro想定、GPU1台・
# batch_size=768固定)だった。experiments/0015_20260806_paper_barlowtwins_vitb16/
# run_slurm.sh でMiyabi実機のqsub投入まで検証済みのサイト固有値(キュー/
# 課金コード/select句/apptainer/SIF_PATH/SCRATCH_ROOT)にここで合わせ、
# あわせてBSZ/LRも論文値に揃える。マルチノードDDP機構自体は
# experiments/0017_20260806_paper_dino_vitb16/ (4ノード、batch 256/GPU、
# job 2522081で100 epoch完走)で実機検証済み。
#
# batch_size 512/GPU: SLURM版(0013_20260805_paper_mae_vitb16)のvram_probe.py
# 実測(A6000 48GB、MAE ViT-B/16 + AdamW + bf16 autocast、2026-08-05 job 8283)で
#   bs=512  -> peak 28.62GB (56%) OK
#   bs=768  -> peak 42.10GB (83%) OK
#   bs=1024 -> OOM
# の3点が判明済み。768/GPU(0013と同じ)ではノード数8を綺麗な整数比で論文の
# 4096に割れない(768×8=6144)ため、8で割り切れて確実に安全マージンのある
# 512/GPUを採用(GH200 120GBはA6000 48GBの約2.5倍のVRAMがあるため、512は
# 一層余裕がある想定だがGH200上での実測はまだない)。
#
# lrについて: base_lr 1.5e-4 はbatch_size=256基準(MAE公式のlinear scaling
# rule: lr = base_lr × batch_size/256)。今回batch_size 512/GPU × 8ノード = 4096
# なので --lr = 1.5e-4 × 4096/256 = 2.4e-3 に変更(旧: 単一ノードbatch=768の
# ときは 1.5e-4×768/256=4.5e-4)。MAEの再構成lossはサンプルごと(バッチ統計に
# 依存しない)なので、Barlow Twins等と違いこの分割は数学的に単一の大きい
# batchと等価。
#
# 要点(0015/0017と共通、詳細根拠は experiments/0019_20260806_pbs_test/
# run_slurm.sh 参照):
#   - キュー名は実行キュー(small-g等)ではなく親のルーティングキュー `regular-g`。
#   - 課金/グループは `-P` ではなく `-W group_list=gd43`。
#   - select句は `select=8`(8ノード、各ノードGPU1台・ngpusは無効resource)。
#   - SCRATCH_ROOT: ノードローカルNVMe SSDの実マウント点は `/local`。
#   - apptainer: ジョブスクリプトはmodule環境を引き継がないため明示的に
#     `module load apptainer/1.3.5` が必要。
#   - SIF_PATH: このリポジトリ専用の env/env.sif は未ビルドのため、暫定で
#     共有コンテナ pytorch-ngc-26.06.sif を使う。
#   - walltime上限: `regular-g` は48時間が上限。RUN_COMMAND側に
#     `--save_interval 5 --resume` があるので、打ち切られても同じ `qsub` で
#     直近チェックポイントから再開できる想定(実際の再開動作は未検証)。
#   - ⚠️ 8ノード同時にNFSから141GBをrsyncするリスクは0015と同様未検証
#     (失敗する場合はデータロードの早い段階で落ちるはずなので、48h丸ごと
#     無駄になるような壊れ方にはなりにくい想定)。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0016_20260806_paper_mae_vitb16_pbs`
# してから `qsub experiments/0016_20260806_paper_mae_vitb16_pbs/run_slurm.sh`。
# 48h経過でジョブが打ち切られたら、同じコマンドで再度 `qsub` して再開する。
# =====================================================
#PBS -N 0016_20260806_paper_mae_vitb16_pbs
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0016_20260806_paper_mae_vitb16_pbs/
#PBS -e logs/0016_20260806_paper_mae_vitb16_pbs/

# ジョブスクリプトはmodule環境を引き継がないため明示的にロードする。
module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0016_20260806_paper_mae_vitb16_pbs"

# ノードローカルSSDの実際のマウント点(既定の/scratchはMiyabiに存在しない)。
export SCRATCH_ROOT="/local"

# 共有コンテナを暫定使用(このリポジトリ専用の env/env.sif は未ビルド)。
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage (理由はEXP15冒頭コメント参照。data/ssl_patchesのNFSランダム読みは遅い)
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
# Multi-node DDP (8 nodes) — batch_size/lrを論文値に揃える
#
# wsi-ad experiments/20260714_paper_mae_vitb16/run_slurm.sh のpaper-faithfulな
# CLI引数(MAE, He et al. 2022, ViT-B/16)を踏襲しつつ、BSZ/LRは論文の実値
# (batch_size=4096)にここで揃える。
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_mae_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name mae \
    --optimizer adamw --beta2 0.95 --lr 2.4e-3 --weight_decay 0.05 \
    --batch_size 512 \
    --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
    --num_epoch 100 --warmup_t 40 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
