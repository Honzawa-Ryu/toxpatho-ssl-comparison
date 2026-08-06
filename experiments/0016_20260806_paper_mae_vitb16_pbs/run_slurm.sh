#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — スパコン(Miyabi等のPBS Pro環境)向け。
# GPU 1台(96GB想定)・DDP無し(単一プロセス python 実行)。
#
# サイト固有の確認事項(#PBS -q/-P/-l select、apptainer利用可否、SIF_PATH/.env準備)
# は experiments/0014_20260806_paper_simsiam_vitb16/run_slurm.sh 冒頭のコメントを参照。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0016_20260806_paper_mae_vitb16_pbs`
# してから `qsub experiments/0016_20260806_paper_mae_vitb16_pbs/run_slurm.sh`。
# =====================================================
#PBS -N 0016_20260806_paper_mae_vitb16_pbs
#PBS -q <FIXME: queue name>
#PBS -P <FIXME: project/account code>
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=196:00:00
#PBS -j oe
#PBS -o logs/0016_20260806_paper_mae_vitb16_pbs/
#PBS -e logs/0016_20260806_paper_mae_vitb16_pbs/

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0016_20260806_paper_mae_vitb16_pbs"

# export SIF_PATH="${PROJECT_ROOT}/env/env.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage (理由はEXP14冒頭コメント参照。data/ssl_patchesのNFSランダム読みは遅い)
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
# Single run（デフォルト）
#
# wsi-ad experiments/20260714_paper_mae_vitb16/run_slurm.sh のpaper-faithfulな
# CLI引数(MAE, He et al. 2022, ViT-B/16)を踏襲。GPU 1台なのでDDP無し。
#
# batch_size 768: wsi-adの単一GPU実績(512)より大きい値。0013_20260805_paper_mae_vitb16
# のvram_probe.pyでA6000 48GB上で実測済み(bs=768 -> peak 42.10GB=83%, bs=1024はOOM)。
# 96GBならさらに大きく出来る可能性が高いが未検証。同スクリプトをMiyabi上でも
# 動かしてから(1GPUのみ使う設定でよい)、より大きいbatch_sizeを試すのが安全。
# batch_sizeを変える場合、lrはlinear scaling rule (base_lr 1.5e-4 @ bs256) で
# 再計算すること(今のbatch_size=768なら 1.5e-4×768/256=4.5e-4、既にこの値)。
# MAEの再構成lossはサンプルごと(バッチ統計に依存しない)なので、batch_sizeを
# 変えても損失の数学的な性質自体は変わらない(Barlow Twins等と違う点)。
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_mae_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name mae \
    --optimizer adamw --beta2 0.95 --lr 4.5e-4 --weight_decay 0.05 \
    --batch_size 768 \
    --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
    --num_epoch 100 --warmup_t 40 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
