#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — スパコン(Miyabi等のPBS Pro環境)向け。
# GPU 1台(96GB想定)・DDP無し(単一プロセス python 実行)。
#
# ⚠️ サイト固有で埋め・確認が必要な項目(このファイルではFIXMEにしてある):
#   - #PBS -q   : 実際のキュー名
#   - #PBS -P   : 課金/プロジェクト(グループ)コード
#   - #PBS -l select=... : Miyabi側のノード/GPU/CPU/メモリ選択構文
#     (select=1:ncpus=16:ngpus=1:mem=110gb は一般的なPBS Pro記法の一例。
#      Miyabi独自の書式がある場合は要調整)
#   - apptainer(SIF)がそのまま使えるか: scripts/slurm_entry.sh は
#     `apptainer exec --nv ${SIF_PATH} ...` を前提にしている。Miyabiで
#     apptainer/singularityが無い、module load方式、または別のcontainer
#     runtimeが必要な場合はscripts/slurm_entry.sh側の対応も要る
#     (今回は未対応)。env.sif(6.2GB)を転送するか、Miyabi上で
#     `apptainer build env/env.sif env/env.def` を再ビルドするかも要確認。
#   - README「セットアップ」節(SIF_PATH/.bashrc/.env)はSlurm前提の記述なので、
#     Miyabi側でも同様にSIF_PATH等を用意すること。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0014_20260806_paper_simsiam_vitb16`
# してから、リポジトリのルートで `qsub experiments/0014_20260806_paper_simsiam_vitb16/run_slurm.sh`
# を実行する(runx相当の自動化はPBS未対応、README参照)。
# =====================================================
#PBS -N 0014_20260806_paper_simsiam_vitb16
#PBS -q <FIXME: queue name>
#PBS -P <FIXME: project/account code>
#PBS -l select=1:ncpus=16:ngpus=1:mem=110gb
#PBS -l walltime=196:00:00
#PBS -j oe
#PBS -o logs/0014_20260806_paper_simsiam_vitb16/
#PBS -e logs/0014_20260806_paper_simsiam_vitb16/

# PBS_O_WORKDIR = qsub を実行したディレクトリ。SLURM版のようにmake create_exp時に
# ホスト固有の絶対パスを埋め込む方式ではなく、投入先(Miyabi)でのリポジトリの
# 実際の配置パスに追従できるようこちらを使う(リポジトリのルートでqsubする前提)。
export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0014_20260806_paper_simsiam_vitb16"

# Miyabi側でSIF_PATH/.venvの準備ができていることが前提(README「セットアップ」参照)。
# 未設定だとscripts/slurm_entry.shが分かりやすいメッセージで落ちる。
# export SIF_PATH="${PROJECT_ROOT}/env/env.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage
# andre01ノードでの実測(EXP13: 0013_20260805_paper_mae_vitb16)で、
# data/ssl_patches(141GB)をNFS越しにshuffle=Trueでランダム読みすると
# DataLoaderのプリフェッチが枯渇して1stepが数倍〜十数倍遅くなることを確認済み。
# ノードローカルSSDへ一度だけrsyncしてから読むようにしておく
# (Miyabi側のローカルディスクのマウント点が/scratchと異なる場合は
# SCRATCH_ROOT環境変数で上書きすること)。
# =====================================================

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=1
DATA_SUBDIRS=(
    "ssl_patches"
)

# =====================================================
# python path
#
# experiments/${EXP_NAME}/experiment.py ではなく、本番の学習エントリポイント
# (scripts/train/train_tggate.py、実体は lib/trainer/entry.py)を直接叩く。
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/scripts/train/train_tggate.py"

# =====================================================
# Single run（デフォルト）
#
# wsi-ad experiments/20260714_paper_simsiam_vitb16/run_slurm.sh のpaper-faithfulな
# CLI引数(SimSiam, Chen & He 2021, ViT-B/16)をそのまま踏襲。GPU 1台なのでDDPは無し
# (torchrunではなくpython単体で起動)。
#
# batch_size 256: wsi-adの単一GPU(A6000 48GB)実績値。今回は96GBなので恐らく
# もっと大きく出来るが未検証(このプロジェクトのexperiments/0013.../vram_probe.py
# と同じ手法でMiyabi上で実測してから増やすのが安全)。batch_sizeを変える場合、
# lrはlinear scaling rule (base_lr 0.05 @ bs256) で再計算すること
# (例: bs512にするなら --lr 0.1)。
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_simsiam_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name simsiam \
    --optimizer sgd --lr 0.05 --fix_pred_lr --weight_decay 1e-4 \
    --batch_size 256 \
    --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
