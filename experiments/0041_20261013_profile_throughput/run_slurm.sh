#!/bin/bash
# =====================================================
# exp 0041 — 学習の律速を切り分けるプロファイル(1 ノード・1 GPU・約 30〜40 分)。学習はしない。
#
# 何を測るか(scripts/train/profile_throughput.py):
#   BT 0035 / SimSiam 0038 / MAE 0040 の run_slurm.sh と同じ引数でモデル・optimizer・損失を組み、
#   1 GPU あたりのバッチを変えて
#     synth   合成データでの forward+backward+optimizer.step(データ供給を除いた GPU 側の上限)
#     fwdbwd / opt   その内訳(opt は LARS の Python ループなど optimizer.step の重さ)
#     loader  DataLoader だけの供給能力(実データ・実拡張・16 worker・ノードローカル SSD)
#     real    本番と同じ train_epoch(テレメトリ込み)の速度
#     profile 本番のバッチでの torch.profiler の内訳(CPU / CUDA 時間の上位 30)
#   peak_mem で、そのバッチが 1 GPU(GH200 120GB)に載るかも分かる。
#
# 読み方: real ≈ synth → GPU 律速(バッチを上げて GPU を埋め、ノードを減らせばノード時間が減る)。
#   real ≈ loader → CPU のデータ供給が律速。どちらより遅い → step ごとの CPU 側オーバーヘッド。
#   1 GPU なのでノード間通信は含まない。本番の 1 GPU あたりの速度(BT 0039: ≈850 画像/s、SimSiam 0038: ≈980、
#   MAE 0037: ≈3000)との差が通信のコスト。
#
# ■ バッチを変えても結果が変わらない条件(global batch は固定し、ノード数で割り直す):
#   MAE / SimSiam / DINO はサンプルごとの損失の平均 + DDP の勾配平均 + SyncBN なので、1 GPU あたりの
#   バッチ・ノード数に依らない(データの順序と乱数が変わるだけ)。BT は勾配が「真の勾配 / world_size」になるので
#   bias/LN の実効 lr がノード数で変わる → --bt_grad_ref_world 16(2026-10-13 追加)を付けて揃える。
#
# 準備と投入:
#   bash experiments/0041_20261013_profile_throughput/preflight.sh
#   qsub experiments/0041_20261013_profile_throughput/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0041_profile
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/0041_20261013_profile_throughput/
#PBS -e logs/0041_20261013_profile_throughput/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0041_20261013_profile_throughput"

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

# 先頭のバッチが本番の 1 GPU あたりのバッチ(profiler の内訳はそこだけ取る)
RUN_MODE="single"
RUN_COMMAND="python ${PROJECT_ROOT}/scripts/train/profile_throughput.py \
    --out ${PROJECT_ROOT}/outputs/${EXP_NAME} --steps 30 \
    --exp 0035_20261011_bt_vitb16_lrbias_quarter_clip1:256,512,1024 \
    --exp 0038_20261012_simsiam_vitb16_step_matched:256,512,1024 \
    --exp 0040_20261013_mae_vitb16_lr_half:512,1024,2048"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
