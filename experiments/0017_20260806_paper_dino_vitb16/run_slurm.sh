#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。batch_size 256/GPU × 4 = 実質1024
# (論文のbatch 1024相当に近づけつつ、8ノードよりTOKENコストを抑える狙い。
# 256は実績値128の2倍で、GH200(120GB)はA6000(48GB)の約2.5倍のメモリが
# あるためOOMリスクは比較的低いと推測されるが未検証)。
#
# サイト固有値は experiments/0015_20260806_paper_barlowtwins_vitb16/run_slurm.sh
# (単一ノードで実機検証済み)に合わせてある。マルチノードDDP自体の起動機構
# (scripts/slurm_entry.sh の _run_multi_node: pbsdsh + torchrun c10d rendezvous)
# は experiments/0020_20260808_pbs_multinode_ddp_smoketest/ の2ノードスモーク
# テストで実機確認済み(2026-08-08、job 2507356: 2物理ノード間のNCCL
# all_reduceが正しい値を返すことを確認)。
#
# ⚠️ 未検証の残リスク:
#   - USE_LOCAL_SSD_INPUT=1 によるノードごとのデータrsync(_run_multi_node内の
#     data_staging_cmds)は、上記スモークテストではデータを一切使わないため
#     一度も実機で通していない。4ノード同時にNFSから141GBをrsyncするので
#     初回起動が重くなる可能性がある。
#   - batch_size 256/GPU がGH200 120GBのVRAMに収まるかも未検証(実績があるのは
#     128まで)。
#   いずれも失敗する場合はデータロード/最初の数stepという早い段階で落ちる
#   はずなので、48h丸ごと無駄になるような壊れ方にはなりにくい想定。
#
# 要点(0015/0019/0020と共通):
#   - キュー名は実行キュー(small-g等)ではなく親のルーティングキュー `regular-g`。
#   - 課金/グループは `-P` ではなく `-W group_list=gd43`。
#   - select句は `select=4`(4ノード、各ノードGPU1台・ngpus指定は無効resource)。
#   - `regular-g` のwalltime上限は48h。4ノード並列で従来の単一GPU比おおよそ
#     4倍速くなる想定だが未検証。`--save_interval 5 --resume` は残してあるので、
#     万一48h打ち切りになっても同じ `qsub` を再実行すれば再開できる。
#   - apptainerはジョブスクリプトがmodule環境を引き継がないため明示的にload。
#   - SIF_PATHはこのリポジトリ専用の env/env.sif が未ビルドのため、暫定で
#     共有コンテナ pytorch-ngc-26.06.sif を使う。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0017_20260806_paper_dino_vitb16`
# してから `qsub experiments/0017_20260806_paper_dino_vitb16/run_slurm.sh`。
# 48h経過でジョブが打ち切られたら、同じコマンドで再度 `qsub` して再開する。
# =====================================================
#PBS -N 0017_20260806_paper_dino_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0017_20260806_paper_dino_vitb16/
#PBS -e logs/0017_20260806_paper_dino_vitb16/

# ジョブスクリプトはmodule環境を引き継がないため明示的にロードする。
module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0017_20260806_paper_dino_vitb16"

# ノードローカルSSDの実際のマウント点(既定の/scratchはMiyabiに存在しない)。
export SCRATCH_ROOT="/local"

# 共有コンテナを暫定使用(このリポジトリ専用の env/env.sif は未ビルド)。
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

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
# Multi-node DDP (4 nodes)
#
# wsi-ad experiments/20260714_paper_dino_vitb16/run_slurm.sh のpaper-faithfulな
# CLI引数(DINO, Caron et al. 2021, ViT-B/16)を踏襲。
#
# batch_size 256/GPU: 実績値128(A6000 48GB)の2倍。GH200(120GB)なら乗る
# だろうという推測で、実機未検証(vram_probe.py等での事前実測はしていない)。
#
# lrについて: --ddp_linear_scale_lr は「入力した--lrをworld_size倍する」
# だけの実装(batch_sizeの変化は考慮しない)。論文のlinear scaling rule
# (lr = base_lr(0.0005) × global_batch/256)を満たすには、
#   目標lr = 0.0005 × 1024/256 = 2e-3
#   入力--lr = 目標lr / world_size(4) = 5e-4
# なので、batch_sizeを128→256に上げたことに合わせて --lr も
# 2.5e-4→5e-4 に変更している(2.5e-4のままだとworld_size倍しても
# 1e-3にしかならず、論文の値からずれる)。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_dino_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --batch_size 256 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 100 --warmup_t 10 --lr_min 1e-6 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
