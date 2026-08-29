#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi-G マルチノードDDP 疎通確認(最小)。
#
# 目的: batch_size=1024を8ノードDDPで回すDINO実験(0017)の前段として、
# scripts/slurm_entry.sh に新規追加した `_run_multi_node`(pbsdsh + torchrun
# のc10d rendezvousで複数PBSノードにまたがるDDPを起動する仕組み、2026-08-08
# 追加)が実機で動くかどうかだけを、2ノード・最小コードで確認する。
#
# ⚠️ 実機未検証の前提が複数ある(このジョブで初めて確認する):
#   - pbsdsh がジョブに割り当てられた全ノードでコマンドを1回ずつ起動するか
#     (PBS Proの一般的な仕様のはずだが、Miyabiでの実績なし)
#   - torchrunのc10d rendezvousが、pbsdshが継承する(または継承しない)
#     ネットワーク環境でMASTER_ADDR:PORT越しに疎通するか
#   - NCCLがMiyabi-GのInfiniBand/NVLinkファブリックを追加設定なしで
#     使えるか(疎通しない場合、NCCL_SOCKET_IFNAME等の追加設定が要る可能性)
#
# data/lib は一切使わず、experiment.py は torch.distributed のみで
# 2ノード分のall_reduceが正しい値を返すかを確認する(トークン/計算資源を
# 節約するため、実際のモデル学習は行わない)。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0020_20260808_pbs_multinode_ddp_smoketest`
# してから
#   `module load apptainer/1.3.5 && qsub experiments/0020_20260808_pbs_multinode_ddp_smoketest/run_slurm.sh`
# =====================================================
#PBS -N 0020_pbs_multinode_ddp_smoketest
#PBS -q debug-g
#PBS -W group_list=gd43
#PBS -l select=2
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o logs/0020_20260808_pbs_multinode_ddp_smoketest/
#PBS -e logs/0020_20260808_pbs_multinode_ddp_smoketest/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0020_20260808_pbs_multinode_ddp_smoketest"

export SCRATCH_ROOT="/local"
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
# トラブルシュート用。疎通しない場合、このログのNCCL初期化メッセージから
# InfiniBand/ソケットのどちらでハングしているか切り分ける材料にする。
export NCCL_DEBUG=INFO

# =====================================================
# Storage — data不使用のため両方無効のまま
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
DATA_SUBDIRS=()

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Multi-node DDP
#
# NNODES=2: 上の #PBS -l select=2 と一致させること(ずれるとscripts/
# slurm_entry.shの_run_multi_nodeが警告を出す)。torchrunへは
# scripts/slurm_entry.sh側で自動的にrendezvous引数を前置するため、
# ここでは素の "python <script>" のままでよい。
# =====================================================

NNODES=2
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH}"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
