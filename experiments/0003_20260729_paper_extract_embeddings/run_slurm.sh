#!/bin/bash
#SBATCH --job-name=0003_20260729_paper_extract_embeddings
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/filesrv02/honzawa/wsi-ad/logs/0003_20260729_paper_extract_embeddings/%j_0003_20260729_paper_extract_embeddings.out
#SBATCH --error=/workspace/filesrv02/honzawa/wsi-ad/logs/0003_20260729_paper_extract_embeddings/%j_0003_20260729_paper_extract_embeddings.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Paper analysis phase 1/4: 20260714_paper_* の model_epN.pt を埋め込む。
#
# 上流フェーズのジョブに依存させたい場合は、下の行を有効化して job_id を埋める
# （job_id は outputs/{上流exp}/latest_job_id.txt を参照）:
# #SBATCH --dependency=afterok:<job_id>

export PROJECT_ROOT="/workspace/filesrv02/honzawa/wsi-ad"
export EXP_NAME="0003_20260729_paper_extract_embeddings"

# 解析は行列演算が主で、学習ほど DataLoader worker を並べない。
# 移行前の run_analysis*_slurm.sh と同じ 4 を維持する。
export OMP_NUM_THREADS=4

# foundation model (UNI) は HF キャッシュから読む。計算ノードは
# 外に出られないのでオフラインで解決させる。
export HF_HUB_OFFLINE=1

# =====================================================
# Storage
# =====================================================

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=1

# data/ のうち解析に要るのは shards/ だけ（REFACTOR_PLAN.md §7-2）
export LOCAL_SSD_INPUT_PATHS="shards"

# =====================================================
# Run
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH}"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
