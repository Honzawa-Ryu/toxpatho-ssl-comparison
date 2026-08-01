#!/bin/bash
#SBATCH --job-name=0006_20260729_paper_visualize
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/filesrv02/honzawa/wsi-ad/logs/0006_20260729_paper_visualize/%j_0006_20260729_paper_visualize.out
#SBATCH --error=/workspace/filesrv02/honzawa/wsi-ad/logs/0006_20260729_paper_visualize/%j_0006_20260729_paper_visualize.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Paper analysis phase 4/4: loss / eff_rank / feat_std / t-SNE by slide。
#
# 上流フェーズのジョブに依存させたい場合は、下の行を有効化して job_id を埋める
# （job_id は outputs/{上流exp}/latest_job_id.txt を参照）:
# #SBATCH --dependency=afterok:<job_id>   # 0003_20260729_paper_extract_embeddings

export PROJECT_ROOT="/workspace/filesrv02/honzawa/wsi-ad"
export EXP_NAME="0006_20260729_paper_visualize"

# 解析は行列演算が主で、学習ほど DataLoader worker を並べない。
# 移行前の run_analysis*_slurm.sh と同じ 4 を維持する。
export OMP_NUM_THREADS=4


# =====================================================
# Storage
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1


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
