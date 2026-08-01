#!/bin/bash
#SBATCH --job-name=20260513_133414_tggate_learning
#SBATCH --partition=x-large-andre01
#SBATCH --time=48:00:00
#SBATCH --output=outputs/20260513_133414_tggate_learning/%j_slurm.out
#SBATCH --error=outputs/20260513_133414_tggate_learning/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# ==========================================
# Paths
# ==========================================
SUBMIT_DIR="/workspace/andre01/honzawa/wsi-ad"
PROJECT_NAME="wsi-ad"
PROJECT_DIR="/scratch/honzawa/${PROJECT_NAME}_${SLURM_JOB_ID}"
EXP_NAME="20260513_133414_tggate_learning"
WORKSPACE_OUTPUT_DIR="${SUBMIT_DIR}/outputs/${EXP_NAME}"
ENV_FILE="${SUBMIT_DIR}/.env"

# ==========================================
# Cleanup: 結果をworkspaceに戻す
# ==========================================
cleanup() {
    echo "Syncing results back to workspace..."
    mkdir -p "${WORKSPACE_OUTPUT_DIR}"
    rsync -a "${PROJECT_DIR}/outputs/${EXP_NAME}/" "${WORKSPACE_OUTPUT_DIR}/"
    echo "Cleaning up scratch..."
    rm -rf "${PROJECT_DIR}"
    echo "Done."
}
trap cleanup EXIT INT TERM

# ==========================================
# 1. scratchにプロジェクトを展開
# ==========================================
echo "Setting up project at ${PROJECT_DIR}..."
mkdir -p "${PROJECT_DIR}"

# コード類を転送
rsync -a \
    "${SUBMIT_DIR}/scripts" \
    "${SUBMIT_DIR}/src" \
    "${SUBMIT_DIR}/env" \
    "${PROJECT_DIR}/"

# データをscratchに転送 (ここが今回の肝)
echo "Staging data to scratch..."
mkdir -p "${PROJECT_DIR}/data"
rsync -a "${SUBMIT_DIR}/data/shards/" "${PROJECT_DIR}/data/shards/"
echo "Data staging complete."

# 出力先をscratch内に作成
mkdir -p "${PROJECT_DIR}/outputs/${EXP_NAME}"

# ==========================================
# 2. 学習実行 (scratch上で完結)
# ==========================================
echo "Loading env..."
set -a; source "${ENV_FILE}"; set +a

echo "Starting training..."
apptainer exec --nv \
    --bind "${PROJECT_DIR}" \
    "${SUBMIT_DIR}/env/env.sif" bash -c "
        cd ${PROJECT_DIR} && \
        source ${SUBMIT_DIR}/.venv/bin/activate && \
        python ${PROJECT_DIR}/scripts/train/train_tggate.py \
            --project_path ${PROJECT_DIR} \
            --dir_result outputs/${EXP_NAME} \
            --patience 100 \
            --model_name ViTB16 \
            --ssl_name byol \
            --batch_size 32
    "