#!/bin/bash
#SBATCH --job-name=20260612_181246_tggate_resnet18_barlowtwins
#SBATCH --partition=x-large-andre01
#SBATCH --time=48:00:00
#SBATCH --output=outputs/20260612_181246_tggate_resnet18_barlowtwins/%j_slurm.out
#SBATCH --error=outputs/20260612_181246_tggate_resnet18_barlowtwins/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8       
#SBATCH --mem=32G

# ==========================================
# Experiment Configuration
# ==========================================
# 0: Use workspace input / output
# 1: Use local SSD for input staging / output (with final sync back to workspace)
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
# dependent experiments outputs
# ex: export DEPENDENT_EXPS="20260327_101530_pretrain, 20260328_120000_stage2"
export DEPENDENT_EXPS=""
# ==========================================

# 1. Absolute paths injected dynamically at creation time
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260612_181246_tggate_resnet18_barlowtwins"
ENV_FILE="${PROJECT_ROOT}/.env"
WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/20260612_181246_tggate_resnet18_barlowtwins"

# 2. Load environment variables securely
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from: $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: Could not find .env file at $ENV_FILE"
    exit 1
fi

# 3. Output Directory Setup based on the flag
if [ "$USE_LOCAL_SSD_OUTPUT" -eq 1 ]; then
    echo "Output mode: Local SSD -> workspace"
    export OUTPUT_DIR="${LOCAL_SSD_DIR}/output"
    mkdir -p "${OUTPUT_DIR}"
else
    echo "Output mode: Direct to workspace"
    export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"
fi

# 4. Data Staging: Extract data from workspace directly to compute node's fast NVMe SSD
# TODO: check the data path (/data/*...), there are no need to copy the whole data directory if only a subset is needed for the experiment
if [ "$USE_LOCAL_SSD_INPUT" -eq 1 ]; then
    echo "Input mode: Staging to ${LOCAL_SSD_DIR}"
    mkdir -p "${LOCAL_SSD_DIR}/data"
    echo "Staging dataset to ${LOCAL_SSD_DIR}..."
    rsync -a "${PROJECT_ROOT}/data/" "${LOCAL_SSD_DIR}/data/"
    export DATASET_DIR="${LOCAL_SSD_DIR}/data"
else
    echo "Input mode: Direct from workspace"
    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

# 5. Execute experiment inside Apptainer
echo "start experiment..."
echo "========================================="
echo "Model Name: ResNet18"
echo "SSL Name: barlowtwins"
echo "Batch Size: 64"
echo "Epochs: 2"
echo "========================================="
apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --project_path ${PROJECT_ROOT} \
        --dir_result ${OUTPUT_DIR} \
        --patience 100 \
        --model_name ResNet18 \
        --ssl_name barlowtwins \
        --batch_size 256 \
        --num_epoch 2
"

# 6. Sync back outputs to workspace (if local SSD was used)
if [ "$USE_LOCAL_SSD_OUTPUT" -eq 1 ]; then
    echo "Syncing experiment outputs back to workspace..."
    rsync -a "${OUTPUT_DIR}/" "${WORKSPACE_OUTPUT_DIR}/"
fi

# 7. Cleanup
echo "Cleaning up local SSD..."
rm -rf "${LOCAL_SSD_DIR:?}"/*
echo "Job finished."