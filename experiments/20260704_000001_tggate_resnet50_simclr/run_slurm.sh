#!/bin/bash
#SBATCH --job-name=20260704_000001_tggate_resnet50_simclr
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260704_000001_tggate_resnet50_simclr/%j_slurm.out
#SBATCH --error=outputs/20260704_000001_tggate_resnet50_simclr/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Backbone: ResNet50 | SSL: SimCLR | Batch: 256 | Epoch: 100
# Note: SimCLR performance is sensitive to batch size.
# With batch=256 (vs. 4096 in original paper), NT-Xent loss has fewer negatives.
# This is acceptable for comparative benchmarking under fixed compute constraints.
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
export DEPENDENT_EXPS=""

export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260704_000001_tggate_resnet50_simclr"
ENV_FILE="${PROJECT_ROOT}/.env"
WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"

if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "Error: Could not find .env file at $ENV_FILE"; exit 1
fi

if [ "$USE_LOCAL_SSD_OUTPUT" -eq 1 ]; then
    export OUTPUT_DIR="${LOCAL_SSD_DIR}/output"
    mkdir -p "${OUTPUT_DIR}"
else
    export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"
fi

mkdir -p "${WORKSPACE_OUTPUT_DIR}"

if [ "$USE_LOCAL_SSD_INPUT" -eq 1 ]; then
    mkdir -p "${LOCAL_SSD_DIR}/data"
    rsync -a "${PROJECT_ROOT}/data/" "${LOCAL_SSD_DIR}/data/"
    export DATASET_DIR="${LOCAL_SSD_DIR}/data"
else
    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

echo "========================================="
echo "Backbone:    ResNet50"
echo "SSL:         SimCLR (NT-Xent, temp=0.5)"
echo "Batch size:  256"
echo "Epochs:      100"
echo "========================================="

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"
    export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note ssl_bench_simclr_resnet50 \
        --project_path ${PROJECT_ROOT} \
        --dir_result ${OUTPUT_DIR} \
        --model_name ResNet50 \
        --ssl_name simclr \
        --batch_size 256 \
        --num_epoch 100 \
        --patience 100 \
        --color_plob 0.8 \
        --blur_plob 0.4 \
        --solar_plob 0.0 \
        --lr 1e-3 \
        --rank_monitor_interval 5
"

if [ "$USE_LOCAL_SSD_OUTPUT" -eq 1 ]; then
    rsync -a "${OUTPUT_DIR}/" "${WORKSPACE_OUTPUT_DIR}/"
fi

echo "Cleaning up local SSD..."
rm -rf "${LOCAL_SSD_DIR:?}"/*
echo "Job finished."
