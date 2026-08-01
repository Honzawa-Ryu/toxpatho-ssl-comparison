#!/bin/bash
#SBATCH --job-name=20260630_000001_tggate_resnet18_bt_aug_std
#SBATCH --partition=x-large-andre01
#SBATCH --time=48:00:00
#SBATCH --output=outputs/20260630_000001_tggate_resnet18_bt_aug_std/%j_slurm.out
#SBATCH --error=outputs/20260630_000001_tggate_resnet18_bt_aug_std/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Backbone: ResNet18 | SSL: BarlowTwins | Aug: standard | Epoch: 100
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
export DEPENDENT_EXPS=""

export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260630_000001_tggate_resnet18_bt_aug_std"
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

if [ "$USE_LOCAL_SSD_INPUT" -eq 1 ]; then
    mkdir -p "${LOCAL_SSD_DIR}/data"
    rsync -a "${PROJECT_ROOT}/data/" "${LOCAL_SSD_DIR}/data/"
    export DATASET_DIR="${LOCAL_SSD_DIR}/data"
else
    export DATASET_DIR="${PROJECT_ROOT}/data"
fi

echo "========================================="
echo "Backbone:    ResNet18"
echo "SSL:         BarlowTwins"
echo "Aug:         standard (color=0.8, blur=0.4, solar=0.0)"
echo "Batch size:  256"
echo "Epochs:      100"
echo "========================================="

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"
    export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note bt_resnet18_aug_std \
        --project_path ${PROJECT_ROOT} \
        --dir_result ${OUTPUT_DIR} \
        --model_name ResNet18 \
        --ssl_name barlowtwins \
        --batch_size 256 \
        --num_epoch 100 \
        --patience 100 \
        --color_plob 0.8 \
        --blur_plob 0.4 \
        --solar_plob 0.0 \
        --rank_monitor_interval 5
"

if [ "$USE_LOCAL_SSD_OUTPUT" -eq 1 ]; then
    rsync -a "${OUTPUT_DIR}/" "${WORKSPACE_OUTPUT_DIR}/"
fi

echo "Cleaning up local SSD..."
rm -rf "${LOCAL_SSD_DIR:?}"/*
echo "Job finished."
