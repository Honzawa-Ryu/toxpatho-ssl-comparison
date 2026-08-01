#!/bin/bash
#SBATCH --job-name=20260713_000001_tggate_vitl16_mae
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260713_000001_tggate_vitl16_mae/%j_slurm.out
#SBATCH --error=outputs/20260713_000001_tggate_vitl16_mae/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# Backbone: ViT-L/16 | SSL: MAE (non-contrastive, generative) | Batch: 256
# Backbone-scaling probe: identical recipe to 20260710_000001 (ViT-B MAE) with the
# encoder swapped to ViT-L/16 (--model_name ViTL16 -> mae_vit_large_patch16, 329.5M).
# Goal: isolate how much of the UNI (ViT-L, Mass-100K) vs in-house gap is due to
# backbone size alone. Everything else (lr, warmup, wd, aug-off, mask 0.75, bs 256,
# 100ep, patience 15) is kept identical for a fair single-variable comparison.
# NOTE: ViT-L is ~2-2.5x slower/epoch than ViT-B, so 100ep may exceed the 72h wall;
# state.pt is checkpointed every epoch -> resume via tools/resume_exp.sh if needed.
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0
export DEPENDENT_EXPS=""

export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260713_000001_tggate_vitl16_mae"
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
echo "Backbone:    ViT-L/16 (329.5M)"
echo "SSL:         MAE (mask ratio 0.75)"
echo "Batch size:  256"
echo "Epochs:      100 (early stopping, patience 15)"
echo "========================================="

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"
    export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note ssl_bench_mae_vitl16 \
        --project_path ${PROJECT_ROOT} \
        --dir_result ${OUTPUT_DIR} \
        --model_name ViTL16 \
        --ssl_name mae \
        --batch_size 256 \
        --num_epoch 100 \
        --patience 15 \
        --color_plob 0.0 \
        --blur_plob 0.0 \
        --solar_plob 0.0 \
        --lr 1.5e-4 \
        --weight_decay 0.05 \
        --optimizer adamw \
        --warmup_t 10 \
        --rank_monitor_interval 5
"

if [ "$USE_LOCAL_SSD_OUTPUT" -eq 1 ]; then
    rsync -a "${OUTPUT_DIR}/" "${WORKSPACE_OUTPUT_DIR}/"
fi

echo "Cleaning up local SSD..."
rm -rf "${LOCAL_SSD_DIR:?}"/*
echo "Job finished."
