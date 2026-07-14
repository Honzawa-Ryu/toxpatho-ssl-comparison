#!/bin/bash
#SBATCH --job-name=20260714_paper_swav_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_swav_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_swav_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# =====================================================================
# SwAV — paper-faithful config (backbone unified to ViT-B/16)
#   optimizer : SGD (momentum 0.9)
#   base lr    : 0.3  -> linear-scaled: 0.3 * 2048/256 = 2.4
#   batch      : 2048
#   prototypes : 3000  (avoids intractable Sinkhorn)
#   multicrop  : 2x224 + 6x96
#   weight_decay: 1e-6
#   warmup 10ep, cosine to lr_min; freeze prototypes first epoch
# NEW ARGS REQUIRED (not yet in train_tggate.py): --n_prototypes, --n_local_crops,
#   --local_crop_size, --n_global_crops (multicrop is currently disabled: split+multi=False)
# NOTE: bs2048 will OOM on a single 48GB GPU -> revisit (grad-accum / DDP) per plan.
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_swav_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_swav_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name swav \
        --optimizer sgd \
        --lr 2.4 \
        --weight_decay 1e-6 \
        --batch_size 2048 \
        --n_prototypes 3000 \
        --n_global_crops 2 --n_local_crops 6 --local_crop_size 96 \
        --num_epoch 100 --warmup_t 10 --lr_min 6e-4 \
        --rank_monitor_interval 5
"
echo "Job finished."
