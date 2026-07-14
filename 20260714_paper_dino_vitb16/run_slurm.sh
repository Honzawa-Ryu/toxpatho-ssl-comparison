#!/bin/bash
#SBATCH --job-name=20260714_paper_dino_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_dino_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_dino_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# =====================================================================
# DINO — paper-faithful config (ViT-B/16, patch16)
#   optimizer  : AdamW
#   base lr     : 0.0005 -> linear-scaled 0.0005*1024/256 = 0.002
#   batch       : 1024
#   weight_decay: cosine schedule 0.04 -> 0.4
#   multicrop   : 2x224 + 8x96   (p=16)
#   warmup 10ep (lr); teacher temp 0.04->0.07 warmup; last-layer freeze 1ep (in DINO class)
# NEW ARGS REQUIRED: --weight_decay_end (wd cosine end), --n_local_crops, --local_crop_size
#   plus wd-cosine-schedule in the train loop (wd currently fixed) and n_local_crops CLI wiring
#   into sslutils.DINO (default 0 -> 8 enables 2x224+8x96 multicrop).
# NOTE: bs1024 will OOM on one 48GB GPU -> revisit (grad-accum / DDP).
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_dino_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_dino_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name dino \
        --optimizer adamw \
        --lr 0.002 \
        --weight_decay 0.04 --weight_decay_end 0.4 \
        --batch_size 1024 \
        --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
        --num_epoch 100 --warmup_t 10 --lr_min 1e-6 \
        --rank_monitor_interval 5
"
echo "Job finished."
