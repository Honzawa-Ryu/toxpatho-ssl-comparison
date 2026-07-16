#!/bin/bash
#SBATCH --job-name=20260714_paper_dino_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_dino_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_dino_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=44G

# =====================================================================
# DINO — paper-faithful (ViT-B/16), batch=512, epoch=100, no early stop
#   optimizer  : AdamW
#   lr         : 0.0005 base -> linear-scaled x512/256 = 0.001
#   weight_decay: cosine schedule 0.04 -> 0.4
#   teacher temp / EMA / last-layer freeze : DINO class defaults
#   warmup 10ep
#   multicrop  : 2x224 + 8x96 (paper); the DINO timm backbone already uses
#     dynamic_img_size, and _forward_views groups crops by resolution.
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_dino_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline; export OMP_NUM_THREADS=1
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_dino_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name dino \
        --optimizer adamw --lr 2.5e-4 \
        --weight_decay 0.04 --weight_decay_end 0.4 \
        --batch_size 128 \
        --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
        --num_epoch 100 --warmup_t 10 --lr_min 1e-6 \
        --rank_monitor_interval 5 --save_interval 5 --resume
"
echo "Job finished."
