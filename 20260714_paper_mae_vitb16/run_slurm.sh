#!/bin/bash
#SBATCH --job-name=20260714_paper_mae_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_mae_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_mae_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# =====================================================================
# MAE — paper-faithful config (He et al. 2022, ViT-B/16)
#   optimizer  : AdamW, betas (0.9, 0.95)
#   base lr     : 1.5e-4 -> linear-scaled 1.5e-4*4096/256 = 2.4e-3
#   batch       : 4096
#   weight_decay: 0.05
#   warmup      : 40 epochs, cosine decay
#   mask ratio  : 0.75 ; normalized-pixel loss ; aug = crop + flip only (no color)
# FULLY REPRESENTABLE with current args (beta2, warmup_t, aug-off all supported).
# NOTE: bs4096 will OOM on one 48GB GPU (bs256~14GB) -> revisit (grad-accum / DDP).
#   The current running ViT-B run (20260710_000001) is the near-paper reference but used
#   bs256 / beta2 0.999 / warmup 10 -> this shell is the strict version.
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_mae_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_mae_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name mae \
        --optimizer adamw --beta2 0.95 \
        --lr 2.4e-3 \
        --weight_decay 0.05 \
        --batch_size 4096 \
        --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
        --num_epoch 100 --warmup_t 40 --lr_min 0.0 \
        --rank_monitor_interval 5
"
echo "Job finished."
