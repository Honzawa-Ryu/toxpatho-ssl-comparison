#!/bin/bash
#SBATCH --job-name=20260714_paper_mae_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_mae_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_mae_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =====================================================================
# MAE — paper-faithful (He et al. 2022, ViT-B/16), batch=512, epoch=100, no early stop
#   optimizer  : AdamW, betas (0.9, 0.95)
#   lr         : 1.5e-4 base -> linear-scaled x512/256 = 3e-4
#   weight_decay: 0.05 ; warmup 40ep ; mask 0.75 ; normalized-pixel loss
#   augmentation: crop + flip only (no color/blur/solarize)
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
        --optimizer adamw --beta2 0.95 --lr 3e-4 --weight_decay 0.05 \
        --batch_size 512 \
        --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
        --num_epoch 100 --warmup_t 40 --lr_min 0.0 \
        --rank_monitor_interval 5
"
echo "Job finished."
