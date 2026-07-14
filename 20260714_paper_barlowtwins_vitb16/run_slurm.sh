#!/bin/bash
#SBATCH --job-name=20260714_paper_barlowtwins_vitb16
#SBATCH --partition=x-large-david01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_barlowtwins_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_barlowtwins_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=110G

# =====================================================================
# Barlow Twins — paper-faithful (ViT-B/16), batch=512, epoch=100, no early stop
#   optimizer  : LARS
#   lr (weights): 0.2  base -> x512/256 = 0.4
#   lr (biases) : 0.0048 base -> x512/256 = 0.0096
#   projector   : 3 x 8192 ; lambda (off-diag) : 5e-3 (class default)
#   weight_decay: 1.5e-6 ; bias & BN excluded from LARS adaptation AND weight decay
#   warmup 10ep, cosine decay
#   NOTE: BT loss is batch-coupled (cross-correlation over the batch); bs512 is a
#     faithful *small-batch* run (not gradient-accumulation-equivalent to bs2048).
# =====================================================================
export PROJECT_ROOT="/workspace/david01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_barlowtwins_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_barlowtwins_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name barlowtwins \
        --optimizer lars --lr 0.4 --lr_bias 0.0096 --lars_exclude_bias_bn \
        --weight_decay 1.5e-6 \
        --batch_size 512 \
        --proj_dim 512 --bt_lambda 5e-3 \
        --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
        --rank_monitor_interval 5
"
echo "Job finished."
