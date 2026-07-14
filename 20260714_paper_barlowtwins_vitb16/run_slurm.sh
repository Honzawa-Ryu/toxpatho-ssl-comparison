#!/bin/bash
#SBATCH --job-name=20260714_paper_barlowtwins_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_barlowtwins_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_barlowtwins_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1

#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

# =====================================================================
# Barlow Twins — paper-faithful config (backbone unified to ViT-B/16)
#   optimizer  : LARS
#   lr (weights): 0.2   -> linear-scaled 0.2*2048/256   = 1.6
#   lr (biases) : 0.0048-> linear-scaled 0.0048*2048/256 = 0.0384
#   batch       : 2048
#   projector   : 3 x 8192  (BN on first two layers)
#   lambda (off-diag): 5e-3
#   weight_decay: 1.5e-6
#   biases & BN params: excluded from LARS adaptation AND from weight decay
#   warmup 10ep, cosine decay
# NEW ARGS REQUIRED: --optimizer lars, --lr_bias, --proj_dim 8192, --bt_lambda,
#   --lars_exclude_bias_bn ; plus LARS optimizer + param-group builder in train_tggate.py
#   and projection_dim/pred_dim=8192 wiring in BarlowTwins.prepare_model.
# NOTE: bs2048 will OOM on one 48GB GPU -> revisit (grad-accum / DDP).
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
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
        --optimizer lars \
        --lr 1.6 --lr_bias 0.0384 \
        --weight_decay 1.5e-6 --lars_exclude_bias_bn \
        --batch_size 2048 \
        --proj_dim 8192 --bt_lambda 5e-3 \
        --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
        --rank_monitor_interval 5
"
echo "Job finished."
