#!/bin/bash
#SBATCH --job-name=20260714_paper_simsiam_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_simsiam_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_simsiam_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# =====================================================================
# SimSiam — paper-faithful (Chen & He 2021, ViT-B/16), batch=512, epoch=100, no early stop
#   optimizer  : SGD, momentum 0.9
#   lr         : 0.05 base -> linear-scaled x512/256 = 0.1
#   weight_decay: 1e-4
#   predictor  : FIXED lr (not decayed by the scheduler)  <- key anti-collapse knob
#   warmup 10ep, cosine decay ; projector 3-layer (BN), pred_dim 512, dim 2048
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_simsiam_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_simsiam_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name simsiam \
        --optimizer sgd --lr 0.1 --fix_pred_lr --weight_decay 1e-4 \
        --batch_size 512 \
        --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
        --rank_monitor_interval 5
"
echo "Job finished."
