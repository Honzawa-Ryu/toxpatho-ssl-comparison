#!/bin/bash
#SBATCH --job-name=20260714_paper_swav_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --time=72:00:00
#SBATCH --output=outputs/20260714_paper_swav_vitb16/%j_slurm.out
#SBATCH --error=outputs/20260714_paper_swav_vitb16/%j_slurm.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

# =====================================================================
# SwAV — paper-faithful (ViT-B/16), batch=512, epoch=100, no early stop
#   optimizer : SGD (momentum 0.9)
#   lr        : 0.3 base -> linear-scaled x512/256 = 0.6
#   weight_decay: 1e-6 ; prototypes: 3000 ; warmup 10ep
#   multicrop  : 2x224 + 6x96 (paper); uses a timm ViT-B/16 (dynamic_img_size)
#     backbone so 96px local crops interpolate the pos-embed.
#   CAVEAT: small batch (512) with 3000 prototypes & no queue may be unstable
#     (paper uses a feature queue for small batch). Watch eff_rank.
# =====================================================================
export PROJECT_ROOT="/workspace/andre01/honzawa/wsi-ad"
export EXP_NAME="20260714_paper_swav_vitb16"
ENV_FILE="${PROJECT_ROOT}/.env"; WORKSPACE_OUTPUT_DIR="${PROJECT_ROOT}/outputs/${EXP_NAME}"
if [ -f "$ENV_FILE" ]; then set -a; source "$ENV_FILE"; set +a; else echo "no .env"; exit 1; fi
export OUTPUT_DIR="${WORKSPACE_OUTPUT_DIR}"; mkdir -p "${WORKSPACE_OUTPUT_DIR}"
export DATASET_DIR="${PROJECT_ROOT}/data"

apptainer exec --nv --bind "${LOCAL_SSD_DIR}" "${PROJECT_ROOT}/env/env.sif" bash -c "
    source ${PROJECT_ROOT}/.venv/bin/activate
    export PYTHONPATH=\"${PROJECT_ROOT}:\${PYTHONPATH:-}\"; export WANDB_MODE=offline; export OMP_NUM_THREADS=1
    cd ${PROJECT_ROOT}
    python ${PROJECT_ROOT}/scripts/train/train_tggate.py \
        --note paper_swav_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${OUTPUT_DIR} \
        --model_name ViTB16 --ssl_name swav \
        --optimizer sgd --lr 0.1875 --weight_decay 1e-6 \
        --batch_size 160 \
        --n_prototypes 3000 \
        --n_global_crops 2 --n_local_crops 6 --local_crop_size 96 \
        --num_epoch 100 --warmup_t 10 --lr_min 6e-4 \
        --rank_monitor_interval 5 --save_interval 5 --resume
"
echo "Job finished."
