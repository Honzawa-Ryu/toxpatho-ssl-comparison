#!/bin/bash
#SBATCH --job-name=uv_sync
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --output=logs/uv-sync-log_%j.out
#SBATCH --error=logs/uv-sync-log_%j.out

# Resolve absolute path of the project root
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

echo "=========================================="
echo "Starting uv sync on compute node..."
echo "Project Root: ${PROJECT_ROOT}"
echo "=========================================="

# Execute uv sync inside the Apptainer container
# Note: 'cd' into PROJECT_ROOT inside the container to ensure .venv is created in the right place
apptainer exec --nv "${SIF_PATH}" bash -c "
    cd ${PROJECT_ROOT}
    uv sync
"

echo "=========================================="
echo "uv sync completed successfully."
echo "=========================================="
