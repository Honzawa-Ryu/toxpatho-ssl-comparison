#!/bin/bash
# Usage:
#
# Resume latest experiment:
#   bash tools/resume_exp.sh
#
# Resume experiment 0004:
#   bash tools/resume_exp.sh 4
#
# Resume with custom suffix:
#   bash tools/resume_exp.sh 4 retry
#
# Resume by explicit experiment name:
#   bash tools/resume_exp.sh 0042_20260509_baseline
#

set -euo pipefail

INPUT=${1:-}
SUFFIX=${2:-resumed}

# =========================================================
# Resolve project root
# =========================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

source "${PROJECT_ROOT}/scripts/exp_common.sh"

EXP_ROOT="${PROJECT_ROOT}/experiments"
OUT_ROOT="${PROJECT_ROOT}/outputs"

# =========================================================
# Resolve source experiment
# =========================================================

SRC_EXP_DIR=$(exp_resolve_dir "${EXP_ROOT}" "${INPUT}")

if [ -z "${SRC_EXP_DIR}" ]; then
    echo "❌ No experiments found."
    exit 1
fi

SRC_EXP=$(basename "${SRC_EXP_DIR}")

SRC_OUT_DIR="${OUT_ROOT}/${SRC_EXP}"

# =========================================================
# Validation
# =========================================================

if [ ! -d "${SRC_EXP_DIR}" ]; then
    echo "❌ Experiment directory not found:"
    echo "   ${SRC_EXP_DIR}"
    exit 1
fi

# =========================================================
# Generate new experiment ID
# =========================================================

NEW_ID=$(exp_next_id "${EXP_ROOT}")

# =========================================================
# Parse source experiment name
# =========================================================

BASE_NAME=$(echo "${SRC_EXP}" | sed -E 's/^[0-9]{4}_[0-9]{8}_//')
BASE_NAME=$(echo "${BASE_NAME}" | sed -E 's/_FAILED_.*$//')
BASE_NAME=$(echo "${BASE_NAME}" | sed -E 's/_(resumed|retry|resume)$//')

DATE=$(date +"%Y%m%d")

SAFE_SUFFIX=$(
    echo "${SUFFIX}" \
    | tr ' /' '__' \
    | tr -cd '[:alnum:]_-'
)

NEW_EXP_NAME="${NEW_ID}_${DATE}_${BASE_NAME}_${SAFE_SUFFIX}"

NEW_EXP_DIR="${EXP_ROOT}/${NEW_EXP_NAME}"
NEW_OUT_DIR="${OUT_ROOT}/${NEW_EXP_NAME}"

# =========================================================
# Safety check
# =========================================================

if [ -e "${NEW_EXP_DIR}" ]; then
    echo "❌ Resume target already exists:"
    echo "   ${NEW_EXP_NAME}"
    exit 1
fi

# =========================================================
# Create directories
# =========================================================

mkdir -p "${NEW_EXP_DIR}"
mkdir -p "${NEW_OUT_DIR}"

# =========================================================
# Copy experiment files
# =========================================================

rsync -a \
    --exclude="FAILED_REASON.txt" \
    --exclude="uncommitted_changes.diff" \
    --exclude="metadata.yaml" \
    --exclude="__pycache__/" \
    "${SRC_EXP_DIR}/" "${NEW_EXP_DIR}/"

# =========================================================
# Rewrite experiment references
# =========================================================

RUN_SCRIPT="${NEW_EXP_DIR}/run_slurm.sh"

if [ -f "${RUN_SCRIPT}" ]; then
    sed -i "s|${SRC_EXP}|${NEW_EXP_NAME}|g" "${RUN_SCRIPT}"

    # 元の実験で #SBATCH --dependency= が有効化されていた場合、古いjob_idへの
    # 依存を resume 先へ無自覚に引き継がないよう、再度コメントアウトし直す
    # （必要ならユーザーが新しいjob_idで手動で有効化し直す）。
    if grep -qE '^#SBATCH[[:space:]]+--dependency=' "${RUN_SCRIPT}"; then
        sed -i -E 's/^#SBATCH([[:space:]]+--dependency=.*)$/# #SBATCH\1/' "${RUN_SCRIPT}"
        echo "⚠️  元の実験の #SBATCH --dependency= を無効化しました（古いjob_idを引き継がないため）。"
        echo "    必要なら新しいjob_idで手動で有効化し直してください。"
    fi
fi

CONFIG_PATH="${NEW_EXP_DIR}/config.yml"

if [ -f "${CONFIG_PATH}" ]; then
    sed -i "s|${SRC_EXP}|${NEW_EXP_NAME}|g" "${CONFIG_PATH}"
fi

# =========================================================
# Resume metadata
# =========================================================

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "git_not_available")

cat > "${NEW_EXP_DIR}/metadata.yaml" <<EOF
exp_id: ${NEW_ID}
exp_name: ${NEW_EXP_NAME}
created_date: ${DATE}
git_commit: ${COMMIT_HASH}
resumed_from_exp: ${SRC_EXP}
resume_checkpoint_dir: ${SRC_OUT_DIR}
resume_checkpoint_file: null

EOF

# =========================================================
# Update latest symlink
# =========================================================

ln -sfn "${NEW_EXP_DIR}" "${EXP_ROOT}/latest"
ln -sfn "${NEW_OUT_DIR}" "${OUT_ROOT}/latest"

# =========================================================
# Done
# =========================================================

echo ""
echo "✅ Resumed experiment created"
echo ""
echo "Source    : ${SRC_EXP}"
echo "New       : ${NEW_EXP_NAME}"
echo "Experiment: ${NEW_EXP_DIR}"
echo "Outputs   : ${NEW_OUT_DIR}"
echo ""