#!/bin/bash
# Usage:
#
# Rename latest experiment:
#   bash tools/rename_exp.sh new_name
#
# Rename experiment 0004:
#   bash tools/rename_exp.sh 4 better_baseline
#
# Rename by explicit experiment name:
#   bash tools/rename_exp.sh 0042_20260509_baseline roberta_large
#

set -euo pipefail

INPUT=${1:-}
NEW_LABEL=${2:-}

if [ -z "$INPUT" ] || [ -z "$NEW_LABEL" ]; then
    echo "❌ Usage: bash tools/rename_exp.sh <exp_id_or_name> <new_name>"
    exit 1
fi

# =========================================================
# Resolve project root
# =========================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

EXP_ROOT="${PROJECT_ROOT}/experiments"
OUT_ROOT="${PROJECT_ROOT}/outputs"

# =========================================================
# Helper: resolve experiment dir from ID or name
# =========================================================

_resolve_exp() {
    local input="$1"

    if [[ "$input" =~ ^[0-9]+$ ]]; then
        local padded
        padded=$(printf "%04d" "$input")

        find "${EXP_ROOT}" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name "${padded}_*" \
        | head -n1
    else
        echo "${EXP_ROOT}/${input}"
    fi
}

# =========================================================
# Resolve experiment
# =========================================================

EXP_DIR=$(_resolve_exp "$INPUT")
EXP_NAME=$(basename "${EXP_DIR}")
OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

# =========================================================
# Validation
# =========================================================

if [ ! -d "${EXP_DIR}" ]; then
    echo "❌ Experiment directory not found:"
    echo "   ${EXP_DIR}"
    exit 1
fi

# =========================================================
# Parse experiment name
# =========================================================

PARTS=($(echo "${EXP_NAME}" | tr '_' ' '))

ID="${PARTS[0]}"

HAS_DATE=0

if [[ "${PARTS[1]:-}" =~ ^[0-9]{8}$ ]]; then
    HAS_DATE=1
    DATE="${PARTS[1]}"
    PREFIX="${ID}_${DATE}"
else
    PREFIX="${ID}"
fi

# =========================================================
# Preserve FAILED suffix
# =========================================================

if [ "${HAS_DATE}" -eq 1 ]; then
    MAIN_BODY=$(echo "${EXP_NAME}" | sed -E 's/^[0-9]{4}_[0-9]{8}_//')
else
    MAIN_BODY=$(echo "${EXP_NAME}" | sed -E 's/^[0-9]{4}_//')
fi

FAILED_SUFFIX=""

if [[ "${MAIN_BODY}" == *_FAILED_* ]]; then
    FAILED_SUFFIX=$(echo "${MAIN_BODY}" | sed -E 's/^.*(_FAILED_.*)$/\1/')
fi

# =========================================================
# Safe new label
# =========================================================

SAFE_LABEL=$(
    echo "${NEW_LABEL}" \
    | tr ' /' '__' \
    | tr -cd '[:alnum:]_-'
)

NEW_EXP_NAME="${PREFIX}_${SAFE_LABEL}${FAILED_SUFFIX}"

NEW_EXP_DIR="${EXP_ROOT}/${NEW_EXP_NAME}"
NEW_OUT_DIR="${OUT_ROOT}/${NEW_EXP_NAME}"

# =========================================================
# Conflict check
# =========================================================

if [ -e "${NEW_EXP_DIR}" ]; then
    echo "❌ Target experiment already exists:"
    echo "   ${NEW_EXP_NAME}"
    exit 1
fi

# =========================================================
# Rename directories
# =========================================================

mv "${EXP_DIR}" "${NEW_EXP_DIR}"

if [ -d "${OUT_DIR}" ]; then
    mv "${OUT_DIR}" "${NEW_OUT_DIR}"
fi

# =========================================================
# Rewrite internal references
# =========================================================

RUN_SCRIPT="${NEW_EXP_DIR}/run_slurm.sh"

if [ -f "${RUN_SCRIPT}" ]; then
    sed -i "s|${EXP_NAME}|${NEW_EXP_NAME}|g" "${RUN_SCRIPT}"
fi

CONFIG_PATH="${NEW_EXP_DIR}/config.yml"

if [ -f "${CONFIG_PATH}" ]; then
    sed -i "s|${EXP_NAME}|${NEW_EXP_NAME}|g" "${CONFIG_PATH}"
fi

# =========================================================
# Update latest symlink if needed
# =========================================================

LATEST_LINK="${EXP_ROOT}/latest"

if [ -L "${LATEST_LINK}" ]; then
    CURRENT_TARGET=$(readlink -f "${LATEST_LINK}")

    if [ "${CURRENT_TARGET}" = "${EXP_DIR}" ]; then
        ln -sfn "${NEW_EXP_DIR}" "${LATEST_LINK}"
    fi
fi

LATEST_OUT_LINK="${OUT_ROOT}/latest"

if [ -L "${LATEST_OUT_LINK}" ]; then
    CURRENT_TARGET=$(readlink -f "${LATEST_OUT_LINK}")

    if [ "${CURRENT_TARGET}" = "${OUT_DIR}" ]; then
        ln -sfn "${NEW_OUT_DIR}" "${LATEST_OUT_LINK}"
    fi
fi

# =========================================================
# Done
# =========================================================

echo ""
echo "✅ Renamed experiment"
echo ""
echo "Old: ${EXP_NAME}"
echo "New: ${NEW_EXP_NAME}"
echo ""