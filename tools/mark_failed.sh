#!/bin/bash
# Usage:
#
# Mark latest experiment as failed:
#   bash tools/mark_failed.sh
#
# Mark experiment 0004:
#   bash tools/mark_failed.sh 4 oom
#
# Mark by explicit experiment name:
#   bash tools/mark_failed.sh 0042_20260509_baseline timeout
#

set -euo pipefail

INPUT=${1:-}
REASON=${2:-}

if [ -z "$REASON" ]; then
    echo "❌ Usage: bash tools/mark_failed.sh <exp_id_or_name> <reason>"
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

_latest_exp() {
    find "${EXP_ROOT}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        ! -name latest \
        -printf "%f\n" 2>/dev/null \
    | grep -E '^[0-9]{4}_' \
    | sort -t '_' -k1,1n \
    | tail -n1
}

# =========================================================
# Resolve experiment
# =========================================================

if [ -z "$INPUT" ]; then
    EXP_NAME=$(_latest_exp)

    if [ -z "${EXP_NAME:-}" ]; then
        echo "❌ No experiments found."
        exit 1
    fi

    EXP_DIR="${EXP_ROOT}/${EXP_NAME}"
else
    EXP_DIR=$(_resolve_exp "$INPUT")
    EXP_NAME=$(basename "${EXP_DIR}")
fi

OUT_DIR="${OUT_ROOT}/${EXP_NAME}"

# =========================================================
# Validation
# =========================================================

if [ ! -d "${EXP_DIR}" ]; then
    echo "❌ Experiment directory not found:"
    echo "   ${EXP_DIR}"
    exit 1
fi

if [[ "${EXP_NAME}" == *_FAILED_* ]]; then
    echo "⚠️  Already marked as FAILED:"
    echo "   ${EXP_NAME}"
    exit 0
fi

# =========================================================
# Safe reason string
# =========================================================

SAFE_REASON=$(
    echo "${REASON}" \
    | tr ' /' '__' \
    | tr -cd '[:alnum:]_-'
)

DATE=$(date +"%Y%m%d")

NEW_EXP_NAME="${EXP_NAME}_FAILED_${DATE}_${SAFE_REASON}"

NEW_EXP_DIR="${EXP_ROOT}/${NEW_EXP_NAME}"
NEW_OUT_DIR="${OUT_ROOT}/${NEW_EXP_NAME}"

# =========================================================
# Safety check
# =========================================================

if [ -e "${NEW_EXP_DIR}" ]; then
    echo "❌ FAILED experiment already exists:"
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
# FAILED metadata
# =========================================================

cat > "${NEW_EXP_DIR}/FAILED_REASON.txt" <<EOF
failed_at: $(date +"%Y-%m-%d %H:%M:%S")
reason: ${REASON}
original_experiment: ${EXP_NAME}
EOF

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
echo "❌ Marked experiment as FAILED"
echo ""
echo "Original : ${EXP_NAME}"
echo "Renamed  : ${NEW_EXP_NAME}"
echo "Reason   : ${REASON}"
echo ""