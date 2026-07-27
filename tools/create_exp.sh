#!/bin/bash
# Usage:
#   bash tools/create_exp.sh <exp_name>

set -euo pipefail

EXP_NAME=${1:-}

if [ -z "$EXP_NAME" ]; then
    echo "❌ Error: <exp_name> is required."
    echo "Usage: bash tools/create_exp.sh <exp_name>"
    exit 1
fi

# =========================================================
# Resolve project root
# =========================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

EXP_ROOT="${PROJECT_ROOT}/experiments"
OUT_ROOT="${PROJECT_ROOT}/outputs"

mkdir -p "${EXP_ROOT}"
mkdir -p "${OUT_ROOT}"

# =========================================================
# Generate experiment ID
# =========================================================

LAST_ID=$(
  find "$EXP_ROOT" -mindepth 1 -maxdepth 1 -type d ! -name latest -printf '%f\n' \
  | awk -F_ '/^[0-9]{4}_/ {print $1}' \
  | sort -n \
  | tail -n1
)

if [ -z "${LAST_ID:-}" ]; then
  NEXT_ID="0001"
else
  NEXT_ID=$(printf "%04d" $((10#$LAST_ID + 1)))
fi

if [ -z "${LAST_ID:-}" ]; then
    NEXT_ID=1
else
    NEXT_ID=$((10#$LAST_ID + 1))
fi

EXP_ID=$(printf "%04d" "${NEXT_ID}")

# =========================================================
# Naming
# =========================================================

DATE=$(date +"%Y%m%d")

# Sanitize experiment name
SAFE_EXP_NAME=$(
    echo "$EXP_NAME" \
    | tr ' /' '__' \
    | tr -cd '[:alnum:]_-'
)

DIR_NAME="${EXP_ID}_${DATE}_${SAFE_EXP_NAME}"

EXP_PATH="${EXP_ROOT}/${DIR_NAME}"
OUT_PATH="${OUT_ROOT}/${DIR_NAME}"

# =========================================================
# Safety check
# =========================================================

if [ -e "${EXP_PATH}" ]; then
    echo "❌ Experiment already exists:"
    echo "   ${DIR_NAME}"
    exit 1
fi

# =========================================================
# Create directories
# =========================================================

mkdir -p "${EXP_PATH}"
mkdir -p "${OUT_PATH}"

# =========================================================
# Git info
# =========================================================

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "git_not_available")
GIT_DIFF=$(git diff HEAD 2>/dev/null || true)

# =========================================================
# Resolve partition (owner + time-scale) and signal margin
#
# テンプレートの #SBATCH --time= を初期値として、実験作成時に一度だけ
# partition と time-limit警告用のsignal marginを計算し埋め込む。
# 後で run_slurm.sh の --time を大きく変更した場合、この2つは自動追従
# しないため、必要なら手動で --partition/--signal も合わせて編集すること。
# =========================================================

TEMPLATE_DIR="${PROJECT_ROOT}/templates"

_resolve_partition() {
    local root="$1"
    local time_limit="$2"
    local owner=""

    case "$root" in
        /workspace/andre01/*) owner="andre01" ;;
        /workspace/david01/*) owner="david01" ;;
        /workspace/david02/*) owner="david02" ;;
        /workspace/filesrv01/*) owner="creator" ;;
        /workspace/grace01/*) owner="grace01" ;;
        /workspace/grace02/*) owner="grace02" ;;
        *)
            echo "❌ Could not infer partition owner from path: ${root}" >&2
            return 1
            ;;
    esac

    local hh mm ss total_minutes scale
    IFS=: read -r hh mm ss <<< "$time_limit"
    total_minutes=$((10#$hh * 60 + 10#$mm))

    if [ "$total_minutes" -le 60 ]; then
        scale="small"
    elif [ "$total_minutes" -le 120 ]; then
        scale="medium"
    elif [ "$total_minutes" -le 240 ]; then
        scale="large"
    else
        scale="x-large"
    fi

    echo "${scale}-${owner}"
}

_resolve_signal_margin() {
    local time_limit="$1"
    local hh mm ss total_seconds margin

    IFS=: read -r hh mm ss <<< "$time_limit"
    total_seconds=$((10#$hh * 3600 + 10#$mm * 60 + 10#$ss))
    margin=$((total_seconds / 100))
    [ "$margin" -lt 30 ] && margin=30

    echo "$margin"
}

TEMPLATE_DEFAULT_TIME=$(
    grep -oP '(?<=--time=)\S+' "${TEMPLATE_DIR}/run_slurm.sh" | head -n1
)

PARTITION=$(_resolve_partition "${PROJECT_ROOT}" "${TEMPLATE_DEFAULT_TIME}")
SIGNAL_MARGIN=$(_resolve_signal_margin "${TEMPLATE_DEFAULT_TIME}")

# =========================================================
# Templates
# =========================================================

# run_slurm.sh
sed \
    -e "s|__EXP_NAME__|${DIR_NAME}|g" \
    -e "s|__PROJECT_ROOT__|${PROJECT_ROOT}|g" \
    -e "s|__PARTITION__|${PARTITION}|g" \
    -e "s|__SIGNAL_MARGIN__|${SIGNAL_MARGIN}|g" \
    "${TEMPLATE_DIR}/run_slurm.sh" \
    > "${EXP_PATH}/run_slurm.sh"

chmod +x "${EXP_PATH}/run_slurm.sh"

# experiment.py
cp \
    "${TEMPLATE_DIR}/experiment.py" \
    "${EXP_PATH}/experiment.py"

# config.yml
cp \
    "${TEMPLATE_DIR}/config.yml" \
    "${EXP_PATH}/config.yml"

# =========================================================
# Metadata
# =========================================================

cat > "${EXP_PATH}/metadata.yaml" <<EOF
exp_id: ${EXP_ID}
exp_name: ${DIR_NAME}
created_date: ${DATE}
git_commit: ${COMMIT_HASH}
EOF

# =========================================================
# Save uncommitted diff
# =========================================================

if [ -n "${GIT_DIFF}" ]; then
    echo "${GIT_DIFF}" \
        > "${EXP_PATH}/uncommitted_changes.diff"
fi

# =========================================================
# Update latest symlink
# =========================================================

ln -sfn "${EXP_PATH}" "${EXP_ROOT}/latest"
ln -sfn "${OUT_PATH}" "${OUT_ROOT}/latest"

# =========================================================
# Done
# =========================================================

echo ""
echo "✅ Created experiment"
echo ""
echo "ID        : ${EXP_ID}"
echo "Name      : ${DIR_NAME}"
echo "Experiment: ${EXP_PATH}"
echo "Outputs   : ${OUT_PATH}"
echo ""