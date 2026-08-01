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

source "${PROJECT_ROOT}/scripts/exp_common.sh"

EXP_ROOT="${PROJECT_ROOT}/experiments"
OUT_ROOT="${PROJECT_ROOT}/outputs"

mkdir -p "${EXP_ROOT}"
mkdir -p "${OUT_ROOT}"

# =========================================================
# Generate experiment ID
# =========================================================

EXP_ID=$(exp_next_id "${EXP_ROOT}")

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
# Resolve partition (owner + time-scale) and signal margin
#
# テンプレートの #SBATCH --time= を初期値として、実験作成時に一度だけ
# partition と time-limit警告用のsignal marginを計算し埋め込む。
# 後で run_slurm.sh の --time を大きく変更した場合、この2つは自動追従
# しないため、必要なら手動で --partition/--signal も合わせて編集すること。
#
# ⚠️ この計算は必ずディレクトリ作成より前に行うこと。partition の解決に
#    失敗すると set -e で落ちるが、その時点で experiments/NNNN_.../ を
#    作ってしまっていると、空ディレクトリが残ってその ID が永久に消費される。
# =========================================================

# exp_resolve_partition / exp_resolve_signal_margin は scripts/exp_common.sh 側。
TEMPLATE_DIR="${PROJECT_ROOT}/templates"

TEMPLATE_DEFAULT_TIME=$(
    grep -oP '(?<=--time=)\S+' "${TEMPLATE_DIR}/run_slurm.sh" | head -n1
)

if [ -z "${TEMPLATE_DEFAULT_TIME:-}" ]; then
    echo "❌ templates/run_slurm.sh から #SBATCH --time= を読めませんでした。" >&2
    exit 1
fi

PARTITION=$(exp_resolve_partition "${PROJECT_ROOT}" "${TEMPLATE_DEFAULT_TIME}")
SIGNAL_MARGIN=$(exp_resolve_signal_margin "${TEMPLATE_DEFAULT_TIME}")

# =========================================================
# Create directories
#
# ここまでで失敗しうる処理（partition解決を含む）は全て終わっている。
# =========================================================

mkdir -p "${EXP_PATH}"
mkdir -p "${OUT_PATH}"

# =========================================================
# Git info
# =========================================================

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "git_not_available")
GIT_DIFF=$(git diff HEAD 2>/dev/null || true)

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