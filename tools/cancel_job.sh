#!/bin/bash

JOB_ID=${1:-}
REASON=${2:-"user_cancel"}

if [ -z "${JOB_ID}" ]; then
    echo "Usage:"
    echo "  cancel_job.sh <job_id> [reason]"
    exit 1
fi

# =====================================================
# Root
# =====================================================

PROJECT_ROOT=$(git rev-parse --show-toplevel)

# =====================================================
# Resolve metadata
# Array job の場合 logs 下に {job_id}_{array_id}/ や
# Slurm内部IDのディレクトリができるため前方一致で探す
# =====================================================

META_FILES=$(
    find "${PROJECT_ROOT}/logs" \
        -path "*/${JOB_ID}*/run_metadata.yaml" \
        2>/dev/null
)

if [ -z "${META_FILES}" ]; then
    echo "⚠️  Could not find metadata for job ${JOB_ID}"
    echo "    Proceeding with scancel only."
else
    # =====================================================
    # Update metadata for all matching tasks
    # =====================================================

    while IFS= read -r META_FILE; do

        echo "Updating: ${META_FILE}"

        python3 - <<EOF
from pathlib import Path
import yaml

path = Path("${META_FILE}")
data = yaml.safe_load(path.read_text())
data["status"] = "CANCELLED"
data["cancel_reason"] = "${REASON}"
data["cancelled_by"] = "${USER}"
path.write_text(yaml.safe_dump(data, sort_keys=False))
EOF

    done <<< "${META_FILES}"
fi

# =====================================================
# Parse exp name for Slack
# (最初に見つかったメタデータから取得)
# =====================================================

FIRST_META=$(echo "${META_FILES}" | head -n1)

EXP_NAME=""

if [ -n "${FIRST_META}" ]; then
    EXP_NAME=$(
        grep '^exp_name:' "${FIRST_META}" \
        | awk '{print $2}'
    )
fi

# =====================================================
# Slack notify
# =====================================================

source "${PROJECT_ROOT}/scripts/notify_slack.sh"

export EXP_NAME="${EXP_NAME}"
export SLURM_JOB_ID="${JOB_ID}"

notify_fail_fast "USER_CANCEL (reason: ${REASON}, by: ${USER})"

# =====================================================
# Cancel
# =====================================================

echo ""
echo "🛑 Cancelling job ${JOB_ID}"
echo ""

if command -v scancel &>/dev/null; then
    scancel "${JOB_ID}"
elif command -v qdel &>/dev/null; then
    # アレイジョブ対応：PBSではアレイジョブ全体を削除する場合、IDのみで指定可能
    qdel "${JOB_ID}"
else
    echo "❌ No scheduler cancel command found (scancel/qdel)"
    exit 1
fi

echo "✅ Done"