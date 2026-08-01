#!/bin/bash
set -euo pipefail

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
# Metadata helper (Python/PyYAML に依存しない。
# scripts/slurm_entry.sh の _yaml_put と同じ方式)
# =====================================================

_yaml_escape() {
    printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

_yaml_put() {
    local yaml_file="$1"
    local key="$2"
    local value
    value=$(_yaml_escape "$3")

    if [ -f "${yaml_file}" ] && grep -q "^${key}:" "${yaml_file}"; then
        local tmp="${yaml_file}.tmp.$$"
        awk -v k="${key}" -v v="${value}" '
            $0 ~ "^" k ":" { printf "%s: \"%s\"\n", k, v; next }
            { print }
        ' "${yaml_file}" > "${tmp}"
        mv "${tmp}" "${yaml_file}"
    else
        printf '%s: "%s"\n' "${key}" "${value}" >> "${yaml_file}"
    fi
}

# =====================================================
# Resolve metadata
# Array job の場合 logs 下に {job_id}_{array_id}/ ができるため、
# 完全一致（{job_id}/）と array task 一致（{job_id}_*/）だけを対象にする
# （前方一致だと job_id=123 が 1234/12345 にもマッチしてしまうため不可）
# =====================================================

META_FILES=$(
    find "${PROJECT_ROOT}/logs" \
        \( -path "*/${JOB_ID}/run_metadata.yaml" -o -path "*/${JOB_ID}_*/run_metadata.yaml" \) \
        2>/dev/null
)

# =====================================================
# Parse exp name for Slack
# (最初に見つかったメタデータから取得)
# =====================================================

FIRST_META=$(echo "${META_FILES}" | head -n1)

EXP_NAME=""

if [ -n "${FIRST_META}" ]; then
    EXP_NAME=$(
        grep '^exp_name:' "${FIRST_META}" \
        | awk '{print $2}' \
        | tr -d '"'
    )
fi

if [ -z "${META_FILES}" ]; then
    echo "⚠️  Could not find metadata for job ${JOB_ID}"
    echo "    Proceeding with scancel only."
fi

# =====================================================
# Cancel
#
# scancel/qdel を先に実行する。ここが失敗した場合、メタデータに
# "CANCELLED" と記録したり Slack に通知したりすると、ジョブは実際には
# 動き続けているのに記録上は死んだことになってしまうため、
# メタデータ更新・Slack通知は scancel 成功後にのみ行う。
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

# =====================================================
# Update metadata for all matching tasks
# =====================================================

if [ -n "${META_FILES}" ]; then
    while IFS= read -r META_FILE; do
        echo "Updating: ${META_FILE}"
        _yaml_put "${META_FILE}" "status" "CANCELLED"
        _yaml_put "${META_FILE}" "cancel_reason" "${REASON}"
        _yaml_put "${META_FILE}" "cancelled_by" "${USER}"
    done <<< "${META_FILES}"
fi

# =====================================================
# Slack notify
# =====================================================

source "${PROJECT_ROOT}/scripts/notify_slack.sh"

export EXP_NAME="${EXP_NAME}"
export JOB_ID="${JOB_ID}"

notify_fail_fast "USER_CANCEL (reason: ${REASON}, by: ${USER})"

echo "✅ Done"