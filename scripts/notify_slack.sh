#!/bin/bash

# JOB IDから色を生成 (6桁16進数カラーコード)
generate_color_from_id() {
    local job_id="$1"
    # JOB IDをハッシュ化して色に変換
    local hash=$(echo -n "$job_id" | md5sum | cut -c1-6)
    echo "#${hash}"
}

# Slack通知を送信 (attachments形式で色付き)
notify_slack() {
    local title="$1"
    local emoji="$2"
    local color="$3"
    local fields="$4"
    
    [ -z "${SLACK_WEBHOOK_URL:-}" ] && return 0
    
    # JSON構築
    local payload=$(jq -nc \
        --arg title "$title" \
        --arg emoji "$emoji" \
        --arg color "$color" \
        --arg fields "$fields" \
        '{
            text: ($emoji + " " + $title),
            attachments: [{
                color: $color,
                fields: ($fields | split("\n") | map(split(" : ") | {title: .[0], value: .[1], short: true}))
            }]
        }')
    
    curl -s -X POST \
        -H 'Content-type: application/json' \
        --data "$payload" \
        "${SLACK_WEBHOOK_URL}" \
        > /dev/null
}

_array_fields() {
    [ "${RUN_MODE:-}" != "array" ] && return
    echo "TASK : $((SLURM_ARRAY_TASK_ID + 1))/${ARRAY_TOTAL_TASKS}"
    echo "OPTS : ${ARRAY_TASK_OPTIONS}"
}

_extra_fields() {
    [ -n "${ADD_NOTIFY:-}" ] && echo "${ADD_NOTIFY}"
}

_append_optional_fields() {
    local -n _fields_ref=$1
    local array_info extra_info

    array_info=$(_array_fields)
    [ -n "${array_info}" ] && _fields_ref="${_fields_ref}
${array_info}"

    extra_info=$(_extra_fields)
    [ -n "${extra_info}" ] && _fields_ref="${_fields_ref}
${extra_info}"
}

notify_start() {
    [ "${SLACK_NOTIFY_ON_START:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}
PART : ${SLURM_JOB_PARTITION}
TIME : ${JOB_TIME_LIMIT}"

    _append_optional_fields fields
    notify_slack "STARTED" "🚀" "$color" "$fields"
}

notify_finish() {
    [ "${SLACK_NOTIFY_ON_FINISH:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}"

    _append_optional_fields fields
    notify_slack "FINISHED" "✅" "$color" "$fields"
}

notify_fail() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}
STATE : $1"

    _append_optional_fields fields
    notify_slack "FAILED" "❌" "$color" "$fields"
}

notify_fail_fast() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${SLURM_JOB_ID}")
    local fields="EXP : ${EXP_NAME}
JOB : ${SLURM_JOB_ID}
STATE : $1"

    _append_optional_fields fields
    notify_slack "INTERRUPTED" "⚡" "$color" "$fields"
}