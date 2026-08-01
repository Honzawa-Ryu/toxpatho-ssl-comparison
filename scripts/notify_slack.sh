#!/bin/bash
# Slack 通知ヘルパー。
#
# ここで参照してよいジョブ情報は scripts/slurm_entry.sh がスケジューラ非依存に
# export する JOB_ID / JOB_PARTITION / ARRAY_TASK_ID / JOB_TIME_LIMIT だけである。
# SLURM_* を直接参照すると PBS と SCHEDULER=local で `set -u` により未定義変数エラーになり、
# notify_start の失敗が ERR trap を踏んで「本体コマンドを1行も実行せずジョブ終了」になる。
# 単体で source される経路（tools/cancel_job.sh）もあるため、全参照に既定値を付ける。

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

    if ! command -v jq >/dev/null 2>&1; then
        echo "⚠️  jq が無いため Slack通知をスキップします: ${title}" >&2
        return 0
    fi

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

# 以下のヘルパーは「該当情報が無ければ何も出さない」のが正常系なので、
# 空を返す経路が非ゼロ終了にならないよう明示的に `return 0` する
# （呼び出し元は set -e 下の slurm_entry.sh。ここで 1 を返すと通知どころか
#  ジョブ本体が落ちる）。
_array_fields() {
    [ "${RUN_MODE:-}" = "array" ] || return 0
    echo "TASK : $(( ${ARRAY_TASK_ID:-0} + 1 ))/${ARRAY_TOTAL_TASKS:-?}"
    echo "OPTS : ${ARRAY_TASK_OPTIONS:-}"
}

_extra_fields() {
    [ -n "${ADD_NOTIFY:-}" ] || return 0
    echo "${ADD_NOTIFY}"
}

_append_optional_fields() {
    local -n _fields_ref=$1
    local array_info extra_info

    array_info=$(_array_fields)
    if [ -n "${array_info}" ]; then
        _fields_ref="${_fields_ref}
${array_info}"
    fi

    extra_info=$(_extra_fields)
    if [ -n "${extra_info}" ]; then
        _fields_ref="${_fields_ref}
${extra_info}"
    fi

    return 0
}

# EXP / JOB / STATE の3フィールドは全通知で共通なので1箇所にまとめる
_base_fields() {
    echo "EXP : ${EXP_NAME:-unknown}"
    echo "JOB : ${JOB_ID:-unknown}"
}

notify_start() {
    [ "${SLACK_NOTIFY_ON_START:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="$(_base_fields)
PART : ${JOB_PARTITION:-unknown}
TIME : ${JOB_TIME_LIMIT:-unknown}"

    _append_optional_fields fields
    notify_slack "STARTED" "🚀" "$color" "$fields"
}

notify_finish() {
    [ "${SLACK_NOTIFY_ON_FINISH:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="$(_base_fields)"

    _append_optional_fields fields
    notify_slack "FINISHED" "✅" "$color" "$fields"
}

notify_fail() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="$(_base_fields)
STATE : $1"

    _append_optional_fields fields
    notify_slack "FAILED" "❌" "$color" "$fields"
}

notify_fail_fast() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="$(_base_fields)
STATE : $1"

    _append_optional_fields fields
    notify_slack "INTERRUPTED" "⚡" "$color" "$fields"
}

# 残り時間僅少の警告。ジョブは継続するため ⚡ INTERRUPTED とは別の絵文字にする
# （README の通知表では ⚡ =「キャンセル・割り込み」なので、これを使うと
#  通知を見た人が「中断された」と誤解する）。
notify_time_limit_warning() {
    [ "${SLACK_NOTIFY_ON_FAIL:-1}" -eq 1 ] || return 0

    local color=$(generate_color_from_id "${JOB_ID:-unknown}")
    local fields="$(_base_fields)
TIME : ${JOB_TIME_LIMIT:-unknown}
STATE : ${1:-TIME_LIMIT_WARNING}"

    _append_optional_fields fields
    notify_slack "TIME_LIMIT_WARNING (job still running)" "⏳" "$color" "$fields"
}
