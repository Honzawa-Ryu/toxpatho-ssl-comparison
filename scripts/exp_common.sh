#!/usr/bin/env bash
# 実験まわりの共通シェル関数（ディレクトリ解決と partition 解決）。
#
# これまで同じ「実験ディレクトリを列挙する / 最新を取る / IDから引く」処理が
# .bashrc.d/1-experiments.sh (cdx/lsx/runx)、tools/create_exp.sh、
# tools/resume_exp.sh、tools/mark_failed.sh、tools/rename_exp.sh に
# 4通りの実装で散らばっており、`latest` symlink を弾く/弾かないのズレが
# cdx/lsx だけに残っていた。全員がここを使うことで、この種のズレが
# 構造的に起きないようにする。
#
# 規約:
#   - experiments/ 直下で `NNNN_` (4桁ID + アンダースコア) で始まるディレクトリ
#     だけが実験。`latest` は直近の実験を指す symlink であって実験ではない。
#   - 並び順は常にID昇順（`sort -t _ -k1,1n`）。辞書順ではないので
#     `latest` や 2桁IDが紛れ込んでも壊れない。
#
# 使い方: source "${PROJECT_ROOT}/scripts/exp_common.sh"
# 依存: bash / GNU findutils / coreutils

# 実験ディレクトリ名をID昇順で列挙する（`latest` を除外）。
# 1つも無い場合は何も出力せず、exit 0 を返す（set -e 下の呼び出し元を落とさない）。
#
# $1: experiments/ のパス
exp_list_names() {
    local exp_root="$1"

    [ -d "${exp_root}" ] || return 0

    find "${exp_root}" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        ! -name latest \
        -printf '%f\n' 2>/dev/null \
    | { grep -E '^[0-9]{4}_' || true; } \
    | sort -t '_' -k1,1n
}

# 最新（最大ID）の実験ディレクトリ名。無ければ空文字。
# $1: experiments/ のパス
exp_latest_name() {
    exp_list_names "$1" | tail -n1
}

# 既存の最大ID（4桁ゼロ埋め）。実験が無ければ空文字。
# $1: experiments/ のパス
exp_last_id() {
    local latest
    latest=$(exp_latest_name "$1")
    [ -n "${latest}" ] || return 0
    printf '%s' "${latest%%_*}"
}

# 次に採番すべきID（4桁ゼロ埋め）。実験が無ければ 0001。
# $1: experiments/ のパス
exp_next_id() {
    local last
    last=$(exp_last_id "$1")

    if [ -z "${last}" ]; then
        printf '%04d' 1
    else
        printf '%04d' "$((10#${last} + 1))"
    fi
}

# ID（数値）から実験ディレクトリ名を引く。見つからなければ空文字。
# $1: experiments/ のパス, $2: ID（ゼロ埋めしていなくてよい）
exp_name_by_id() {
    local exp_root="$1"
    local id="$2"
    local padded
    padded=$(printf '%04d' "$((10#${id}))")

    exp_list_names "${exp_root}" | { grep -E "^${padded}_" || true; } | head -n1
}

# 「IDまたは実験名」を実験ディレクトリの絶対パスに解決する。
# 入力が空なら最新の実験を返す。解決できなければ空文字（存在確認は呼び出し側の責務）。
#
# $1: experiments/ のパス, $2: ID または実験名（省略可）
exp_resolve_dir() {
    local exp_root="$1"
    local input="${2:-}"
    local name=""

    if [ -z "${input}" ]; then
        name=$(exp_latest_name "${exp_root}")
    elif [[ "${input}" =~ ^[0-9]+$ ]]; then
        name=$(exp_name_by_id "${exp_root}" "${input}")
    else
        name="${input}"
    fi

    [ -n "${name}" ] || return 0
    printf '%s/%s' "${exp_root}" "${name}"
}

# =========================================================
# Partition / signal margin
#
# tools/create_exp.sh から使う。純粋な文字列→文字列の変換なので、
# tests/test_exp_common.sh で表の内容ごと検証できるようここに置いている。
# =========================================================

# パスから partition の owner を推測する。
#
# このテーブルは新しいファイルサーバが増えるたびに陳腐化するので、
# EXP_PARTITION_OWNER が設定されていればそちらを優先する
# （~/.bashrc で `export EXP_PARTITION_OWNER=yourname` しておけば表は使われない）。
#
# $1: プロジェクトルートの絶対パス
exp_resolve_owner() {
    local root="$1"

    if [ -n "${EXP_PARTITION_OWNER:-}" ]; then
        echo "${EXP_PARTITION_OWNER}"
        return 0
    fi

    case "$root" in
        /workspace/andre01/*)   echo "andre01" ;;
        /workspace/david01/*)   echo "david01" ;;
        /workspace/david02/*)   echo "david02" ;;
        /workspace/filesrv01/*) echo "creator" ;;
        /workspace/filesrv02/*) echo "creator" ;;
        /workspace/grace01/*)   echo "grace01" ;;
        /workspace/grace02/*)   echo "grace02" ;;
        *)
            echo "❌ Could not infer partition owner from path: ${root}" >&2
            echo "   EXP_PARTITION_OWNER を export して明示的に指定してください。" >&2
            echo "   例: EXP_PARTITION_OWNER=creator make create_exp name=<exp_name>" >&2
            return 1
            ;;
    esac
}

# 時間制限（HH:MM:SS）から partition 名（{scale}-{owner}）を組み立てる。
# $1: プロジェクトルートの絶対パス, $2: 時間制限
exp_resolve_partition() {
    local root="$1"
    local time_limit="$2"
    local owner

    owner=$(exp_resolve_owner "$root") || return 1

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

# 時間制限（HH:MM:SS）から --signal=B:USR1@<margin> のマージン秒数を出す。
# $1: 時間制限
exp_resolve_signal_margin() {
    local time_limit="$1"
    local hh mm ss total_seconds margin

    IFS=: read -r hh mm ss <<< "$time_limit"
    total_seconds=$((10#$hh * 3600 + 10#$mm * 60 + 10#$ss))
    margin=$((total_seconds / 100))
    [ "$margin" -lt 30 ] && margin=30

    echo "$margin"
}
