#!/usr/bin/env bash
# scripts/exp_common.sh の純粋関数群に対するユニットテスト。
#
# ここでの主目的は report.md が指摘したバグの再発防止:
#   - `latest` symlink が実験として拾われないこと（exp_list_names/exp_latest_name/exp_resolve_dir）
#   - partition owner 解決が EXP_PARTITION_OWNER で上書きできること
#   - time_limit → partition scale / signal margin の変換が期待通りであること
#
# 実行: bash tests/test_exp_common.sh
# 依存: bash / coreutils / findutils のみ

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPO_ROOT}/scripts/exp_common.sh"

FAILURES=0
PASSES=0

_ok()   { PASSES=$((PASSES + 1)); echo "  ✅ $1"; }
_fail() { FAILURES=$((FAILURES + 1)); echo "  ❌ $1"; }

_assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "${expected}" = "${actual}" ]; then
        _ok "${desc}"
    else
        _fail "${desc} (expected='${expected}' actual='${actual}')"
    fi
}

# =====================================================
# Fixture: experiments/ に latest symlink を混ぜて用意する
# =====================================================

EXP_ROOT=$(mktemp -d)
mkdir -p "${EXP_ROOT}/0001_20260101_first" "${EXP_ROOT}/0002_20260102_second"
ln -sfn "${EXP_ROOT}/0002_20260102_second" "${EXP_ROOT}/latest"

# =====================================================
# 1. exp_list_names: latest を除外し、ID昇順で列挙する
# =====================================================

echo "▶ exp_list_names"
LIST=$(exp_list_names "${EXP_ROOT}" | tr '\n' ',')
_assert_eq "latest を含まずID昇順" "0001_20260101_first,0002_20260102_second," "${LIST}"

# =====================================================
# 2. exp_latest_name / exp_last_id / exp_next_id
# =====================================================

echo "▶ exp_latest_name / exp_next_id"
_assert_eq "最新は0002 (latestシンボリックリンクではない)" \
    "0002_20260102_second" "$(exp_latest_name "${EXP_ROOT}")"
_assert_eq "次のIDは0003" "0003" "$(exp_next_id "${EXP_ROOT}")"

EMPTY_ROOT=$(mktemp -d)
_assert_eq "実験が無ければ次のIDは0001" "0001" "$(exp_next_id "${EMPTY_ROOT}")"
rm -rf "${EMPTY_ROOT}"

# =====================================================
# 3. exp_resolve_dir: 空入力/ID/名前のいずれでも latest symlink を拾わない
# =====================================================

echo "▶ exp_resolve_dir"
_assert_eq "空入力は最新の実験ディレクトリ" \
    "${EXP_ROOT}/0002_20260102_second" "$(exp_resolve_dir "${EXP_ROOT}" "")"
_assert_eq "ID指定(1)は0001をゼロ埋めで解決" \
    "${EXP_ROOT}/0001_20260101_first" "$(exp_resolve_dir "${EXP_ROOT}" "1")"
_assert_eq "実験名を直接指定した場合はそのまま結合" \
    "${EXP_ROOT}/0002_20260102_second" "$(exp_resolve_dir "${EXP_ROOT}" "0002_20260102_second")"

rm -rf "${EXP_ROOT}"

# =====================================================
# 4. exp_resolve_owner: パス推測とEXP_PARTITION_OWNERによる上書き
# =====================================================

echo "▶ exp_resolve_owner"
_assert_eq "filesrv02 は creator" \
    "creator" "$(exp_resolve_owner /workspace/filesrv02/someuser/proj)"
_assert_eq "andre01 は andre01" \
    "andre01" "$(exp_resolve_owner /workspace/andre01/proj)"

if OUT=$(exp_resolve_owner /workspace/unknown_partition/proj 2>&1); then
    _fail "未知のpartitionなのに成功してしまった: ${OUT}"
else
    if grep -q "EXP_PARTITION_OWNER" <<< "${OUT}"; then
        _ok "未知のpartitionはEXP_PARTITION_OWNERの案内付きで失敗"
    else
        _fail "未知のpartitionの失敗メッセージが期待通りでない: ${OUT}"
    fi
fi

_assert_eq "EXP_PARTITION_OWNERが未知パスでも優先される" \
    "myuser" "$(EXP_PARTITION_OWNER=myuser exp_resolve_owner /workspace/unknown_partition/proj)"

# =====================================================
# 5. exp_resolve_partition: time_limit → scale-owner
# =====================================================

echo "▶ exp_resolve_partition"
_assert_eq "60分以下はsmall" \
    "small-creator" "$(exp_resolve_partition /workspace/filesrv02/proj 00:30:00)"
_assert_eq "60分超120分以下はmedium" \
    "medium-creator" "$(exp_resolve_partition /workspace/filesrv02/proj 01:30:00)"
_assert_eq "120分超240分以下はlarge" \
    "large-creator" "$(exp_resolve_partition /workspace/filesrv02/proj 03:30:00)"
_assert_eq "240分超はx-large" \
    "x-large-creator" "$(exp_resolve_partition /workspace/filesrv02/proj 10:00:00)"

# =====================================================
# 6. exp_resolve_signal_margin: time_limit秒数の1%（最低30秒）
# =====================================================

echo "▶ exp_resolve_signal_margin"
_assert_eq "短い時間制限は最低30秒にクランプ" \
    "30" "$(exp_resolve_signal_margin 00:10:00)"
_assert_eq "2時間は72秒" \
    "72" "$(exp_resolve_signal_margin 02:00:00)"
_assert_eq "10時間は360秒" \
    "360" "$(exp_resolve_signal_margin 10:00:00)"

# =====================================================
# Summary
# =====================================================

echo ""
echo "test_exp_common: ${PASSES} passed, ${FAILURES} failed"
[ "${FAILURES}" -eq 0 ]
