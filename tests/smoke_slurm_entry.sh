#!/usr/bin/env bash
# scripts/slurm_entry.sh のスモークテスト。
#
# 目的は「ジョブ本体が1行も実行されずに無言で死ぬ」経路の検出。この失敗モードは
# 8日回すはずのGPUジョブが起動直後に落ちる形で現れるため、実機に投げる前に
# ここで捕まえたい。具体的に守っているのは:
#
#   - SCHEDULER=local / pbs で notify_* が set -u に触れないこと
#   - #PBS -l walltime= 形式のヘッダーでも JOB_TIME_LIMIT の取得で落ちないこと
#   - run_metadata.yaml が PyYAML 無しで書け、status が正しく更新されること
#   - SIF_PATH 未設定時に、原因の分かるメッセージで落ちること
#
# 実行: bash tests/smoke_slurm_entry.sh
# 依存: bash / coreutils のみ（apptainer も python の yaml も不要）

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

FAILURES=0
PASSES=0

_ok()   { PASSES=$((PASSES + 1)); echo "  ✅ $1"; }
_fail() { FAILURES=$((FAILURES + 1)); echo "  ❌ $1"; }

# =====================================================
# Fixture: slurm_entry.sh が動く最小の PROJECT_ROOT を組み立てる
# =====================================================

# $1: exp_name, $2: ヘッダー種別 (sbatch|pbs|none)
_make_project() {
    local exp_name="$1"
    local header_kind="$2"
    local root
    root=$(mktemp -d)

    mkdir -p "${root}/scripts" "${root}/logs/${exp_name}" \
             "${root}/experiments/${exp_name}" "${root}/outputs" \
             "${root}/data" "${root}/.venv/bin" "${root}/bin"

    cp "${REPO_ROOT}/scripts/slurm_entry.sh" "${root}/scripts/"
    cp "${REPO_ROOT}/scripts/notify_slack.sh" "${root}/scripts/"

    # apptainer / .venv はスモークテストの対象外なのでスタブで置き換える。
    # slurm_entry.sh は `apptainer exec [opts] <sif> bash -c <script>` の形で呼ぶので、
    # スタブは最後の引数（script本体）だけを取り出して素の bash で実行する。
    cat > "${root}/bin/apptainer" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
script="${!#}"   # 最後の引数 = bash -c に渡されるスクリプト本体
exec bash -c "${script}"
STUB
    chmod +x "${root}/bin/apptainer"

    # rsync はこの smoke test 環境に無いことがあるため、`rsync -a SRC DST` の
    # trailing-slash 有無（内容コピー/サブディレクトリごとコピー）だけを
    # 再現する最小スタブで代用する。
    cat > "${root}/bin/rsync" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
src="${@: -2:1}"
dst="${@: -1}"
mkdir -p "${dst}"
if [[ "${src}" == */ ]]; then
    cp -a "${src}." "${dst}"
else
    cp -a "${src}" "${dst}"
fi
STUB
    chmod +x "${root}/bin/rsync"

    echo '# stub venv activate (smoke test)' > "${root}/.venv/bin/activate"

    {
        case "${header_kind}" in
            sbatch) echo '#SBATCH --time=00:10:00' ;;
            pbs)    echo '#PBS -l walltime=00:10:00' ;;
            none)   : ;;
        esac
    } > "${root}/experiments/${exp_name}/run_slurm.sh"

    echo "${root}"
}

# $1: root, $2: exp_name, 残り: env 追加設定
_run_entry() {
    local root="$1"; shift
    local exp_name="$1"; shift

    env -i \
        HOME="${HOME}" \
        USER="${USER:-smoke}" \
        PATH="${root}/bin:/usr/local/bin:/usr/bin:/bin" \
        PROJECT_ROOT="${root}" \
        EXP_NAME="${exp_name}" \
        SCRATCH_ROOT="${root}/scratch" \
        RUN_COMMAND='echo SMOKE_BODY_EXECUTED' \
        "$@" \
        bash -c 'source "${PROJECT_ROOT}/scripts/slurm_entry.sh"'
}

_status_of() {
    grep -E '^status:' "$1" | head -n1 | awk '{print $2}' | tr -d '"'
}

# =====================================================
# 1. SCHEDULER=local: 本体が実行され COMPLETED になること
# =====================================================

echo "▶ local scheduler, #SBATCH header"
ROOT=$(_make_project smoke_local sbatch)
if OUT=$(_run_entry "${ROOT}" smoke_local SIF_PATH=/nonexistent/dummy.sif 2>&1); then
    if grep -q SMOKE_BODY_EXECUTED <<< "${OUT}"; then
        _ok "ジョブ本体が実行された"
    else
        _fail "ジョブ本体が実行されなかった: ${OUT}"
    fi

    META="${ROOT}/logs/smoke_local/$(ls "${ROOT}/logs/smoke_local" | grep -v latest | head -n1)/run_metadata.yaml"
    if [ -f "${META}" ] && [ "$(_status_of "${META}")" = "COMPLETED" ]; then
        _ok "run_metadata.yaml の status が COMPLETED"
    else
        _fail "run_metadata.yaml が期待通りでない: $(cat "${META}" 2>/dev/null)"
    fi

    if grep -q 'time_limit: "00:10:00"' "${META}"; then
        _ok "#SBATCH --time= から time_limit を取得した"
    else
        _fail "time_limit が取得できていない"
    fi
else
    _fail "SCHEDULER=local でジョブが落ちた: ${OUT}"
fi
rm -rf "${ROOT}"

# =====================================================
# 2. PBS 形式ヘッダー: walltime にフォールバックして落ちないこと
# =====================================================

echo "▶ pbs scheduler, #PBS -l walltime header"
ROOT=$(_make_project smoke_pbs pbs)
if OUT=$(_run_entry "${ROOT}" smoke_pbs SIF_PATH=/nonexistent/dummy.sif PBS_JOBID=12345.pbsserver PBS_QUEUE=regular 2>&1); then
    if grep -q SMOKE_BODY_EXECUTED <<< "${OUT}"; then
        _ok "PBS 環境でもジョブ本体が実行された"
    else
        _fail "PBS 環境でジョブ本体が実行されなかった: ${OUT}"
    fi

    META="${ROOT}/logs/smoke_pbs/12345/run_metadata.yaml"
    if [ -f "${META}" ] && grep -q 'time_limit: "00:10:00"' "${META}"; then
        _ok "#PBS -l walltime= から time_limit を取得した"
    else
        _fail "PBS の walltime が取得できていない: $(cat "${META}" 2>/dev/null)"
    fi
else
    _fail "PBS 環境でジョブが落ちた: ${OUT}"
fi
rm -rf "${ROOT}"

# =====================================================
# 3. 時間制限のヘッダーが無くても落ちないこと（grep 失敗の許容）
# =====================================================

echo "▶ no time-limit header"
ROOT=$(_make_project smoke_notime none)
if OUT=$(_run_entry "${ROOT}" smoke_notime SIF_PATH=/nonexistent/dummy.sif 2>&1); then
    _ok "時間制限ヘッダー無しでも完走した"
else
    _fail "時間制限ヘッダー無しで落ちた: ${OUT}"
fi
rm -rf "${ROOT}"

# =====================================================
# 4. 本体が非ゼロ終了したら FAILED になること
# =====================================================

echo "▶ failing body command"
ROOT=$(_make_project smoke_fail sbatch)
OUT=$(_run_entry "${ROOT}" smoke_fail SIF_PATH=/nonexistent/dummy.sif RUN_COMMAND='exit 3' 2>&1) || true
META="${ROOT}/logs/smoke_fail/$(ls "${ROOT}/logs/smoke_fail" | grep -v latest | head -n1)/run_metadata.yaml"
if [ -f "${META}" ] && [ "$(_status_of "${META}")" = "FAILED" ]; then
    _ok "非ゼロ終了で status=FAILED になった"
else
    _fail "非ゼロ終了が FAILED になっていない: $(cat "${META}" 2>/dev/null)"
fi
if grep -q 'fail_reason: "NONZERO_EXIT_3"' "${META}" 2>/dev/null; then
    _ok "fail_reason に終了コードが記録された"
else
    _fail "fail_reason が期待通りでない"
fi
rm -rf "${ROOT}"

# =====================================================
# 5. SIF_PATH 未設定は、原因の分かるメッセージで落ちること
# =====================================================

echo "▶ missing SIF_PATH"
ROOT=$(_make_project smoke_nosif sbatch)
if OUT=$(_run_entry "${ROOT}" smoke_nosif 2>&1); then
    _fail "SIF_PATH 未設定なのに成功してしまった"
else
    if grep -q 'SIF_PATH' <<< "${OUT}"; then
        _ok "SIF_PATH 未設定を名指しして落ちた"
    else
        _fail "SIF_PATH に触れないメッセージで落ちた: ${OUT}"
    fi
fi
rm -rf "${ROOT}"

# =====================================================
# 6. USE_LOCAL_SSD_INPUT=1 + DATA_SUBDIRS: 指定したサブディレクトリだけが
#    scratch へコピーされ、data/ 全体はコピーされないこと
# =====================================================

echo "▶ USE_LOCAL_SSD_INPUT=1 with DATA_SUBDIRS (scoped copy)"
ROOT=$(_make_project smoke_subdirs sbatch)
mkdir -p "${ROOT}/data/keep" "${ROOT}/data/skip"
echo keep > "${ROOT}/data/keep/file.txt"
echo skip > "${ROOT}/data/skip/file.txt"

if OUT=$(env -i \
    HOME="${HOME}" \
    USER="${USER:-smoke}" \
    PATH="${ROOT}/bin:/usr/local/bin:/usr/bin:/bin" \
    PROJECT_ROOT="${ROOT}" \
    EXP_NAME="smoke_subdirs" \
    SCRATCH_ROOT="${ROOT}/scratch" \
    RUN_COMMAND='echo SMOKE_BODY_EXECUTED' \
    USE_LOCAL_SSD_INPUT=1 \
    SIF_PATH=/nonexistent/dummy.sif \
    bash -c 'DATA_SUBDIRS=("keep"); source "${PROJECT_ROOT}/scripts/slurm_entry.sh"' 2>&1); then

    SCRATCH_DATA=$(find "${ROOT}/scratch" -maxdepth 3 -type d -name data | head -n1)

    if [ -f "${SCRATCH_DATA}/keep/file.txt" ]; then
        _ok "指定したサブディレクトリ(keep)はコピーされた"
    else
        _fail "keep がコピーされていない: ${OUT}"
    fi

    if [ ! -e "${SCRATCH_DATA}/skip" ]; then
        _ok "指定していないサブディレクトリ(skip)はコピーされなかった"
    else
        _fail "DATA_SUBDIRS を無視して skip までコピーしてしまった"
    fi
else
    _fail "DATA_SUBDIRS 指定ありでジョブが落ちた: ${OUT}"
fi
rm -rf "${ROOT}"

# =====================================================
# 7. USE_LOCAL_SSD_INPUT=1 かつ DATA_SUBDIRS 未指定: 後方互換で
#    data/ 全体がコピーされること
# =====================================================

echo "▶ USE_LOCAL_SSD_INPUT=1 without DATA_SUBDIRS (whole-dir fallback)"
ROOT=$(_make_project smoke_wholedir sbatch)
mkdir -p "${ROOT}/data/keep" "${ROOT}/data/skip"
echo keep > "${ROOT}/data/keep/file.txt"
echo skip > "${ROOT}/data/skip/file.txt"

if OUT=$(_run_entry "${ROOT}" smoke_wholedir SIF_PATH=/nonexistent/dummy.sif USE_LOCAL_SSD_INPUT=1 2>&1); then

    SCRATCH_DATA=$(find "${ROOT}/scratch" -maxdepth 3 -type d -name data | head -n1)

    if [ -f "${SCRATCH_DATA}/keep/file.txt" ] && [ -f "${SCRATCH_DATA}/skip/file.txt" ]; then
        _ok "DATA_SUBDIRS 未指定時は data/ 全体がコピーされた（後方互換）"
    else
        _fail "DATA_SUBDIRS 未指定時の全体コピーが期待通りでない: ${OUT}"
    fi
else
    _fail "DATA_SUBDIRS 未指定・全体コピーでジョブが落ちた: ${OUT}"
fi
rm -rf "${ROOT}"

# =====================================================
# Summary
# =====================================================

echo ""
echo "smoke_slurm_entry: ${PASSES} passed, ${FAILURES} failed"
[ "${FAILURES}" -eq 0 ]
