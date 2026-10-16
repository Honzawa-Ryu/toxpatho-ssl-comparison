#!/bin/bash
# exp 0047(プロファイル)の投入前点検。qsub はしない。
set -euo pipefail
EXP_NAME="0047_20261016_bt_val_recompute"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fail() { echo "NG: $*" >&2; exit 1; }
echo "=== exp 0047 preflight ==="
VENV_CFG="${PROJECT_ROOT}/.venv/pyvenv.cfg"
[ -f "${VENV_CFG}" ] && grep -q "^home = /usr/bin" "${VENV_CFG}" && grep -q "^include-system-site-packages = true" "${VENV_CFG}" \
  || fail ".venv が壊れている。bash tools/rebuild_venv.sh --apply"
[ -d "${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/timm" ] || fail ".venv に timm が無い"
echo "ok : .venv は健全"
[ -f "${PROJECT_ROOT}/data/ssl_patches/patches.memmap" ] || fail "patches.memmap が無い"
[ -f "${PROJECT_ROOT}/experiments/0035_20261011_bt_vitb16_lrbias_quarter_clip1/run_slurm.sh" ] || fail "0035 の定義が無い"
N=$(grep -c "name: ep" "${PROJECT_ROOT}/scripts/analysis/bt_epochs.yaml"); [ "$N" -ge 30 ] || fail "bt_epochs.yaml のスナップショットが少ない($N)"
for p in $(grep -oE "outputs/[^ ]+model_ep[0-9]+\.pt" "${PROJECT_ROOT}/scripts/analysis/bt_epochs.yaml"); do [ -f "${PROJECT_ROOT}/$p" ] || fail "$p が無い"; done
echo "ok : 0035 の定義と bt_epochs.yaml の $N 本のスナップショットあり"
mkdir -p "${PROJECT_ROOT}/logs/${EXP_NAME}" "${PROJECT_ROOT}/outputs/${EXP_NAME}"
python3 -c "import ast; ast.parse(open('${PROJECT_ROOT}/scripts/analysis/bt_val_recompute.py').read())" || fail "スクリプトの構文エラー"
grep -q "^#PBS -l select=1" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "select=1 でない"
grep -q "^#PBS -q debug-g" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "debug-g でない"
grep -q "def batch_stat_bn" "${PROJECT_ROOT}/lib/trainer/loop.py" || fail "batch_stat_bn(2026-10-16)が無い"
echo "ok : debug-g・1 ノード・walltime 30 分(最大 0.5 ノード時間)"
echo
echo "=== 点検通過。投入するなら(⚠️ ユーザー承認後) ==="
echo "  cd ${PROJECT_ROOT} && qsub experiments/${EXP_NAME}/run_slurm.sh"
