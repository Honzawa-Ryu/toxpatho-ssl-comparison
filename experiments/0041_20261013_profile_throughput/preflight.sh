#!/bin/bash
# exp 0041(プロファイル)の投入前点検。qsub はしない。
set -euo pipefail
EXP_NAME="0041_20261013_profile_throughput"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fail() { echo "NG: $*" >&2; exit 1; }
echo "=== exp 0041 preflight ==="
VENV_CFG="${PROJECT_ROOT}/.venv/pyvenv.cfg"
[ -f "${VENV_CFG}" ] && grep -q "^home = /usr/bin" "${VENV_CFG}" && grep -q "^include-system-site-packages = true" "${VENV_CFG}" \
  || fail ".venv が壊れている。bash tools/rebuild_venv.sh --apply"
[ -d "${PROJECT_ROOT}/.venv/lib/python3.12/site-packages/timm" ] || fail ".venv に timm が無い"
echo "ok : .venv は健全"
[ -f "${PROJECT_ROOT}/data/ssl_patches/patches.memmap" ] || fail "patches.memmap が無い"
for e in 0035_20261011_bt_vitb16_lrbias_quarter_clip1 0038_20261012_simsiam_vitb16_step_matched 0040_20261013_mae_vitb16_lr_half; do
  [ -f "${PROJECT_ROOT}/experiments/${e}/run_slurm.sh" ] || fail "${e}/run_slurm.sh が無い"
done
echo "ok : 対象の 3 実験の定義あり"
mkdir -p "${PROJECT_ROOT}/logs/${EXP_NAME}" "${PROJECT_ROOT}/outputs/${EXP_NAME}"
python3 -c "import ast; ast.parse(open('${PROJECT_ROOT}/scripts/train/profile_throughput.py').read())" || fail "スクリプトの構文エラー"
grep -q "^#PBS -l select=1" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "select=1 でない"
echo "ok : 1 ノード・walltime 1h(最大 1 ノード時間)"
echo
echo "=== 点検通過。投入するなら(⚠️ ユーザー承認後) ==="
echo "  cd ${PROJECT_ROOT} && qsub experiments/${EXP_NAME}/run_slurm.sh"
