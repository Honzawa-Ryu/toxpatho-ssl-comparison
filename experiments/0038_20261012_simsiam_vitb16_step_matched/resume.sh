#!/bin/bash
# exp 0038 (Barlow Twins) の続き（2本目以降）を投入するための preflight + qsub。
#
#   bash experiments/0038_20261012_simsiam_vitb16_step_matched/resume.sh            # 点検のみ(既定)
#   bash experiments/0038_20261012_simsiam_vitb16_step_matched/resume.sh --submit   # 点検して qsub
#
# walltime 48h で切れたら、このスクリプトで続きを投入する(分/epoch は見積り。1 本目の実測で上書きすること)。
# `state.pt` は rank0 が**毎 epoch** 書いており（loop.py:266）、`--dir_result` で
# プロジェクト側の outputs/ を直接指しているので（scratch 経由ではない）、
# walltime で強制終了されても直前の epoch から再開できる。
#
# 落とし穴（lib/trainer/entry.py:179）: `model_ssl.pt` が存在すると
#   "model_ssl.pt already exists; training complete, nothing to resume."
# で即終了する。正常終了や abort の後はこれが書かれているので、続けたいなら退避が必要。
# walltime 切れの場合は書かれていないので、そのまま再投入すればよい。

set -euo pipefail

EXP_NAME="0038_20261012_simsiam_vitb16_step_matched"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PROJECT_ROOT}/outputs/${EXP_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}"
SCRIPT="${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh"
NUM_EPOCH=160
MIN_PER_EPOCH="${MIN_PER_EPOCH:-9.0}"   # ⚠️ 見積り(実測なし)。1 本目のログの elapsed_time で上書きすること
SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1

fail() { echo "NG: $*" >&2; exit 1; }

echo "=== exp 0038 resume preflight ==="

# 1. 同じ実験のジョブが走っていないか（二重投入すると同じ state.pt を両方が書く）
if qstat 2>/dev/null | grep -q "0038_si"; then
  qstat 2>/dev/null | grep "0038_si"
  fail "0038 のジョブがまだキュー/実行中。終わってから投入すること"
fi
echo "ok : 実行中の 0038 ジョブは無い"

# 2〜4. どこまで「保存済み」か。完了判定はログではなく保存済みの状態と成果物で行う(レビュー指摘 2026-10-12):
#   epoch のログ行は val_loss・state.pt 保存より前に出るので、ログの最終 epoch は保存されていないことがある。
#   state.json は state.pt を atomic に置き換えた**後**に書かれる(lib/trainer/loop.py:write_state_meta)。
STATE_JSON="${OUT}/state.json"
if [ -f "${STATE_JSON}" ]; then
  SAVED_EP=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['completed_epochs'])" "${STATE_JSON}") \
    || fail "state.json を読めない: ${STATE_JSON}"
  echo "ok : state.json の保存済み epoch = ${SAVED_EP} / ${NUM_EPOCH}（$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['saved_at'])" "${STATE_JSON}")）"
else
  # state.json を書く前のコードで走ったラン。ログの最終 epoch は未保存かもしれないので 1 引いて扱い、完了とはみなさない
  LOG_EP=$(grep -ah "Epoch: [0-9]*, train_loss" "${LOG_DIR}"/*.OU 2>/dev/null \
           | sed 's/.*Epoch: \([0-9]*\),.*/\1/' | sort -n | tail -1)
  [ -n "${LOG_EP}" ] || fail "state.json もログの epoch 行も無い。${OUT} と ${LOG_DIR} を確認すること"
  SAVED_EP=$(( LOG_EP - 1 ))
  echo "⚠️ : state.json が無い(旧コードのラン)。ログの最終 epoch ${LOG_EP} は未保存の可能性があるので ${SAVED_EP} とみなす"
fi
LAST_EP="${SAVED_EP}"

if [ -f "${OUT}/model_ssl.pt" ]; then
  if [ "${SAVED_EP}" -ge "${NUM_EPOCH}" ]; then
    echo "完走済み: model_ssl.pt があり、state.json は ${SAVED_EP}/${NUM_EPOCH} epoch。resume は不要。"
    exit 0
  fi
  echo "NG : ${OUT}/model_ssl.pt があるのに保存済みは ${SAVED_EP}/${NUM_EPOCH} epoch = 崩壊/NaN 等で途中停止したラン" >&2
  echo "     entry.py は model_ssl.pt があると即終了する。崩壊での停止なら resume せず新しい実験番号で(Goal.yaml 方針3)" >&2
  exit 1
fi
echo "ok : model_ssl.pt は無い（resume を妨げない）"

[ -f "${OUT}/state.pt" ] || fail "${OUT}/state.pt が無い。--resume は fresh start になる"
[ ! -f "${OUT}/state.pt.tmp" ] || echo "⚠️ : state.pt.tmp が残っている(保存中に終了した)。state.pt は直前の正常な版。tmp は消してよい"
echo "ok : state.pt あり ($(du -h "${OUT}/state.pt" | cut -f1), 更新 $(date -r "${OUT}/state.pt" '+%m/%d %H:%M'))"

REMAIN=$(( NUM_EPOCH - LAST_EP ))
if [ "${REMAIN}" -le 0 ]; then
  # 全 epoch 保存済みだが model_ssl.pt が無い(最終保存の直後に終了)。resume すればループは即抜けて成果物を書く
  REMAIN=1
  echo "ok : 全 epoch 保存済みだが model_ssl.pt が無い。resume で成果物だけ書き出す"
else
  echo "ok : 保存済み ${LAST_EP} / ${NUM_EPOCH}（残り ${REMAIN}）"
fi

# 5. 必要 walltime（MIN_PER_EPOCH に 30% の余裕）
NEED_H=$(awk -v r="${REMAIN}" -v m="${MIN_PER_EPOCH}" 'BEGIN{printf "%d", (r*m/60)*1.3 + 1}')
[ "${NEED_H}" -gt 48 ] && NEED_H=48
WALLTIME=$(printf "%02d:00:00" "${NEED_H}")
echo "ok : 必要 walltime の見積り ${WALLTIME}（残り ${REMAIN} epoch x ${MIN_PER_EPOCH}分 + 30%）"

# 6. 崩壊で止まったランを resume しない（Goal.yaml 方針3: 崩壊したら新しい実験番号で再実行）
#    collapse abort / NaN abort の後は model_ssl.pt が書かれるので項目2でも止まるが、
#    「退避すれば続けられる」と誤読されないよう、ここで理由を明示して止める。
LAST_LINE=$(grep -ah "Epoch: [0-9]*, train_loss" "${LOG_DIR}"/*.OU 2>/dev/null | tail -1)
echo "     最終行: ${LAST_LINE#*\[INFO\] }"
if grep -aqE "Collapse detected|Non-finite train_loss" "${LOG_DIR}"/*.OU 2>/dev/null; then
  grep -ahE "Collapse detected|Non-finite train_loss" "${LOG_DIR}"/*.OU | tail -2 >&2
  fail "崩壊検知で停止済み。resume しても回復しない。原因を切り分け、新しい実験番号で再実行すること"
fi
case "${LAST_LINE}" in
  *"train_loss: nan"*|*"train_loss: inf"*)
    fail "train_loss が非有限。resume しても回復しない" ;;
esac
echo "ok : 崩壊検知ログも非有限 loss も無い"

# 7. .venv が学習ジョブを動かせる状態か
#    ログインノードで `uv run` / `uv sync` を叩くと uv が .venv を作り直してしまい、
#    torch(コンテナ側)への道筋と timm が消える。2026-09-19 にこれで job 3398346 が
#    全4ノード即死した(ModuleNotFoundError: No module named 'torch')。
#    4ノードを確保してから落ちるのは高くつくので、投入前にここで止める。
VENV_CFG="${PROJECT_ROOT}/.venv/pyvenv.cfg"
VENV_SP="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages"
VENV_NG="uv に作り直された疑い。bash tools/rebuild_venv.sh --apply で直すこと"
[ -f "${VENV_CFG}" ] || fail ".venv が無い。bash tools/rebuild_venv.sh --apply で作り直すこと"
grep -q "^home = /usr/bin" "${VENV_CFG}" || {
  grep -E "^(home|include-system-site-packages) = " "${VENV_CFG}" >&2
  fail ".venv の土台がコンテナの python ではない。${VENV_NG}"
}
grep -q "^include-system-site-packages = true" "${VENV_CFG}" \
  || fail ".venv が system-site-packages を見ない=コンテナ側 torch が見えない。${VENV_NG}"
[ -d "${VENV_SP}/timm" ] || fail ".venv に timm が無い。${VENV_NG}"
echo "ok : .venv はコンテナ python 土台 + system-site-packages + timm あり"

QSUB_CMD="qsub -l walltime=${WALLTIME} ${SCRIPT}"

echo
if [ "${SUBMIT}" -eq 1 ]; then
  echo "=== 投入する ==="
  mkdir -p "${LOG_DIR}"
  cd "${PROJECT_ROOT}"
  eval "${QSUB_CMD}"
  echo "投入後の確認: qstat | grep 0038_si"
  echo "resume できたかの確認（ログに出るはず）:"
  echo "  grep 'Resumed from state.pt' ${LOG_DIR}/<新jobid>.opbs.OU"
else
  echo "=== 点検のみ。投入するなら ==="
  echo "  ${QSUB_CMD}"
  echo "  （または $0 --submit）"
fi
