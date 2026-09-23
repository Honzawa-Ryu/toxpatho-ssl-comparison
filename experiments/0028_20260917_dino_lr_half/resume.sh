#!/bin/bash
# exp 0028 の続き（2本目）を投入するための preflight + qsub。
#
#   bash experiments/0028_20260917_dino_lr_half/resume.sh            # 点検のみ(既定)
#   bash experiments/0028_20260917_dino_lr_half/resume.sh --submit   # 点検して qsub
#
# 480 epoch は 6.9分/epoch で約55時間かかり、walltime 48h では ep415 前後で切れる。
# `state.pt` は rank0 が**毎 epoch** 書いており（loop.py:266）、`--dir_result` で
# プロジェクト側の outputs/ を直接指しているので（scratch 経由ではない）、
# walltime で強制終了されても直前の epoch から再開できる。
#
# 落とし穴（lib/trainer/entry.py:179）: `model_ssl.pt` が存在すると
#   "model_ssl.pt already exists; training complete, nothing to resume."
# で即終了する。正常終了や abort の後はこれが書かれているので、続けたいなら退避が必要。
# walltime 切れの場合は書かれていないので、そのまま再投入すればよい。

set -euo pipefail

EXP_NAME="0028_20260917_dino_lr_half"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PROJECT_ROOT}/outputs/${EXP_NAME}"
LOG_DIR="${PROJECT_ROOT}/logs/${EXP_NAME}"
SCRIPT="${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh"
NUM_EPOCH=480
MIN_PER_EPOCH=6.9      # 実測(job 3382307 完走: 419 epoch / 47.8h = 6.84分/epoch)
SUBMIT=0
[ "${1:-}" = "--submit" ] && SUBMIT=1

fail() { echo "NG: $*" >&2; exit 1; }

echo "=== exp 0028 resume preflight ==="

# 1. 同じ実験のジョブが走っていないか（二重投入すると同じ state.pt を両方が書く）
if qstat 2>/dev/null | grep -q "0028_dino"; then
  qstat 2>/dev/null | grep "0028_dino"
  fail "0028 のジョブがまだキュー/実行中。終わってから投入すること"
fi
echo "ok : 実行中の 0028 ジョブは無い"

# 2. resume を止めるファイルが無いか
if [ -f "${OUT}/model_ssl.pt" ]; then
  echo "NG : ${OUT}/model_ssl.pt が存在する。entry.py がここで即終了する" >&2
  echo "     続けるなら退避してから再実行:" >&2
  echo "       mv ${OUT}/model_ssl.pt ${OUT}/model_ssl_stopped_\$(date +%Y%m%d).pt" >&2
  exit 1
fi
echo "ok : model_ssl.pt は無い（resume を妨げない）"

# 3. state.pt の有無と鮮度
[ -f "${OUT}/state.pt" ] || fail "${OUT}/state.pt が無い。--resume は fresh start になる"
echo "ok : state.pt あり ($(du -h "${OUT}/state.pt" | cut -f1), 更新 $(date -r "${OUT}/state.pt" '+%m/%d %H:%M'))"

# 4. どこまで進んだか（state.pt は 1.7GB あるのでログの最終 epoch 行で見る）
LAST_EP=$(grep -h "Epoch: " "${LOG_DIR}"/*.OU 2>/dev/null \
          | sed 's/.*Epoch: \([0-9]*\),.*/\1/' | sort -n | tail -1)
[ -n "${LAST_EP}" ] || fail "ログから最終 epoch を取れない。${LOG_DIR} を確認すること"
REMAIN=$(( NUM_EPOCH - LAST_EP ))
echo "ok : ログの最終 epoch = ${LAST_EP} / ${NUM_EPOCH}（残り ${REMAIN}）"

if [ "${REMAIN}" -le 0 ]; then
  echo "480 epoch に到達済み。resume は不要。"
  exit 0
fi

# 5. 必要 walltime（実測 6.9分/epoch に 30% の余裕）
NEED_H=$(awk -v r="${REMAIN}" -v m="${MIN_PER_EPOCH}" 'BEGIN{printf "%d", (r*m/60)*1.3 + 1}')
[ "${NEED_H}" -gt 48 ] && NEED_H=48
WALLTIME=$(printf "%02d:00:00" "${NEED_H}")
echo "ok : 必要 walltime の見積り ${WALLTIME}（残り ${REMAIN} epoch x ${MIN_PER_EPOCH}分 + 30%）"

# 6. 崩壊していないかを最後に一応見る（吸収状態から resume しても意味が無い）
LAST_LINE=$(grep -h "Epoch: ${LAST_EP}," "${LOG_DIR}"/*.OU 2>/dev/null | tail -1)
echo "     最終行: ${LAST_LINE#*\[INFO\] }"
case "${LAST_LINE}" in
  *"train_loss: 11.09"*)
    fail "train_loss が ln(65536)=11.0904 に張り付いている＝吸収状態。resume しても回復しない" ;;
esac
echo "ok : 吸収状態ではない"

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
  echo "投入後の確認: qstat | grep 0028_dino"
  echo "resume できたかの確認（ログに出るはず）:"
  echo "  grep 'Resumed from state.pt' ${LOG_DIR}/<新jobid>.opbs.OU"
else
  echo "=== 点検のみ。投入するなら ==="
  echo "  ${QSUB_CMD}"
  echo "  （または $0 --submit）"
fi
