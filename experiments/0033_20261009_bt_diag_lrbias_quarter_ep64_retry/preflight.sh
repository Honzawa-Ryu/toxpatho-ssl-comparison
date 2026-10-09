#!/bin/bash
# exp 0033 (診断ラン)の投入前の準備と点検。qsub はしない。
#   bash experiments/0033_20261009_bt_diag_lrbias_quarter_ep64_retry/preflight.sh
# 0029 の ep64 の state.pt を新ディレクトリへコピーする(初回のみ。既にあれば再コピーしない)。
set -euo pipefail

EXP_NAME="0033_20261009_bt_diag_lrbias_quarter_ep64_retry"
SRC_EXP="0029_20260930_bt_vitb16_step_matched"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PROJECT_ROOT}/outputs/${EXP_NAME}"
SRC="${PROJECT_ROOT}/outputs/${SRC_EXP}"
SIF="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"
fail() { echo "NG: $*" >&2; exit 1; }

echo "=== exp 0033 preflight (A/B 診断: lr_bias 1/4) ==="

if qstat 2>/dev/null | grep -qE "003[0-3]_bt|0029_bt"; then
  qstat 2>/dev/null | grep -E "003[0-3]_bt|0029_bt"; fail "0029〜0033 のジョブがキュー/実行中"
fi
echo "ok : 実行中の 0029〜0033 ジョブは無い"

# 1. 元の state.pt(ep64)。⚠️ 0029 の実行を再開しないこと(0029 の成果物は触らない)
[ -f "${SRC}/state.pt" ] || fail "${SRC}/state.pt が無い"
echo "ok : 元の state.pt あり ($(du -h "${SRC}/state.pt" | cut -f1), 更新 $(date -r "${SRC}/state.pt" '+%m/%d %H:%M'))"
LAST_EP=$(grep -ah "Epoch: " "${PROJECT_ROOT}/logs/${SRC_EXP}"/*.OU | sed 's/.*Epoch: \([0-9]*\),.*/\1/' | sort -n | uniq | tail -2 | head -1)
echo "     ログ上の最後の健全な epoch = ${LAST_EP}(期待 64。state.pt はこの epoch の末に保存されている)"
[ "${LAST_EP}" = "64" ] || fail "最後の健全な epoch が 64 でない(${LAST_EP})。state.pt の中身を確かめること"

# 2. 新ディレクトリへコピー(0029 の checkpoint.pt/model_ssl.pt には触れない)
mkdir -p "${OUT}" "${PROJECT_ROOT}/logs/${EXP_NAME}"
if [ -f "${OUT}/state.pt" ]; then
  echo "ok : ${OUT}/state.pt は既にある(再コピーしない)"
else
  cp "${SRC}/state.pt" "${OUT}/state.pt"
  echo "ok : state.pt を ${OUT} へコピーした"
fi
cmp -s "${SRC}/state.pt" "${OUT}/state.pt" || fail "コピー後の state.pt が元と一致しない"
[ ! -f "${OUT}/model_ssl.pt" ] || fail "${OUT}/model_ssl.pt がある(entry.py が即終了する)"
echo "ok : コピーは元と一致、model_ssl.pt は無い"

# 3. .venv (resume.sh 項目7 と同じ)
VENV_CFG="${PROJECT_ROOT}/.venv/pyvenv.cfg"
VENV_SP="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages"
VENV_NG="uv に作り直された疑い。bash tools/rebuild_venv.sh --apply で直すこと"
[ -f "${VENV_CFG}" ] || fail ".venv が無い。${VENV_NG}"
grep -q "^home = /usr/bin" "${VENV_CFG}" || fail ".venv の土台がコンテナの python ではない。${VENV_NG}"
grep -q "^include-system-site-packages = true" "${VENV_CFG}" || fail ".venv がコンテナ側 torch を見ない。${VENV_NG}"
[ -d "${VENV_SP}/timm" ] || fail ".venv に timm が無い。${VENV_NG}"
[ ! -d "${VENV_SP}/torch" ] || fail ".venv 側に torch がある。${VENV_NG}"
echo "ok : .venv は健全"

[ -f "${SIF}" ] || fail "SIF が無い"
[ -f "${PROJECT_ROOT}/data/ssl_patches/patches.memmap" ] || fail "patches.memmap が無い"
echo "ok : SIF と ssl_patches あり"

# 4. 定義の整合: 0029 と同じスケジュール(これが違うと ep65 の lr が変わり再現にならない)
#    ⚠️ コメント行は除いて判定する(ヘッダの説明文に同じフラグ名が出てくるため誤検出する)
R="${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh"
CODE="$(grep -v '^[[:space:]]*#' "${R}")"
for fl in "--num_epoch 1600" "--warmup_t 16" "--lr 1.6 --lr_bias 0.0096" "--resume_override_lr" "--bt_step_log" "--collapse_early_stop"; do
  grep -q -- "${fl}" <<<"${CODE}" || fail "${fl} が run_slurm.sh の実行部に無い"
done
grep -q -- "--collapse_ignore_uniformity" <<<"${CODE}" && fail "--collapse_ignore_uniformity が残っている(診断ランでは有効に戻す)"
grep -q -- "--bt_fp32_head" <<<"${CODE}" && fail "--bt_fp32_head が付いている(診断ランは bf16 のまま再現を見る)"
grep -q "^#PBS -l select=8" "${R}" || fail "select=8 でない(batch 2048 にならない)"
echo "ok : 0029 と同じスケジュール / lr_bias 0.0096 + --resume_override_lr / --bt_step_log / bf16 のまま / 8ノード"
grep -q "bt_probe.record(grad_params, per_param_norms)" "${PROJECT_ROOT}/lib/trainer/loop.py" || fail "loop.py にグループ別 grad_norm の配線が無い(古いコードで投入しようとしている)"
echo "ok : loop.py にグループ別 grad_norm(GradGroupProbe)の配線あり"
grep -q "override_lr_after_resume(args, optimizer, scheduler, start_epoch" "${PROJECT_ROOT}/lib/trainer/entry.py" || fail "entry.py に --resume_override_lr の配線が無い(lr_bias が 0.0384 に戻って A/B にならない)"
echo "ok : entry.py に --resume_override_lr の配線あり"
grep -q "all(p.ndim <= 1 for p in g\['params'\])" "${PROJECT_ROOT}/lib/trainer/optim.py" || fail "optim.py の override が params の形で bias グループを見分けていない(0032 のバグが残っている)"
echo "ok : override は params の形で bias グループを見分ける(0032 のバグは修正済み)"

echo
echo "=== 点検通過。投入するなら(⚠️ ユーザー承認後) ==="
echo "  cd ${PROJECT_ROOT} && qsub experiments/${EXP_NAME}/run_slurm.sh"
echo "起動後の確認: grep -ah \"resume_override_lr: base lr\" logs/${EXP_NAME}/*.OU  → [1.6, 0.0096] (roles weights, bias) であること。[1.6, 1.6] なら即 qdel"
