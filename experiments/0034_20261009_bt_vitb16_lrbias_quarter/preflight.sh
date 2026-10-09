#!/bin/bash
# exp 0034 (Barlow Twins ViT-B/16) の**初回投入前**の点検。qsub はしない。
#   bash experiments/0034_20261009_bt_vitb16_lrbias_quarter/preflight.sh
# 8ノードを確保してから落ちるのは高くつくので、ローカルで一瞬で分かることは先に潰す。
# 2本目以降は resume.sh を使う。
set -euo pipefail

EXP_NAME="0034_20261009_bt_vitb16_lrbias_quarter"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIF="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"
fail() { echo "NG: $*" >&2; exit 1; }

echo "=== exp 0034 preflight (本番・初回投入: lr_bias 1/4) ==="

# 1. 同じ実験のジョブが無いか
if qstat 2>/dev/null | grep -q "0034_bt"; then
  qstat 2>/dev/null | grep "0034_bt"; fail "0034 のジョブがキュー/実行中"
fi
echo "ok : 実行中の 0034 ジョブは無い"

# 2. 初回なので出力ディレクトリに state.pt / model_ssl.pt が無いこと
#    (--resume 付きなので、あると fresh start ではなく途中再開/即終了になる)
OUT="${PROJECT_ROOT}/outputs/${EXP_NAME}"
[ ! -f "${OUT}/state.pt" ] || fail "${OUT}/state.pt が既にある。初回投入ではない。resume.sh を使うこと"
[ ! -f "${OUT}/model_ssl.pt" ] || fail "${OUT}/model_ssl.pt が既にある"
echo "ok : 出力ディレクトリはまっさら"

# 3. 入力データ(141GB を 8ノードが local SSD へコピーする)
MM="${PROJECT_ROOT}/data/ssl_patches/patches.memmap"
[ -f "${MM}" ] || fail "${MM} が無い"
[ -f "${PROJECT_ROOT}/data/ssl_patches/index.csv" ] || fail "index.csv が無い"
echo "ok : ssl_patches あり ($(du -h --apparent-size "${MM}" | cut -f1))"

# 4. SIF
[ -f "${SIF}" ] || fail "SIF が無い: ${SIF}"
echo "ok : SIF あり"

# 5. ログ出力先(qsub -o/-e の先が無いとジョブが即失敗する)
mkdir -p "${PROJECT_ROOT}/logs/${EXP_NAME}"
echo "ok : logs/${EXP_NAME} を用意した"

# 6. .venv が学習ジョブを動かせる状態か (resume.sh 項目7 と同じ。詳細は env/CONTAINER.md)
VENV_CFG="${PROJECT_ROOT}/.venv/pyvenv.cfg"
VENV_SP="${PROJECT_ROOT}/.venv/lib/python3.12/site-packages"
VENV_NG="uv に作り直された疑い。bash tools/rebuild_venv.sh --apply で直すこと"
[ -f "${VENV_CFG}" ] || fail ".venv が無い。bash tools/rebuild_venv.sh --apply で作り直すこと"
grep -q "^home = /usr/bin" "${VENV_CFG}" || fail ".venv の土台がコンテナの python ではない。${VENV_NG}"
grep -q "^include-system-site-packages = true" "${VENV_CFG}" || fail ".venv がコンテナ側 torch を見ない。${VENV_NG}"
[ -d "${VENV_SP}/timm" ] || fail ".venv に timm が無い。${VENV_NG}"
[ ! -d "${VENV_SP}/torch" ] || fail ".venv 側に torch がある(コンテナの NGC torch をシャドウする)。${VENV_NG}"
echo "ok : .venv はコンテナ python 土台 + system-site-packages + timm あり、torch は venv 側に無い"

# 7. 実験定義の整合: 1600 epoch x 390 step ≒ 論文 625,000 step / global batch 2048
grep -q -- "--num_epoch 1600" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "num_epoch が 1600 でない"
grep -q -- "--collapse_early_stop" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "--collapse_early_stop が無い"
for fl in "--collapse_ignore_uniformity" "--collapse_loss_rebound 2.0"; do
  grep -q -- "${fl}" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "${fl} が無い(BT 用の崩壊検知設定。無いと誤停止/見逃しの恐れ)"
done
grep -q "^#PBS -l select=8" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "select=8 でない(batch 2048 にならない)"
echo "ok : 1600 epoch / --collapse_early_stop / 8ノード"
# 8. 0029 からの変更点(lr_bias 1/4)と step テレメトリ。--resume_override_lr は**付けない**(ゼロからの本番)
CODE="$(grep -v '^[[:space:]]*#' "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh")"
grep -q -- "--lr 1.6 --lr_bias 0.0096" <<<"${CODE}" || fail "--lr_bias 0.0096 でない(0029 の 0.0384 のままでは ep65 付近で NaN になる)"
grep -q -- "--bt_step_log" <<<"${CODE}" || fail "--bt_step_log が無い"
grep -q -- "--resume_override_lr" <<<"${CODE}" && fail "--resume_override_lr が付いている(診断ラン専用。本番では付けない)"
grep -q "all(p.ndim <= 1 for p in g\['params'\])" "${PROJECT_ROOT}/lib/trainer/optim.py" || fail "optim.py が 0032 のバグ修正前"
grep -q "restore_healthy_weights" "${PROJECT_ROOT}/lib/trainer/loop.py" || fail "loop.py が P2 修正前"
echo "ok : lr_bias 0.0096 / --bt_step_log / override なし / コードは 0033 時点以降"
# 9. 停止日までの余裕(1600 epoch ≈ 60.5h + 2本目のキュー待ち)
echo "     Miyabi 停止: 2026-10-28 09:00。$(date '+%m/%d %H:%M') 時点で残り $(( ( $(date -d '2026-10-28 09:00' +%s) - $(date +%s) ) / 3600 )) 時間。必要 ≈ 61h + キュー待ち ×2"

echo
echo "=== 点検通過。投入するなら(⚠️ ユーザー承認後) ==="
echo "  cd ${PROJECT_ROOT} && qsub experiments/${EXP_NAME}/run_slurm.sh"
