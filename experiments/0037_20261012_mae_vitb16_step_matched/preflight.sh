#!/bin/bash
# exp 0037 (Barlow Twins ViT-B/16) の**初回投入前**の点検。qsub はしない。
#   bash experiments/0037_20261012_mae_vitb16_step_matched/preflight.sh
# 8ノードを確保してから落ちるのは高くつくので、ローカルで一瞬で分かることは先に潰す。
# 2本目以降は resume.sh を使う。
set -euo pipefail

EXP_NAME="0037_20261012_mae_vitb16_step_matched"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIF="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"
fail() { echo "NG: $*" >&2; exit 1; }

echo "=== exp 0037 preflight (本番・初回投入) ==="

# 1. 同じ実験のジョブが無いか
if qstat 2>/dev/null | grep -q "0037_ma"; then
  qstat 2>/dev/null | grep "0037_ma"; fail "0037 のジョブがキュー/実行中"
fi
echo "ok : 実行中の 0037 ジョブは無い"

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

# 7. 実験定義の整合(MAE: 論文 1600 epoch × 313 step ≒ 501k step / global batch 4096)
CODE="$(grep -v '^[[:space:]]*#' "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh")"
for fl in "--ssl_name mae" "--num_epoch 2568" "--warmup_t 64" "--optimizer adamw --beta2 0.95 --lr 2.4e-3 --weight_decay 0.05" \
          "--batch_size 512" "--color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0" \
          "--collapse_early_stop" "--collapse_ignore_uniformity"; do
  grep -q -- "${fl}" <<<"${CODE}" || fail "${fl} が run_slurm.sh の実行部に無い"
done
grep -q -- "--ddp_linear_scale_lr" <<<"${CODE}" && fail "--ddp_linear_scale_lr が付いている(lr は最終値 2.4e-3 を直接渡す。二重スケールになる)"
grep -q "^#PBS -l select=8" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "select=8 でない(batch 4096 にならない)"
grep -q "restore_healthy_weights" "${PROJECT_ROOT}/lib/trainer/loop.py" || fail "loop.py が P2 修正前"
echo "ok : MAE / 2568 epoch(≒論文 1600 epoch の step 数) / warmup 64 / AdamW 2.4e-3 β2 0.95 wd 0.05 / bs 4096 / 色aug なし"

# 9. 停止日までの余裕(1600 epoch ≈ 60.5h + 2本目のキュー待ち)
echo "     Miyabi 停止: 2026-10-28 09:00。$(date '+%m/%d %H:%M') 時点で残り $(( ( $(date -d '2026-10-28 09:00' +%s) - $(date +%s) ) / 3600 )) 時間。必要 ≈ 30〜45h(見積り) + キュー待ち。停止をまたいだら停止明けに resume.sh"

echo
echo "=== 点検通過。投入するなら(⚠️ ユーザー承認後) ==="
echo "  cd ${PROJECT_ROOT} && qsub experiments/${EXP_NAME}/run_slurm.sh"
