#!/bin/bash
# exp 0049 (Barlow Twins ViT-B/16) の**初回投入前**の点検。qsub はしない。
#   bash experiments/0049_20261016_simsiam_vitb16_proj2048/preflight.sh
# 8ノードを確保してから落ちるのは高くつくので、ローカルで一瞬で分かることは先に潰す。
# 2本目以降は resume.sh を使う。
set -euo pipefail

EXP_NAME="0049_20261016_simsiam_vitb16_proj2048"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SIF="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"
fail() { echo "NG: $*" >&2; exit 1; }

echo "=== exp 0049 preflight (本番・初回投入) ==="

# 1. 同じ実験のジョブが無いか
if qstat 2>/dev/null | grep -q "0049_si"; then
  qstat 2>/dev/null | grep "0049_si"; fail "0049 のジョブがキュー/実行中"
fi
echo "ok : 実行中の 0049 ジョブは無い"

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

# 7. 実験定義の整合(SimSiam: 論文 100 epoch × 2502 step ≒ 250k step / global batch 512)
CODE="$(grep -v '^[[:space:]]*#' "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh")"
for fl in "--ssl_name simsiam" "--num_epoch 160" "--warmup_t 16" "--optimizer sgd --lr 0.05 --fix_pred_lr --weight_decay 1e-4" \
          "--batch_size 256 --simsiam_proj_hidden 2048" "--collapse_early_stop" "--collapse_ignore_uniformity" "--step_log" "--compile_blocks"; do
  grep -q -- "${fl}" <<<"${CODE}" || fail "${fl} が run_slurm.sh の実行部に無い"
done
grep -q -- "--wd_apply_to_bias_norm" <<<"${CODE}" && fail "--wd_apply_to_bias_norm が付いている(ViT では LN ゲインが削られて崩壊する経路)"
grep -q -- "--ddp_linear_scale_lr" <<<"${CODE}" && fail "--ddp_linear_scale_lr が付いている(lr は最終値 0.1 を直接渡す)"
grep -q "^#PBS -l select=2" "${PROJECT_ROOT}/experiments/${EXP_NAME}/run_slurm.sh" || fail "select=2 でない(batch 512 にならない)"
grep -q "restore_healthy_weights" "${PROJECT_ROOT}/lib/trainer/loop.py" || fail "loop.py が P2 修正前"
grep -q "n_comp_90" "${PROJECT_ROOT}/lib/sslmodel/evaluation.py" || fail "次元崩壊の指標(2026-10-15)より前のコード"
echo "ok : SimSiam / 160 epoch / warmup 16 / --optimizer sgd --lr 0.05 --fix_pred_lr --weight_decay 1e-4 / bs 512 / 2ノード / compile"

# 9. 停止日までの余裕(1600 epoch ≈ 60.5h + 2本目のキュー待ち)
echo "     Miyabi 停止: 2026-10-28 09:00。$(date '+%m/%d %H:%M') 時点で残り $(( ( $(date -d '2026-10-28 09:00' +%s) - $(date +%s) ) / 3600 )) 時間。必要 ≈ 24h(外挿) + キュー待ち"

echo
echo "=== 点検通過。投入するなら(⚠️ ユーザー承認後) ==="
echo "  cd ${PROJECT_ROOT} && qsub experiments/${EXP_NAME}/run_slurm.sh"
