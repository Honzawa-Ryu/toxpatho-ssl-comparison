#!/bin/bash
# 学習ジョブ用の .venv を作り直す。
#
#   bash tools/rebuild_venv.sh            # 点検のみ(既定)
#   bash tools/rebuild_venv.sh --apply    # 実際に作り直す
#
# ⚠️ この venv は「コンテナの python3(3.12.3) を土台に、system-site-packages 経由で
# コンテナ側 torch を見る」という構造でなければ学習ジョブが動かない。
# ログインノードの /usr/bin/python3 は 3.9.25 なので、**ログインノードから見ると
# 壊れた venv に見える**。そのため `uv run` / `uv sync` をログインノードの
# プロジェクト直下で叩くと、uv が「使えない環境」と判断して uv 管理の CPython で
# 黙って作り直してしまい、torch も timm も消える(2026-09-19 に実際に起きた。
# job 3398346 が全4ノード ModuleNotFoundError: No module named 'torch' で即死)。
# 詳細は env/CONTAINER.md「⚠️ uv run が .venv を作り直す事故」。
#
# 構成の根拠: job 3382307(ep1〜419 を完走)の wandb 記録
#   wandb/offline-run-20260917_195710-0laflbix/files/requirements.txt (350パッケージ)
# とコンテナ側 pip freeze(217パッケージ)の差分から、venv 固有だったのは25個
# = pyproject の base 依存 + timm だけと確認した。torch/torchvision/wandb は
# すべてコンテナ側から来ている。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF="${SIF_PATH:-/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif}"
UV_BIN="${UV_BIN:-/work/gd43/d43000/.local/bin/uv}"
TIMM_VERSION="${TIMM_VERSION:-1.0.28}"   # job 3382307 が使っていた版
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

fail() { echo "NG: $*" >&2; exit 1; }

echo "=== rebuild_venv preflight ==="
[ -f "${SIF}" ] || fail "SIF が無い: ${SIF}"
echo "ok : SIF   ${SIF}"
[ -x "${UV_BIN}" ] || fail "uv が無い: ${UV_BIN}"
echo "ok : uv    ${UV_BIN} ($(${UV_BIN} --version))"

# 学習ジョブが走っている最中に venv を差し替えると、次の epoch の import で落ちうる
if qstat 2>/dev/null | grep -qE "dino|ssl"; then
  qstat 2>/dev/null | grep -E "dino|ssl" || true
  fail "学習ジョブが走っている。終わってから作り直すこと"
fi
echo "ok : 実行中の学習ジョブは無い"

if [ "${APPLY}" -eq 0 ]; then
  echo
  echo "=== 点検のみ。作り直すなら ==="
  echo "  bash tools/rebuild_venv.sh --apply"
  exit 0
fi

BACKUP="${PROJECT_ROOT}/.venv.bak.$(date +%Y%m%d_%H%M%S)"
if [ -d "${PROJECT_ROOT}/.venv" ]; then
  echo "=== 既存 .venv を退避: ${BACKUP} ==="
  mv "${PROJECT_ROOT}/.venv" "${BACKUP}"
fi

module load apptainer/1.3.5 2>/dev/null || true

echo "=== コンテナ内で作り直す ==="
apptainer exec "${SIF}" bash -c "
set -e
export PATH=\$(dirname ${UV_BIN}):\$PATH
cd ${PROJECT_ROOT}
# UV_PROJECT_ENVIRONMENT を明示して、uv が別の場所を作らないよう固定する
export UV_PROJECT_ENVIRONMENT=${PROJECT_ROOT}/.venv
# --system-site-packages がコンテナ側 torch を見せる唯一の経路
uv venv --python /usr/bin/python3 --system-site-packages ${PROJECT_ROOT}/.venv
source ${PROJECT_ROOT}/.venv/bin/activate
# base 依存のみ。torch extra を入れると torch 2.11/2.14 がコンテナの 2.13.0a0 を
# シャドウしてしまう(env/CONTAINER.md の警告どおり)。--inexact で余計な削除も避ける。
uv sync --active --inexact
# timm だけは base に無いので個別に。--no-deps で torch を引かせない。
uv pip install --no-deps 'timm==${TIMM_VERSION}'
"

echo "=== 検証（本番と同じ経路で import する） ==="
apptainer exec "${SIF}" bash -c "
set -e
source ${PROJECT_ROOT}/.venv/bin/activate
cd ${PROJECT_ROOT}
export PYTHONPATH=${PROJECT_ROOT}:\${PYTHONPATH:-}
python - <<'PY'
import torch, timm, h5py, sklearn, numpy, wandb
print('torch  ', torch.__version__, torch.__file__)
print('timm   ', timm.__version__)
print('h5py   ', h5py.__version__, '/ numpy', numpy.__version__, '/ wandb', wandb.__version__)
assert '/dist-packages/' in torch.__file__, 'torch が venv 側から来ている(コンテナ版でない)'
import lib.sslmodel.sslutils, lib.trainer.entry, lib.trainer.loop
print('lib.* の import OK')
PY
python -m torch.distributed.run --help >/dev/null && echo 'torch.distributed.run OK'
"

echo
echo "=== 完了 ==="
echo "退避した旧 venv: ${BACKUP}（不要になったら rm -rf すること）"
