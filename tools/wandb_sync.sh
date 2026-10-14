#!/bin/bash
# offline で記録した WandB の run をアップロードする(ログインノードで実行。計算ノードは外に出られない)。
#   bash tools/wandb_sync.sh                 # 未同期の run をすべて(走行中の run は除く)
#   bash tools/wandb_sync.sh wandb/offline-run-XXXX ...   # 指定した run だけ
# wandb はログインノードの PATH に無く、コンテナ内(/usr/local/bin/wandb)にある。
# 走行中の run は、その時点までしか上がらないので、ジョブの終了後に同期する(--no-include-synced で二重送信しない)。
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIF="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"
cd "${PROJECT_ROOT}"
if [ $# -gt 0 ]; then
  RUNS="$*"
else
  # 走行中のジョブの run を除く: 最終更新が 10 分以内の .wandb は書き込み中とみなす
  RUNS=$(for d in wandb/offline-run-*; do
           f=$(ls "$d"/*.wandb 2>/dev/null | head -1)
           [ -n "$f" ] && [ -z "$(find "$f" -mmin -10)" ] && echo "$d"
         done | tr '\n' ' ')
fi
[ -n "${RUNS}" ] || { echo "同期する run が無い"; exit 0; }
module load apptainer/1.3.5 2>/dev/null || true
apptainer exec --bind /work "${SIF}" bash -c "cd ${PROJECT_ROOT} && /usr/local/bin/wandb sync --no-include-synced ${RUNS}"
