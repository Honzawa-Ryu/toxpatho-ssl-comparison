#!/bin/bash
# tggate-bench Exp 0011 (anomaly) / 0002 (ridge) on the DINO 0028 embeddings. CPU のみ。
# Task list: python scripts/evaluate/tggate_bench_dino.py list-tasks (28 tasks).
# Submit after extraction: sbatch --dependency=afterok:<extract_job> this_file
#SBATCH --job-name=dino0028_tggate_eval
#SBATCH --partition=large-andre01
#SBATCH --array=0-27%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a_dino0028_tggate_eval.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
BENCH_PY=/workspace/andre01/honzawa/02-playground/tggate-bench/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$BENCH_PY" -u scripts/evaluate/tggate_bench_dino.py eval --task-index "${SLURM_ARRAY_TASK_ID}"
