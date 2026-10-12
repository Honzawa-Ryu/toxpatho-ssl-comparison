#!/bin/bash
# DINO 0028 (7 epochs) embeddings of the tggate-bench min20 tiles. GPU 1 枚。
# Resumable: per-slide shards are skipped on resubmission.
#SBATCH --job-name=dino0028_tggate_extract
#SBATCH --partition=x-large-andre01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_dino0028_tggate_extract.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
BENCH_PY=/workspace/andre01/honzawa/02-playground/tggate-bench/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$BENCH_PY" -u scripts/evaluate/tggate_bench_dino.py extract --workers 6
