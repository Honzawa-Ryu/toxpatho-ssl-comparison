#!/bin/bash
# =====================================================
# exp 0042 — torch.compile(--compile_blocks)と LARS のまとめ計算の効果を測るプロファイル(debug-g・1 ノード・30 分、手法ごとに 1 本)。
# 学習はしない。scripts/train/profile_throughput.py を本番のバッチで 2 回: (1) compile なし(LARS のまとめ計算・
# 評価 worker 8 の現行コード)、(2) --compile。0041 の本番バッチの値(LARS は旧実装)とも比べる。
#   real が synth の何 % か / compile で synth・real が何 % 伸びるか / first_step_s(初回のコンパイル込みの 1 step)
# 準備と投入:
#   bash experiments/0042_20261013_profile_compile/preflight.sh
#   for m in bt simsiam mae; do qsub -v PROFILE_SPEC=$m experiments/0042_20261013_profile_compile/run_slurm.sh; done   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0042_compile
#PBS -q debug-g
#PBS -W group_list=gd43
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/0042_20261013_profile_compile/
#PBS -e logs/0042_20261013_profile_compile/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0042_20261013_profile_compile"

export SCRATCH_ROOT="/local"
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=disabled
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=0
DATA_SUBDIRS=(
    "ssl_patches"
)

# 先頭のバッチが本番の 1 GPU あたりのバッチ(profiler の内訳はそこだけ取る)。qsub -v PROFILE_SPEC=bt|simsiam|mae で選ぶ。
case "${PROFILE_SPEC:-}" in
  bt)      SPEC="0035_20261011_bt_vitb16_lrbias_quarter_clip1:256" ;;
  simsiam) SPEC="0038_20261012_simsiam_vitb16_step_matched:256" ;;
  mae)     SPEC="0040_20261013_mae_vitb16_lr_half:512" ;;
  *) echo "PROFILE_SPEC must be bt|simsiam|mae (qsub -v PROFILE_SPEC=bt ...)" >&2; exit 2 ;;
esac
RUN_MODE="single"
P="python ${PROJECT_ROOT}/scripts/train/profile_throughput.py --out ${PROJECT_ROOT}/outputs/${EXP_NAME} --steps 40 --exp ${SPEC}"
RUN_COMMAND="${P}; ${P} --compile"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
