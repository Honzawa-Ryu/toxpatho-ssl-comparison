#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。
#
# 【0025 の再試行】変更点は --clip_grad 3.0 -> 0.3 の1点のみ。
#
# 0025 の結果(2026-08-31, job 3271936):
#   - weight decay の修正は効いた。ln_gain(backbone LayerNormゲイン平均)は
#     ep20 で 0.9955 を維持(旧コードの0023は同時点で 0.3731)。head の重み膨張も無し。
#     → 恒久崩壊の主因だった「bias/LayerNormゲインへのwd」は解消済み。
#   - しかし ep11->12 で別の崩壊。train_loss が ln(65536)=11.0904 に張り付き
#     grad_norm 1e-4、feat_std 0.018、uniformity 0.0000。このとき ln_gain は
#     0.9692 で固定されたままなので、weight decay 起因ではない。
#   - loss は ep1-11 で 10.79/8.74/10.81/10.65/10.03/10.62/10.11/9.47/9.08/10.21/10.42
#     と振動するだけで、一度も下降トレンドに乗らなかった。
#   - 新しい崩壊検知(train_loss >= ln(out_dim)*0.999 / uniformity > -0.05)が
#     ep19 で abort。旧 eff_rank 基準(ep20時点で21.24)では発火せず48h回し切っていた。
#
# 本実験の変更点と根拠:
#   --clip_grad 0.3
#     公式 main_dino.py の help に「Clipping with norm .3 ~ 1.0 can help
#     optimization for larger ViT architectures.」とある。0025 で使った 3.0 は
#     argparse の既定値だが、ViT-B のような大きいモデルには 0.3〜1.0 が推奨。
#     0025 の実測 grad_norm は 0.25〜3.15 だったので、3.0 のクリップは
#     ほぼ一度も発火していなかった(=実質クリッピング無しで走っていた)。
#     0.3 なら実際に効く。
#
# それ以外は 0025 と完全に同一(out_dim 65536 / drop_path 0.1 /
# momentum 0.996->1.0 / teacher_temp 0.04固定 / 480 epoch / global batch 1024)。
# 1点だけ変えているので、崩壊するかどうかで clip_grad の寄与を切り分けられる。
#
# まだ試していない候補(本実験が駄目なら次に試す):
#   - --dino_freeze_last_layer 3
#     公式 help「Try increasing this value if the loss does not decrease.」
#     まさに 0025 の症状。ただし現状CLI未公開なので引数追加が必要。
#   - out_dim を 8192 に戻す(0023 は 8192 で loss 1.72 まで下降した実績がある。
#     公式 help は「複雑で大規模なデータセットでは 65k が良い」と条件付き)
#   - AMPを切る / 損失計算だけfp32
#     公式 help「loss が不安定なとき、大きいViTを使うときは mixed precision を
#     切ることを推奨」。bf16 は fp16 より仮数部が3bit少なく、centering の
#     (t - center) のような近い値どうしの差には不利。
#
# 判定の見方:
#   - 崩壊すれば --collapse_early_stop が 2.5h 程度で abort する(0025 の実績)。
#     48h 枠で投げておいて問題ない。
#   - 起動直後に stdout の
#     "weight decay 0.04->0.4: decayed 55 tensors / exempt (bias & ndim<=1) 102 tensors"
#     を確認すること。exempt が 0 なら wd の除外が効いていない。
#   - 崩壊の loss 基準は out_dim 65536 なので ln(65536) = 11.0904。
#   - 毎epochの ln_gain が 1.0 付近を維持していること(下がり続けたら wd の再発)。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0026_20260901_dino_clipgrad03`
# してから
#   `qsub experiments/0026_20260901_dino_clipgrad03/run_slurm.sh`
# =====================================================
#PBS -N 0026_dino_clipgrad03
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0026_20260901_dino_clipgrad03/
#PBS -e logs/0026_20260901_dino_clipgrad03/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0026_20260901_dino_clipgrad03"

export SCRATCH_ROOT="/local"
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage
# =====================================================

USE_LOCAL_SSD_INPUT=1
USE_LOCAL_SSD_OUTPUT=1
DATA_SUBDIRS=(
    "ssl_patches"
)

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/scripts/train/train_tggate.py"

# =====================================================
# Multi-node DDP (4 nodes)
#
# lr について: --ddp_linear_scale_lr は入力--lrをworld_size倍するだけの実装なので、
# 論文のlinear scaling rule (lr = 0.0005 × global_batch/256 = 0.0005 × 1024/256
# = 2e-3) を満たすには入力を 2e-3 / 4 = 5e-4 にする必要がある。
# global batch = 256 × 4ノード = 1024 で、公式の ViT-B/16 レシピ
# (2ノード×8GPU×batch_size_per_gpu 64 = 1024) と同じ。
#
# teacher温度は指定しない = 0.04固定 = 公式既定。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note dino_clipgrad03 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --clip_grad 0.3 \
    --batch_size 256 \
    --n_global_crops 2 --n_local_crops 8 --local_crop_size 96 \
    --num_epoch 480 --warmup_t 10 --lr_min 1e-6 \
    --dino_out_dim 65536 --dino_drop_path 0.1 \
    --dino_momentum_start 0.996 --dino_momentum_end 1.0 \
    --collapse_early_stop \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
