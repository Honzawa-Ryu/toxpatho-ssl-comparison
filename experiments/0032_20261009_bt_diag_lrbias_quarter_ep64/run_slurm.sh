#!/bin/bash
# =====================================================
# exp 0032 — 修正案の A/B 診断: ep64 から再開して **--lr_bias を 1/4(0.0384 → 0.0096)** にし、
#            0030/0031 と同じ ep66 step 357 で壊れるかを見る。**完成品ではない**(走り切っても使わない)。
#
# 0031 の結果: 先に跳ねるのは bias/LayerNorm の生勾配グループ(gn_raw, step 324)で、起点のテンソルは
# backbone.blocks.1.norm1.weight(LN ゲイン)。head は無関係。このグループは --lars_exclude_bias_bn で
# LARS 適応も wd も外れ、生の勾配 × lr_bias × momentum で更新される(H2')。
# NaN は 0030/0031 とも ep66 step 357 で決定的に再現したので、同じ state.pt から変数 1 つ(lr_bias)だけ
# 変えて同じ step を越えれば「更新側で抑えられる」ことの直接の証拠になる。
#
# ■ 読み方
#   - step 357 を越えて ep66 を完走し、walltime まで(数 epoch)健全 → lr_bias が効いた。本番 0033 は
#     --lr_bias 0.0096 でゼロから 1600 epoch(途中から変えた本ランは完成品にしない)。
#   - 同じ step で壊れる → 更新側ではなく引き金側(特定バッチ・溜まった状態)。次は --clip_grad か lr。
#   - 別の step で壊れる → 遅延しただけ。lr_bias をさらに下げるか、clip_grad を併用。
#   python3 scripts/analysis/bt_step_report.py outputs/0032_20261009_bt_diag_lrbias_quarter_ep64 --epoch 66 --baseline-epoch 65
#
# ■ ⚠️ --resume_override_lr が必須。optimizer/scheduler の復元で lr_bias は保存時の 0.0384 に黙って戻る
#   (lib/trainer/optim.py:override_lr_after_resume, tests/test_resume_override_lr.py で落とし穴を固定)。
#   起動ログに `--resume_override_lr: base lr per group -> [1.6, 0.0096]` が出ることを必ず確認する。
#
# ■ 論文との関係: 公式は lr_biases 0.0048 × batch/256 = 0.0384 @2048 なので、0029 は公式どおり。
#   0.0096 は**公式から離れる**変更(1/4)。「論文準拠」とは書かない(docs/bt_vitb16_vs_paper.md に追記する)。
#
# ■ 0031 からの変更: --lr_bias 0.0096 / --resume_override_lr / 出力先と名前。他は同一(bf16・step ログ・8ノード・1h)。
# ■ コスト: walltime 1h × 8ノード = 最大 8 ノード時間。NaN なら 5 分で止まる。健全なら 1h 走り切る(≈25 epoch)。
#
# 準備と投入:
#   bash experiments/0032_20261009_bt_diag_lrbias_quarter_ep64/preflight.sh   # state.pt コピーと点検(qsub はしない)
#   qsub experiments/0032_20261009_bt_diag_lrbias_quarter_ep64/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0032_bt_diag
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o logs/0032_20261009_bt_diag_lrbias_quarter_ep64/
#PBS -e logs/0032_20261009_bt_diag_lrbias_quarter_ep64/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0032_20261009_bt_diag_lrbias_quarter_ep64"

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
# Multi-node DDP (8 nodes) — 0029 と同一(batch 256 x 8 = 2048, LARS lr 1.6 / lr_bias 0.0384,
# wd 1.5e-6, 1600 epoch, warmup 16)。**スケジュールを変えないこと**: num_epoch が違うと
# cosine の位置が変わり、ep65 の lr が 1.6 でなくなって再現にならない。
#
# 0029 からの変更(0030/0031 の3点 + lr_bias 1/4 + --resume_override_lr):
#   --bt_step_log                  step 単位の損失・grad_norm・z 統計を bt_steps_ep*.jsonl に記録
#   --collapse_ignore_uniformity   外す(BT の uniformity は実測 -2.9〜-3.4 で健全。有効に戻す)
#   --dir_result                   新しいディレクトリ(0029 の成果物を上書きしない)
# 損失は bf16 のまま(--bt_fp32_head は付けない): まず 0029 の NaN を**そのまま再現**して
# step 単位で観察するため。fp32 にして再現しなくなっても、それは原因の特定にならない。
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note bt_diag_lrbias_quarter_ep64 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name barlowtwins \
    --optimizer lars --lr 1.6 --lr_bias 0.0096 --lars_exclude_bias_bn --resume_override_lr \
    --weight_decay 1.5e-6 \
    --batch_size 256 \
    --proj_dim 8192 --bt_lambda 5e-3 \
    --num_epoch 1600 --warmup_t 16 --lr_min 0.0 \
    --collapse_early_stop --collapse_loss_rebound 2.0 \
    --bt_step_log \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"

