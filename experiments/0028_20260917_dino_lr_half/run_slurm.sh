#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。
#
# 【0026 の再試行 / HANDOFF タスク2-A】
# 変更点は peak lr を 2e-3 -> 1e-3 にする1点のみ
# (--lr 5e-4 -> 2.5e-4、--ddp_linear_scale_lr で world_size=4 倍される)。
#
# 0026 の結果(2026-09-02, job 3280970): ep89 で崩壊、ep94 で abort。
#   ep 21: loss 2.7608  lr 2.00e-3  gn 0.88  ln_gain 0.800
#   ep 50: loss 2.0612  lr 1.97e-3  gn 0.84  ln_gain 0.657
#   ep 66: loss 5.3178  lr 1.94e-3  gn 1.00  ln_gain 0.633  <- 1回目の暴発、自力回復
#   ep 75: loss 1.6647  lr 1.91e-3  gn 1.78  ln_gain 0.625
#   ep 80: loss 1.5993  lr 1.90e-3  gn 1.09  ln_gain 0.615
#   ep 85: loss 1.6132  lr 1.89e-3  gn 5.67  ln_gain 0.611
#   ep 88: loss 1.6051  lr 1.88e-3  gn 5.98  ln_gain 0.610
#   ep 89: loss 6.4304  lr 1.87e-3  gn 3.78  ln_gain 0.605  <- 破綻
#   ep 90: loss 11.0904 lr 1.87e-3  gn 0.0002                <- 吸収状態(回復不能)
#
# 本実験の変更点と根拠:
#   --lr 2.5e-4 (実効 peak 1e-3)
#
#   (1) loss は ep75-88 で 1.66->1.60 とほぼ床に張り付いているのに、同区間で
#       クリップ前 grad_norm が 1.0 -> 6.0 へ単調増大している。学習が進まないのに
#       勾配だけ大きくなる = より鋭い領域で跳ね回っている状態で、lr 高原での
#       最適化不安定の形。ln_gain は 0.61 で平坦なので weight decay 起因ではない
#       (0026 で wd の param group 修正が効いていることの裏づけでもある)。
#
#   (2) このランは実質「定常 lr 2e-3」だった。CosineLRScheduler(t_initial=480,
#       warmup_t=10) では ep89 は cosine の (89-10)/470 = 16% 地点にすぎず、
#       lr は 2.00e-3 -> 1.87e-3 と 6.5% しか落ちていない。lr アニールによる
#       後半の安定化を一度も受けていない。
#
#   (3) クリッピングはもう下げ代が無い。--clip_grad 0.3 は公式 help の推奨レンジ
#       (.3〜1.0) の下限。勾配テンソルは student の約157本なので、全体ノルム
#       0.88(ep21) は1本あたり平均 0.07 でほぼ未発火だったが、5.98(ep88) は
#       1本あたり平均 0.48 で常時発火していた。それでも破綻した。
#       加えて AdamW の更新量は lr*m/(sqrt(v)+eps) で勾配の定数倍に不変
#       (1座標あたり実質 ~lr で頭打ち)なので、勾配側を抑えても step size は
#       lr でしか変わらない。クリップが飽和した後に残るつまみは lr だけ。
#
#   (4) 2e-3 の出自は公式の linear scaling (5e-4 × 1024/256)。これが成り立つのは
#       critical batch size 以下という条件つきで、800枚のWSI由来80万パッチという
#       冗長なデータでは同一スライド内パッチの勾配が強く相関し、critical batch
#       size は ImageNet-1k より小さい。batch 1024 まで線形に引き上げた lr は
#       過大になる。論文値そのものではなく「論文の外挿式をドメイン外に当てた箇所」
#       を動かしている点が、この逸脱の正当化になる。
#
#   同じ形の破綻は out_dim 8192 の 0023 でも起きている(loss 1.72 まで下げた直後
#   ep42 に突然崩壊、lr 一定・grad_norm スパイク無し)。out_dim を変えても同じ
#   lr 高原で再現している。
#
# それ以外は 0026 と完全に同一(clip_grad 0.3 / out_dim 65536 / drop_path 0.1 /
# momentum 0.996->1.0 / teacher_temp 0.04固定 / 480 epoch / warmup 10 /
# global batch 1024)。1点だけ変えているので lr の寄与を切り分けられる。
#
# ⚠️ 論文準拠について(CLAUDE.md「用語の注意」):
#   本ランは公式コード既定(lr = 5e-4 × global_batch/256)からの**意図的な逸脱**。
#   論文本文には clip_grad の記述が無く、lr は linear scaling 式で書かれている。
#   論文に書くときは「lr を論文の linear scaling 値の半分にした。安定性のため」と
#   明示すること。0026 と同じく「論文から外したから伸びた」系の変更にあたる。
#
# 判定の見方(投入前に固定しておく):
#   - 本命の指標は **クリップ前 grad_norm のランプが出るか**。0026 は ep55 付近から
#     ノイズが増え始め ep75 以降で単調増大した。ep100 前後まで grad_norm が
#     1.0〜1.5 圏内に留まっていれば lr 仮説は支持される。
#   - loss の下降は 0026 より遅くなるのが当然(lr 半分)。ep50 時点で 0026 の
#     2.06 に届かないこと自体は失敗ではない。見るのは「床に達してから
#     grad_norm が伸び始めるか」。
#   - 崩壊すれば --collapse_early_stop が abort する(train_loss が
#     ln(65536)=11.0904 に張り付いたら吸収状態=回復しない)。
#   - 起動直後に stdout の
#     "weight decay 0.04->0.4: decayed 55 tensors / exempt (bias & ndim<=1) 102 tensors"
#     を確認すること。exempt が 0 なら wd の除外が効いていない。
#   - "--ddp_linear_scale_lr: lr scaled by world_size=4 -> 0.001" が出ること
#     (0.002 なら --lr を直し忘れている)。
#   - 毎epochの ln_gain が 1.0 付近から緩やかに下がるのは勾配由来で正常
#     (0026 も 0.61 まで下がりつつ健全だった)。単調に 0 へ向かうなら wd の再発。
#
# walltime について:
#   480 epoch ≈ 51.6h (6.45分/epoch 実測)で 48h 枠を超える。--resume 前提で
#   2本に分ける。継続投入時は outputs/ に model_ssl.pt が書かれていると
#   「training complete, nothing to resume」で即終了するので、退避してから再投入。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0028_20260917_dino_lr_half`
# してから
#   `qsub experiments/0028_20260917_dino_lr_half/run_slurm.sh`
#
# 関連: docs/HANDOFF_dino_next.md タスク2-A / PROJECT_STATUS.md「🦕 DINO崩壊調査」
#   タスク2-B (--dino_fp32_head) は実装済み・未投入。精度は丸めノイズがゼロ平均で
#   30 epoch の単調ランプを説明しないため、本ランとは別軸の対抗馬として扱う。
# =====================================================
#PBS -N 0028_dino_lr_half
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0028_20260917_dino_lr_half/
#PBS -e logs/0028_20260917_dino_lr_half/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0028_20260917_dino_lr_half"

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
# peak lr 1e-3 を得るには入力を 1e-3 / 4 = 2.5e-4 にする。
# 公式の linear scaling rule (lr = 0.0005 × global_batch/256 = 2e-3) の半分で、
# これが本ランの唯一の変更点(0026 は --lr 5e-4 = 実効 2e-3)。
#
# global batch = 256 × 4ノード = 1024 で、公式の ViT-B/16 レシピ
# (2ノード×8GPU×batch_size_per_gpu 64 = 1024) と同じ。lr だけ半分。
#
# teacher温度は指定しない = 0.04固定 = 公式コード既定(論文本文は 0.04->0.07 の
# warmup を書いているが、0024 でそれを試して ep9 に恒久崩壊している)。
# =====================================================

NNODES=4
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note dino_lr_half --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 2.5e-4 --ddp_linear_scale_lr \
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
