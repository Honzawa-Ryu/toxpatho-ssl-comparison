#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 8ノード(GH200 120GB×8)マルチノードDDP。batch 512/GPU × 8 = 4096(論文・公式と同じ global batch)。
#
# 【MAE ViT-B/16 — 0040(lr 半減)に per-param の clip_grad 1.0 を足した本番】
#
# ■ 0040 からの変更は **--clip_grad 1.0**(パラメータ毎の L2 ノルムでクリップ)の 1 点だけ。
#   0040 は ep307 から不安定化し ep570 に崩壊検知で停止(0037 の ep251 より約 50 epoch 遅れただけ)。前兆は同じで、
#   patch_embed の出力 RMS が ep100 の 2 → ep300 の 48 と膨らみ、ep300 で bias/LN の勾配(gn_raw)の中央値が 15
#   (ep200 は 0.007)に跳ね、ep305〜307 で grad_norm が最大 1,250。lr を下げるだけでは止まらない。
#   BT 0035 では clip 1.0 が毎 step bias/LN の更新幅を抑えて、0029/0034/0039 が壊れた位置を越えている(ep578 で健全)。
#   健全な MAE の勾配は全体で 0.02〜0.04(テンソル単位ではさらに小さい)なので、clip 1.0 は普段は掛からず、スパイクだけを
#   切り詰める。AdamW は勾配の大きさを 2 次モーメントで正規化するが、スパイクは m / v の推定を引きずるので clip で止める。
#   **公式 MAE の事前学習に clip は無い**(逸脱)。「論文準拠」と書かない。
#
# 以下は 0040 の説明をそのまま引き継ぐ。
# 【MAE ViT-B/16 — 0037 から peak lr を半分にした本番】
#
# ■ 0037 からの変更は **--lr 2.4e-3 → 1.2e-3**(blr 1.5e-4 → 0.75e-4 相当)の 1 点だけ。
#   0037 は ep250 まで健全(val 0.263)だったが、ep251 で grad_norm が 0.3→4.0 と跳ね、損失が戻らず
#   eff_rank 150→4 で ep319 に崩壊検知で停止。起点は最初の block の LayerNorm(blocks.0.norm1)の勾配スパイクで、
#   その前から patch_embed の出力 RMS が 0.8(ep10)→61(ep249)と単調に膨らみ、LN ゲインが 0.29 まで縮んでいた。
#   DINO も同型の不安定(0026)を lr 半減(0028)で越えた前例に合わせる。AdamW は勾配の大きさを 2 次モーメントで
#   正規化するので clip より lr のほうが更新幅に直接効く。**公式(blr 1.5e-4)からの逸脱**。「論文準拠」と書かない。
#   スケジュール・epoch 数(2568 = 論文 1600 ep の step 数)・batch・wd・mask・aug は 0037 と同一。
#
# 以下は 0037 の説明をそのまま引き継ぐ。
#
# ■ step 数の合わせ方(DINO 0028 / BT 0029 と同じ考え方)
#   MAE 論文(He et al. 2022)の ViT-B は「本文の最終結果 1600 epoch / アブレーションと公式 PRETRAIN.md の
#   推奨コマンド 800 epoch」と 2 つある。**ここは本文の最終結果の 1600 epoch に合わせる**(ユーザー決定 2026-10-12:
#   resume できるので目標は元論文にそろえる。Miyabi 停止をまたいでも停止明けに再開すればよい)。
#   「論文準拠」と一語で書かない(公式 PRETRAIN.md の推奨は 800)。
#     論文: 1,281,167 枚 / 4096 = 312.8 → 313 step/epoch × 1600 = 500,800 step
#     本データ: train 800,000 枚 / 4096 = 195 step/epoch(drop_last) → 500,800 / 195 = **2568 epoch**
#     warmup: 論文 40 epoch × 313 = 12,520 step → **64 epoch**(12,480 step)
#
# ■ 公式(main_pretrain.py / PRETRAIN.md)にそろえた点
#   AdamW β=(0.9, 0.95), wd 0.05(bias/LN は除外 = 公式 param_groups_weight_decay と同じ),
#   lr は公式なら blr 1.5e-4 × 4096/256 = 2.4e-3(**本ランは半分の 1.2e-3**), cosine → 0, mask ratio 0.75, norm_pix_loss,
#   decoder 512-d × 8 blocks, augmentation は RandomResizedCrop(0.2,1.0) + flip のみ(色・blur・solarize 0)。
# ■ 逸脱
#   1. 学習データ: ImageNet ではなく TG-GATEs パッチ 800k 枚(スライド内のパッチは互いに似る)。
#   2. 精度: 公式は AMP(fp16 + GradScaler)、ここは bf16 autocast(他手法と同じ)。
#   3. 崩壊検知を付ける: NaN 即停止 + eff_rank<5 + 損失の跳ね返り 2.0。uniformity 判定は外す
#      (MAE の特徴は一様に散る目的で学習しないので、健全でも 0 付近になりうる。誤停止を避ける)。
#   4. augmentation: 公式は RandomResizedCrop(0.2,1.0) + flip のみ。ここは全手法共通の ssl_transform を使うため、
#      **0〜180 度の連続回転(毎回)と RandomGrayscale 0.05 が残る**(色・blur・solarize は 0)。回転は病理パッチの
#      向きの不変性のために全手法で入れている(手法間で揃える)。非 90 度回転は四隅が黒になり、そのパッチの
#      再構成は易しくなる(norm_pix_loss で分散 0 → 目標 0)。
#   5. --step_log: step ごとの損失・grad_norm・エンコーダ/デコーダ別 grad_norm・上位テンソルを steps_ep*.jsonl に記録
#      (学習には影響しない)。
#
# ■ コスト(**見積り。実測なし**): andre01 の旧 MAE ViT-B(bs 256, 3.1 it/s ≈ 800 枚/s/GPU)から GH200 で
#   2〜3 倍と仮定 → 1600〜2400 枚/s/GPU × 8 = 1.3〜1.9 万枚/s → 1 epoch(800k)≈ 0.7〜1.0 分、2568 epoch ≈ 30〜45h
#   ≈ 240〜360 ノード時間(+ データローダ律速なら延びる)。48h 枠で切れたら resume.sh で続きを投入する
#   (Miyabi 停止をまたぐ場合も停止明けに resume.sh)。1 本目の elapsed_time で resume.sh の MIN_PER_EPOCH を更新すること。
#   VRAM: A6000 で bs 512 が 28.6GB(0013 vram_probe)、GH200 120GB では十分余裕。
#
# 投入前提:
#   bash experiments/0043_20261014_mae_vitb16_lr_half_clip1/preflight.sh   # 点検のみ
#   qsub experiments/0043_20261014_mae_vitb16_lr_half_clip1/run_slurm.sh   # ⚠️ ユーザー承認後
#
# ■ 高速化(2026-10-13, exp 0041/0042): --compile_blocks(backbone の各 block の forward だけ torch.compile。
#   学習の計算は同じで、丸めが少し変わる程度)。1 GPU の学習ループで BT +20%・MAE +8%。評価用 DataLoader の worker 8・
#   LARS のまとめ計算は既定で入る(コード側)。
# =====================================================
#PBS -N 0043_mae_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0043_20261014_mae_vitb16_lr_half_clip1/
#PBS -e logs/0043_20261014_mae_vitb16_lr_half_clip1/


module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0043_20261014_mae_vitb16_lr_half_clip1"

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
# Multi-node DDP (8 nodes) — batch 512/GPU × 8 = 4096
# MAE の損失はサンプルごと(バッチ統計に依存しない)なので、この分割は単一の 4096 バッチと等価。
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note mae_vitb16_lr_half_clip1 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name mae \
    --optimizer adamw --beta2 0.95 --lr 1.2e-3 --weight_decay 0.05 \
    --batch_size 512 \
    --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
    --num_epoch 2568 --warmup_t 64 --lr_min 0.0 \
    --collapse_early_stop --collapse_ignore_uniformity --collapse_loss_rebound 2.0 \
    --step_log --compile_blocks --clip_grad 1.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
