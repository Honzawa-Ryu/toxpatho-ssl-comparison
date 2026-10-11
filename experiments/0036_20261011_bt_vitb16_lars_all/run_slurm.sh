#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 8ノード(GH200 120GB×8)マルチノードDDP。batch 256/GPU × 8 = 2048。
#
# 【Barlow Twins ViT-B/16 — 0035 と並列の対抗馬。bias/LayerNorm にも LARS 適応を掛ける(--lars_exclude_bias_bn を外す)】
#
# ■ 位置づけ: 0035(lr_bias 0.0096 + clip_grad 1.0)と**同時に**走らせる別解。0029/0031/0034 の発散は、
#   bias/LN グループが LARS 適応から外れて「生の勾配 × lr_bias」で更新されることによる正のフィードバックだった
#   (PROJECT_STATUS「💥 0034」)。0035 はその更新幅を clip で上から抑える。0036 は**機構そのものを外す**:
#   bias/LN にも trust ratio を掛け、update = lr·eta·||p||·g/||g|| と勾配の大きさに依らなくする。
#
# ■ 0029 からの変更点(「論文準拠」と一語で書かないこと)
#   - **--lars_exclude_bias_bn を外す**: bias/LN(ndim<=1)も LARS 適応の対象。weight decay は従来どおり外れる
#     (lib/trainer/model.py:_wd_groups が ndim<=1 を wd=0・weight_decay_filter=True の別グループにする。
#     lars_adaptation_filter は False のまま = 適応あり)。lr は重みと同じ 1.6(--lr_bias は使われないので指定しない)。
#     公式 BT は bias/BN を LARS 適応から外しているので、**公式からの逸脱**(docs/bt_vitb16_vs_paper.md)。
#   - --bt_step_log(テレメトリ。学習に影響しない)。
#   - clip_grad は付けない(LARS 適応下では勾配の大きさに依らないので効かない。0035 との差を 1 軸に保つ)。
#   それ以外(スケジュール・batch・wd・λ・ヘッド・崩壊検知)は 0029 と同一。コード変更なし。
#
# ■ 0035 と両方完走した場合の選び方(結果を見る前に固定する。下流性能で選ばない):
#   完走したほう。両方完走なら**公式からの逸脱が小さい 0035 を本命**、0036 は感度分析として残す。
#
# ■ 実測コスト(0033): 約 2.27 分/epoch @8ノード(val_loss と 5 epoch ごとの eval/snapshot 込み) + 起動時コピー約 7 分
#   1600 epoch ≈ 60.5 h ≈ 485 ノード時間。48h 枠では約 1,260 epoch で切れるので **resume.sh で 2 本目(約 13h)** が要る。
#   ⚠️ Miyabi は 2026-10-28 09:00 停止。2 本目のキュー待ちを含めて間に合うよう、早めに投入する。
#
# ■ 論文/公式コードに対する立ち位置（「論文準拠」と一語で書かないこと）
#   そろえた点: optimizer=LARS, lr 0.2/0.0048 × bs/256 (=1.6/0.0384 @2048),
#     wd 1.5e-6 (bias/BN は wd も LARS 適応も除外), λ=5e-3, 投影ヘッド 8192-8192-8192,
#     batch 2048, 総 step 数, warmup の step 数。
#   逸脱した点:
#     1. backbone が ViT-B/16 (論文は ResNet-50)。LARS が ViT で安定かは未検証。
#     2. epoch 数: 論文 1000 epoch × 625 step = 625,000 step。本データは train 800,000 パッチ
#        (1M 中 200k=val fold)なので 390 step/epoch。DINO(0017/0028)が論文の step 数に
#        そろえて 480 epoch にしたのと同じ考え方で、**1600 epoch = 624,000 step**(論文の 99.8%)。
#        ⚠️ サンプル総数は論文の 1.28B に対し 1.28B(=0.8M×1600)でほぼ一致するが、
#        画像の多様性は ImageNet より大幅に低い(1スライド内のパッチは互いに似る)。
#     3. warmup: 論文 10 epoch = 6,250 step → 16 epoch(=6,240 step)。
#     4. lr 終端: 論文 end_lr_ratio 0.001、ここは --lr_min 0.0 (差はピーク比 0.1% 以下)。
#     5. 精度: 論文は AMP(fp16)、ここは bf16 autocast(損失も bf16 で返る)。
#
# ■ 崩壊したら止める（Goal.yaml 方針3, 2026-09-30 / 2026-10-01 BT 用に調整）
#   --collapse_early_stop に加えて BT 用に次の設定を入れている:
#   (1) NaN/inf の train_loss は即停止(毎 epoch, patience なし)。BT は出力が定数に潰れると
#       z.std(0)->0 で損失が NaN になる(CPU で再現)。この実装の BT 損失は std に eps を足さない
#       ので NaN になる(公式コードは BatchNorm1d の eps で有限のまま、は未照合)。
#   (2) --collapse_ignore_uniformity: uniformity 判定を無効化。BT の損失はバッチ平均を引いてから
#       相関を取るため投影出力の共通オフセットが自由で、健全でも uniformity ≈ 0 になりうる。
#       BT は out_dim を持たず補助指標が常時有効になるので、有効のままだと ep10 の最初の判定で
#       健全なランを誤停止しうる(DINO の健全時でさえ -0.006。tests/test_collapse_guards.py で再現)。
#   (3) --collapse_loss_rebound 2.0: epoch>=20 以降の損失最小値の2倍超が3 epoch続いたら発散として停止
#       (NaN にならない有限の発散への備え)。⚠️ 比率・patience は較正していない。止める側に倒してある
#       (誤停止の被害は数ノード時間、見逃すと数百ノード時間)。初回ランの損失曲線を見て見直すこと。
#   (4) eff_rank < 5 が rank_monitor_interval=5 x patience=2 で続けば停止(従来どおり。ただし鈍い指標)。
#   ⚠️ 検知できないもの: 損失が有限のまま高止まりする停滞。初回ランの経過は人が確認すること。
#   止まったら原因を切り分けて修正し、**新しい実験番号**で再実行する。
#   修正候補の第一は optimizer (LARS -> AdamW)。ただし逸脱になるので必ず記録する。
#
# ■ コスト見積り（DINO 0028 の実測 6.9分/epoch@4ノード からの外挿。BT の実測は無い）
#   BT は DINO の約 0.48 倍の計算量 => 0.22〜0.35 ノード時間/epoch
#   1600 epoch = 約 350〜560 ノード時間、8ノードで約 44〜69 時間 => 48h では終わらず
#   --resume で 2 本目が必要(resume.sh)。
#
# ■ 点検済み/未検証
#   - 検証済み: qsub/PBS/SIF/local SSD 等のサイト固有値は 0028 と同一(実機で完走)。
#     CPU スモーク(ログインノード, B=8)でヘッド 768->8192->8192->8192・損失・勾配が有限。
#   - 未検証: 8ノード同時の 141GB コピー時間(課金される)、毎 step 8192^2 fp32(約270MB)の
#     all_reduce、batch 256/GPU のメモリ(DINO は 256×(2+8小)で収まっている)、
#     BT の実測スループット。
#   - ⚠️ ログインノードでプロジェクト直下の `uv run` / `uv sync` を叩かないこと(--no-project)。
#     .venv が壊れて全ノード即死する(env/CONTAINER.md)。
#
# 投入前提:
#   bash experiments/0036_20261011_bt_vitb16_lars_all/preflight.sh   # 点検のみ(.venv 等)
#   qsub experiments/0036_20261011_bt_vitb16_lars_all/run_slurm.sh   # ⚠️ ユーザー承認後
# =====================================================
#PBS -N 0036_bt_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0036_20261011_bt_vitb16_lars_all/
#PBS -e logs/0036_20261011_bt_vitb16_lars_all/


module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0036_20261011_bt_vitb16_lars_all"

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
# Multi-node DDP (8 nodes)
#
# lr: 本リポジトリの --lr / --lr_bias は「最終lr」をそのまま渡す実装で、--ddp_linear_scale_lr は
#   --lr しか world_size 倍しない(lr_bias 側に同等のフラグが無い)ため使わず、両方とも
#   計算済みの最終値を渡す。公式の lr = base_lr × batch/256 に従うと
#     --lr      = 0.2    × 2048/256 = 1.6
#     --lr_bias = 0.0048 × 2048/256 = 0.0384  ← 0029 の値。**本ランでは使わない**(bias/LN も LARS 適応・lr 1.6)
#
# epoch: 1600 (= 624,000 step)。steps/epoch = floor(800000 / 2048) = 390 (drop_last)。
# warmup: 16 epoch (= 6,240 step)。
# 保存: 5 epoch ごと(約0.9GB/本 × 320本 ≈ 290GB。group quota 20T に対し現在 1.46T)。
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note bt_vitb16_lars_all --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name barlowtwins \
    --optimizer lars --lr 1.6 \
    --weight_decay 1.5e-6 \
    --batch_size 256 \
    --proj_dim 8192 --bt_lambda 5e-3 \
    --num_epoch 1600 --warmup_t 16 --lr_min 0.0 \
    --collapse_early_stop --collapse_ignore_uniformity --collapse_loss_rebound 2.0 \
    --bt_step_log \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
