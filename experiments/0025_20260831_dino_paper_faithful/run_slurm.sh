#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 4ノード(GH200 120GB×4)マルチノードDDP。
#
# 【本番】DINO ViT-B/16 を公式設定に揃えて再学習する。
#
# 背景（2026-08-31に決着した恒久崩壊の原因）:
#   0017/0021/0023/0024 は全て loss = ln(8192) = 9.0109 に張り付いて恒久崩壊した。
#   原因は teacher momentum でも teacher 温度でもなく、**weight decay を bias /
#   LayerNormゲイン(ndim<=1)にも掛けていたこと**だった。これらは wd に対抗する
#   勾配をほとんど持たないため毎step `γ <- γ(1 - lr*wd)` で単調に削られ、
#   0017ではep100で最終LayerNormのゲインが初期値の0.2%(0.0019)まで消えていた。
#   backboneが入力に依存しない定数を出す -> DINOLossのcenterがその定数へ収束 ->
#   teacher softmaxが厳密に一様 -> 勾配が厳密に0、という吸収状態に落ちる。
#   → lib/trainer/model.py:_wd_groups で公式(utils.get_params_groups)と同じく
#     bias/ndim<=1 を wd=0 の別グループに分離済み。
#   詳細は PROJECT_STATUS.md「🦕 DINO崩壊調査」。
#
# 公式(main_dino.py, --arch vit_base の既定値)への準拠状況:
#   ✅ 一致: lr 0.0005(×batch/256=2e-3) / warmup_epochs 10 / min_lr 1e-6 /
#            weight_decay 0.04->0.4 / adamw / global batch 1024 /
#            local_crops_number 8 / freeze_last_layer 1 / norm_last_layer True /
#            use_bn_in_head False / student_temp 0.1 / teacher_temp 0.04 固定
#            (公式は warmup_teacher_temp_epochs=0 = 0.04固定が既定。
#             0.04->0.07 warmup は ViT-S/16 300ep "boosted" レシピ側のオプション
#             であり ViT-B/16 の設定ではない。0024でこれを入れて崩壊させた)
#            ⚠️ 2026-09-04 訂正: 上の2行は誤り。論文本文(§3.2 Implementation details)は
#             「linear warm-up for tau_t from 0.04 to 0.07 during the first 30 epochs」を
#             アーキ別の但し書き無しに書いている。0.04固定は「論文の設定」ではなく
#             「リリースされたコードの既定」。つまり本ランは *コード既定* に追随しており、
#             0024 は *論文本文* に忠実だった(コード既定から逸脱していた)。
#             詳細は PROJECT_STATUS.md「◆ 論文本文と公式コードが食い違う3点」。
#   ✅ 本実験で論文値に揃えた4点:
#      --dino_out_dim 65536       (公式既定。0024以前は8192。崩壊時のloss基準も
#                                  ln(8192)=9.0109 -> ln(65536)=11.0904 に変わる。
#                                  CollapseMonitorは ssl_class.out_dim から自動追随)
#      --clip_grad 3.0            (公式既定。パラメータ毎のL2ノルムでクリップ)
#      --dino_momentum_start 0.996 --dino_momentum_end 1.0
#                                 (公式既定の cosine 0.996->1.0。公式ヘルプの
#                                  「0.9995 は batch 256 向けの推奨値」に対し、
#                                  本プロジェクトは global batch 1024 なので
#                                  0.996 が論文設定にあたる)
#      --dino_drop_path 0.1       (公式既定の stochastic depth。studentのみに適用)
#   ⚠️ 意図的な逸脱（論文に明記すること）:
#      augmentation。公式は global_crops_scale (0.4,1.0) / local_crops_scale
#      (0.05,0.4) だが、本プロジェクトは (0.2,1.0) / (0.05,0.2) で、回転 p=1.0 /
#      grayscale 0.05 / solarization無効。224pxパッチ済みなので過度なcropは
#      微細構造を壊す、H&Eの色は診断的に重要、という病理ドメイン由来の判断
#      (lib/sslmodel/utils.py:ssl_transform に根拠コメント)。かつ Goal.yaml が
#      手法間の拡張統一を要求しているため、DINOだけ論文値に戻してはいけない。
#      lr/wdスケジュールの粒度もepoch単位のまま(公式はiteration単位)。
#      schedulerは全手法共通なので、ここだけ変えると他手法と比較不能になる。
#
# 480 epoch の根拠:
#   論文の300 epochは ImageNet-1k(1,281,167枚)基準。本プロジェクトの学習データは
#   800,000枚(1M中200k=200WSIはvalidation fold)なので、同じ batch 1024 では
#   781 steps/epoch にしかならない。論文300epoch = 375,300 steps に揃えるには
#   375,300 / 781 ≈ 480 epoch が必要。
#   (0017の100 epoch は 78,100 steps = ImageNet換算で約62 epoch相当しかなかった。)
#
# walltime と本数:
#   regular-g の上限は48h。0023の実測は 6.45分/epoch (out_dim 8192) なので
#   480 epoch ≈ 51.6h。out_dim 65536 と drop_path でやや遅くなるぶんを見込むと
#   48h×2本では足りない可能性がある。`--resume` を付けてあるので、
#   完走するまで同じスクリプトを再投入すればよい(state.pt から再開する)。
#   ⚠️ 0024以前の state.pt からは再開できない(weight decay の param group 構成が
#      変わったため。entry.py が原因を明示して停止する)。必ず新規に開始すること。
#
# 監視の見方（2026-08-31更新）:
#   - 毎epochのログに `ln_gain`(backbone LayerNormゲインの平均)が出る。
#     **初期値1.0から単調に下がり続けていたら weight decay の param group 設定を疑う。**
#     今回の崩壊の主因はこれで、崩壊するずっと前から検知できた指標。
#   - 崩壊の一次指標は `train_loss` が ln(out_dim)=11.0904 に張り付くこと、
#     `uniformity` が 0 に近づくこと、`grad_norm` -> 0。
#   - `eff_rank` は鈍すぎて当てにならない(0024は36epoch崩壊し続けても閾値5.0に
#     掛からなかった)。`alignment` は Wang & Isola の alignment *loss* で
#     小さいほど良い指標なので、単独では崩壊判定に使えない。
#   - 起動直後に stdout の "weight decay 0.04->0.4: decayed N tensors /
#     exempt (bias & ndim<=1) M tensors" を必ず確認すること。M が 0 なら
#     除外が効いていない = 同じ崩壊を繰り返す。
#   - 毎epochのログの teacher_momentum / teacher_temp の実測値で、
#     スケジュールが実際に効いていることも確認すること。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0025_20260831_dino_paper_faithful`
# してから
#   `qsub experiments/0025_20260831_dino_paper_faithful/run_slurm.sh`
# =====================================================
#PBS -N 0025_dino_paper_faithful
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=4
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0025_20260831_dino_paper_faithful/
#PBS -e logs/0025_20260831_dino_paper_faithful/

module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0025_20260831_dino_paper_faithful"

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
    --note dino_paper_faithful --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name dino \
    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr \
    --weight_decay 0.04 --weight_decay_end 0.4 \
    --clip_grad 3.0 \
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
