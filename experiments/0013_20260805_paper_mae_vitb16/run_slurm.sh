#!/bin/bash
#SBATCH --job-name=0013_20260805_paper_mae_vitb16
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0013_20260805_paper_mae_vitb16/%j_0013_20260805_paper_mae_vitb16.out
#SBATCH --error=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0013_20260805_paper_mae_vitb16/%j_0013_20260805_paper_mae_vitb16.out
#SBATCH --signal=B:USR1@7056
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180g
#SBATCH --time=196:00:00

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0013_20260805_paper_mae_vitb16/%A_%a_0013_20260805_paper_mae_vitb16.out
# #SBATCH --error=/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison/logs/0013_20260805_paper_mae_vitb16/%A_%a_0013_20260805_paper_mae_vitb16.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/andre01/honzawa/01-toxpatho/toxpatho-ssl-comparison"
export EXP_NAME="0013_20260805_paper_mae_vitb16"

# =====================================================
# 論文再現(paper-faithful) MAE ViT-B/16, 2x A6000 DDP
#
# なぜMAEから: SimSiam/BarlowTwins/MAE/DINOの4手法中、実VRAM消費が最も軽いと
# 見込まれるため(トークンmasking 75%でencoderが見るのはpatchの1/4だけ)。
# wsi-adの同一レシピの実測でもViT-L(329M, ViT-Bの約3.8倍)・bs=256で約24/48GB
# しか使っておらず、より小さいViT-Bでbs=512(4方式中最大のbatch)でも
# 単一A6000に収まっている(wsi-ad experiments/20260714_paper_mae_vitb16/run_slurm.sh)。
# 他3手法はDINOのmulticrop(2x224+8x96=10 views/sample)などで単位サンプルあたりの
# メモリ消費がより重いと見込まれ、後回し。
#
# なぜこの実験ディレクトリのexperiment.py/config.ymlを使わないか: 本番の学習は
# lib/trainer/entry.py(scripts/train/train_tggate.pyはその薄いシム)が
# checkpoint保存/resume/wandb/effective-rank監視まで一式持っており、
# wsi-adで100epoch実績のある経路をそのまま使うのが安全。experiments/*/experiment.py
# 側のoutput_utils.py規約(config.yml経由)は今回使わないため、生成されたテンプレート
# のままにしてある。
#
# batch_size 768/GPU: 論文の真のbatch=4096(base_lr=1.5e-4はbs=256基準のlinear
# scaling ruleの値)に対し、wsi-adの実績bs=512(単体GPU)はかなり縮小した再現だった。
# 「2GPUのままできる限り大きく」という要望を受け、vram_probe.py(このディレクトリに
# 同梱、run_slurm.shからは呼ばない使い捨てスクリプト)でMAE ViT-B/16 + AdamW +
# bf16 autocastの実メモリを1GPUで実測(2026-08-05, job 8283, A6000 50.9GB):
#   bs=512  -> peak 28.62GB (56%) OK
#   bs=768  -> peak 42.10GB (83%) OK
#   bs=1024 -> OOM (以降4096まで全部OOM)
# 768と1024の間の値は未探索だが、768は83%で十分な安全マージンを残しつつ512より
# 大きいためこれを採用。world_size=2・ddp_linear_scale_lr未指定(既定offのまま)
# なので実効グローバルバッチは768×2=1536(論文4096の約37.5%、wsi-ad実績512の3倍)。
# lrは同じlinear scaling rule(base_lr 1.5e-4 @ bs256)を1536に当てはめて
# 1.5e-4 × 1536/256 = 9e-4 に更新。MAEのlossはサンプルごと(batch統計に依存しない)
# なので、BarlowTwins等と違いgather_distributed相当の補正なしでこの分割は
# 数学的に単一GPU bs=1536相当と等価。
#
# ⚠️ 768/1024の間を詰めていないので、実データでの本番学習ではDataLoaderの
#    ホスト側バッファ等でここより早くOOMする可能性はゼロではない
#    (合成テンソルのみでの実測、かつ5step分のみの確認のため)。OOMしたら
#    まずbatch_sizeを512へ戻すのが最も簡単な対処。
#
# GPU数2: andre01ノードは4GPU。Goal.yamlの「最大3台(他ユーザ配慮)」の範囲内かつ、
# DDP配線はEXP11(0011_20260805_ddp_learning_test)で2GPU実地検証済みのため、
# まずは検証済みの2GPU構成を踏襲。
#
# 前提: data/ssl_patches (EXP8: 0008_20260805_sample_ssl_patches_memmap が作る
# 実パッチmemmap)がまだ存在しない。無い間はlib/trainer/data.pyのprepare_data()が
# FileNotFoundErrorで即座に落ちる想定通りの挙動。実データが揃ったらこのまま
# sbatchするだけでよい。
# =====================================================

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage
# /workspace はNFS（遅い）、/scratch はノード付属のm.2 SSD（速い）。
#
# USE_LOCAL_SSD_INPUT はデフォルト0（NFSを直読み）。data/ 全体をコピーすると
# 実験に不要なデータまで毎回転送して起動が遅くなるため、有効化する場合は
# 必ず DATA_SUBDIRS で実際に読むサブディレクトリだけを列挙すること。
# 有効化した場合、実験コード（experiment.py）側は project_root ではなく
# 環境変数 DATASET_DIR 経由でデータを読むこと（でないとコピーが無駄になる）。
#
# ⚠️ /scratch 側（SCRATCH_DIR）はジョブ終了時に自動削除されない
#    （rm -rf の誤削除リスクを避けるため）。出力は自動で /workspace/outputs/
#    へ回収されるが、SCRATCH_DIR自体は残るので、ディスクを圧迫してきたら
#    slurm.out に出る警告に従って手動で消すこと（詳細はdocs/USAGE.md 3-2節）。
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# USE_LOCAL_SSD_INPUT=1 にする場合のみ、コピー対象を列挙する
# （data/ からの相対パス。空のままだと data/ 全体をコピーする後方互換動作になる）。
# 例:
# DATA_SUBDIRS=(
#     "trident_processed/20x_224px_0px_overlap"
# )
DATA_SUBDIRS=()

# =====================================================
# python path
#
# experiments/${EXP_NAME}/experiment.py ではなく、本番の学習エントリポイント
# (scripts/train/train_tggate.py、実体は lib/trainer/entry.py)を直接叩く。
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/scripts/train/train_tggate.py"

# =====================================================
# Single run（デフォルト）
#
# wsi-ad experiments/20260714_paper_mae_vitb16/run_slurm.sh のpaper-faithfulな
# CLI引数をそのまま踏襲(optimizer/lr/weight_decay/augmentationはMAE原論文設定)。
# batch_sizeのみ512->256に変更(理由は上のコメント参照、2GPUで実効512を維持)。
# =====================================================

NPROC_PER_NODE=2
RUN_MODE="single"
RUN_COMMAND="torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} ${PYTHON_PATH} \
    --note paper_mae_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name mae \
    --optimizer adamw --beta2 0.95 --lr 9e-4 --weight_decay 0.05 \
    --batch_size 768 \
    --color_plob 0.0 --blur_plob 0.0 --solar_plob 0.0 \
    --num_epoch 100 --warmup_t 40 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--model" "--dataset")
#   GRID_VALUES=("bert roberta" "pubmed pmc")
#   → --model bert --dataset pubmed / --model bert --dataset pmc / ...
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it"
#     "BC5CDR BIORED"
# )

# =====================================================
# Seq run にしたい場合（1ジョブ内でGRIDを順次実行）
#
# 上と同様に RUN_MODE="seq" にし、BASE_COMMAND/GRID_ARGS/GRID_VALUES を設定する。
# こちらは #SBATCH --array は不要（1ジョブでループするため）。
# =====================================================

# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
# )
# GRID_VALUES=(
#     "bert roberta"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
