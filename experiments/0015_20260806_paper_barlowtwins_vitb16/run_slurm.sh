#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# 8ノード(GH200 120GB×8)マルチノードDDP。batch_size 256/GPU × 8 = 実質2048
# (論文のbatch 2048と厳密に一致。従来の1ノードbatch 256版から、BSZ/LRを
# 論文値に揃えるために変更)。
#
# マルチノードDDP起動機構(scripts/slurm_entry.sh の _run_multi_node)は
# experiments/0020_20260808_pbs_multinode_ddp_smoketest/ (2ノード疎通確認)、
# および experiments/0017_20260806_paper_dino_vitb16/ (4ノード、batch 256/GPU、
# job 2522081で100 epoch完走)で実機検証済み。batch_size 256/GPUがGH200
# 120GBに収まることもこの0017の完走で確認済み(このBT実験自体は0015として
# 作成時点では未投入・未検証)。
#
# サイト固有値は experiments/0019_20260806_pbs_test/run_slurm.sh で実機の
# qsub投入まで通して検証済み(詳細な根拠はそちらのコメント参照)。要点:
#   - キュー名: `qstat --rsc` で確認できる small-g/medium-g/large-g/x-large-g は
#     execution queueであり直接投入すると `Access to queue is denied` で拒否
#     される。投入すべきはその親のルーティングキュー `regular-g`
#     (walltimeに応じてPBS側が自動でsmall〜x-largeへ振り分ける)。
#   - グループ/課金コード: `-P` ではなく `man qsub` に必須と明記されている
#     `-W group_list=<group>` を使う(値は `gd43`)。
#   - select句: `ngpus` は無効resource(拒否される)。`select=8` で
#     8ノード分(各ノード GH200 120GB GPU×1, 72 vCPU, 213GiB RAM)が
#     割り当てられる(ノードあたりGPU1台なので、8GPU確保にはノード数=8が必要)。
#   - SCRATCH_ROOT: 既定の `/scratch` は権限エラーで使えない。ノードローカル
#     NVMe SSDの実際のマウント点は `/local`。
#   - apptainer: ジョブスクリプトは投入時のシェル環境を引き継がないため、
#     スクリプト内で明示的に `module load apptainer/1.3.5` が必要。
#   - SIF_PATH: このリポジトリ専用の env/env.sif は未ビルド
#     (`apptainer build --fakeroot` に必要な /etc/subuid・/etc/subgid が無く未検証)。
#     暫定で共有コンテナ pytorch-ngc-26.06.sif (Python 3.12.3、torch 2.13.0a0が
#     プリインストール済み)を使う。
#   - .venv: pyproject.tomlのtorch extra(torch==2.11.0+cu130、pytorch-cu130
#     indexから取得)はaarch64(GH200)だと通常のtorch/torchvisionまで
#     まるごとPyPI版に置き換わってしまい、コンテナのNGC最適化ビルドが
#     無駄になる(`uv pip install timm` を素で叩くとtorch/torchvisionが
#     再ダウンロードされることを確認済み)。そのためNGC同梱のtorch/
#     torchvision/wandb/safetensors/tokenizers/huggingface_hub/einopsを
#     そのまま使う方針とし、`.venv` は
#       `uv venv --system-site-packages .venv`
#       `uv sync --active --inexact`(base依存のみ)
#       `uv pip install --python .venv/bin/python --no-deps timm`
#     で作成済み(`--no-deps` はtorch再インストールを防ぐため必須)。
#     lib.trainer.entry / lib.sslmodel のimportは確認済み(ログインノード上、
#     GPUなしのCPUチェックのみ。実際の学習実行はまだ未検証)。
#     accelerate/trl/peft/evaluate/bitsandbytes/sentencepiece/torchaudioは
#     現状のlib/配下のコードからは未参照のため未インストール(必要になった
#     時点で同様に `--no-deps` 付きで個別追加すること)。
#   - ⚠️ 未解決: `data/ssl_patches` がこのMiyabi環境にまだ配置されていない
#     (`data/` ディレクトリ自体が存在しない)。投入前に転送すること。
#   - walltime上限: `regular-g` は48時間が上限(49時間以上は
#     `Job violates queue and/or server resource limits` で拒否されることを
#     bisectionで実機確認済み)。1ノード版の元見積(196時間)は8ノード化で
#     単純計算では約24.5時間まで縮む想定(0017の4ノードDDPが実機で近い
#     スケーリングを示したことが根拠だが、BT自体でのスケーリング実測はまだ
#     ない)。48hに収まる見込みだが、念のためRUN_COMMAND側に
#     `--save_interval 5 --resume` を残してあるので、万一打ち切られても同じ
#     `qsub` を再実行すれば直近チェックポイントから再開できる想定
#     (実際の再開動作は未検証)。
#
# 投入前提: リポジトリのルートで `mkdir -p logs/0015_20260806_paper_barlowtwins_vitb16`
# してから `qsub experiments/0015_20260806_paper_barlowtwins_vitb16/run_slurm.sh`。
# 48h経過でジョブが打ち切られたら、同じコマンドで再度 `qsub` して再開する。
# =====================================================
#PBS -N 0015_20260806_paper_barlowtwins_vitb16
#PBS -q regular-g
#PBS -W group_list=gd43
#PBS -l select=8
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o logs/0015_20260806_paper_barlowtwins_vitb16/
#PBS -e logs/0015_20260806_paper_barlowtwins_vitb16/

# ジョブスクリプトはmodule環境を引き継がないため明示的にロードする。
module load apptainer/1.3.5

export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0015_20260806_paper_barlowtwins_vitb16"

# ノードローカルSSDの実際のマウント点(既定の/scratchはMiyabiに存在しない)。
export SCRATCH_ROOT="/local"

# 共有コンテナを暫定使用(このリポジトリ専用の env/env.sif は未ビルド)。
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage (理由はEXP14冒頭コメント参照。data/ssl_patchesのNFSランダム読みは遅い)
#
# ⚠️ 未検証: 8ノード同時にNFSから141GBをrsyncする(0017の4ノード時点でも
# 未検証だったリスクが、ノード数が倍になったことでさらに大きくなっている)。
# 失敗する場合はデータロードの早い段階で落ちるはずなので、48h丸ごと
# 無駄になるような壊れ方にはなりにくい想定。
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
# Multi-node DDP (8 nodes) — batch_size/lrを論文値に揃える
#
# wsi-ad experiments/20260714_paper_barlowtwins_vitb16/run_slurm.sh のpaper-faithful
# なCLI引数(Barlow Twins, ViT-B/16)を踏襲しつつ、BSZ/LRは論文の実値
# (batch_size=2048)にここで揃える。
#
# batch_size 256/GPU × 8ノード = 実質2048(論文と一致)。
#
# lrについて: 本リポジトリの --lr / --lr_bias は元々「batch_size=256における
# 最終lr」をそのまま渡す実装(--ddp_linear_scale_lr のようなworld_size自動
# 倍加フラグは lr_bias 側に無いため、ここでは使わず両方とも直接計算済みの
# 最終値を渡す)。Barlow Twins公式実装のLARS lrスケジュール
# (lr = base_lr × batch_size/256)に従うと:
#   --lr      = 0.2   × 2048/256 = 1.6
#   --lr_bias = 0.0048 × 2048/256 = 0.0384
# (旧: batch_size=256のときは --lr 0.2 --lr_bias 0.0048 のままでよかった
#  = ratio 1のため。今回batch_sizeを8倍にしたのでlrも同じ比率で8倍にする)
# =====================================================

NNODES=8
NPROC_PER_NODE=1
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} \
    --note paper_barlowtwins_vitb16 --project_path ${PROJECT_ROOT} --dir_result ${PROJECT_ROOT}/outputs/${EXP_NAME} \
    --model_name ViTB16 --ssl_name barlowtwins \
    --optimizer lars --lr 1.6 --lr_bias 0.0384 --lars_exclude_bias_bn \
    --weight_decay 1.5e-6 \
    --batch_size 256 \
    --proj_dim 8192 --bt_lambda 5e-3 \
    --num_epoch 100 --warmup_t 10 --lr_min 0.0 \
    --rank_monitor_interval 5 --save_interval 5 --resume"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
