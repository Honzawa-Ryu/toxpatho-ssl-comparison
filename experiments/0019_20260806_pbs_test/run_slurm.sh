#!/bin/bash
# =====================================================
# PBS (qsub) 投入用スクリプト — Miyabi (Miyabi-G, PBS Pro) 向け。
# GPU 1台・DDP無し(単一プロセス python 実行)。
#
# これは 0001_20260803_1st_repo_test と同じ「Hello world」テンプレート
# (experiment.py は未編集) をそのまま使い、Miyabi上でのジョブ投入〜
# apptainer実行〜ログ回収〜completion.json書き込みの一連のパイプラインが
# 通るかを確認するための疎通確認用ジョブ。
#
# サイト固有値(このリポジトリを実際にMiyabi上で動かして確認済み):
#   - キュー名: `qstat --rsc` で確認できる小分けの表示
#     (small-g/medium-g/large-g/x-large-g)はいずれも直接投入すると
#     `qsub: Access to queue is denied` で拒否される。これらは実行先の
#     execution queueであり、ユーザーが投入すべきなのはその親の
#     ルーティングキュー `regular-g`(walltimeの値に応じてPBS側が
#     自動的にsmall〜x-largeへ振り分ける)。実際に `regular-g` +
#     `walltime=48:00:00` で投入できることを確認済み。
#   - walltime上限: `regular-g` は48時間が上限(49時間以上は
#     `Job violates queue and/or server resource limits` で拒否される
#     ことをbisectionで確認済み)。196時間は通らない。
#     (`debug-g`/`short-g` はさらに短い別キューで、`short-g`は8時間が上限)。
#   - グループ/課金コード: PBS Proの `-P` ではなく `man qsub` に明記されている
#     `-W group_list=<group>`(必須オプション)を使う。値は所属グループ
#     (このリポジトリは /work/gd43 配下にあるので `gd43`)。
#   - select句: Miyabi-Gの `select` chunkは `ngpus` を受け付けない
#     (`qsub: Resource invalid in "select" specification: ngpus` で拒否される)。
#     `select=1` だけで1ノード(GH200 120GB GPU×1, 72 vCPU, 213GiB RAM)が
#     丸ごと割り当てられる(実機で `nvidia-smi -L`/`nproc`/`free -h` で確認済み)。
#     ncpus/memを追加で明示指定することもできるが、指定した値が
#     queueの上限を超えると `Job violates queue and/or server resource limits`
#     で拒否される(具体的な上限値は未調査)ため、ここでは何も指定せず
#     ノード丸ごとをそのまま使う。
#   - apptainer: モジュールとして提供されている(`module load apptainer/1.3.5`)。
#     独自の env/env.def からのビルドは `apptainer build --fakeroot` に必要な
#     /etc/subuid・/etc/subgid のエントリがユーザーに無く未検証。代わりに
#     共有コンテナ /work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif
#     (Python 3.12.3, pyproject.toml の requires-python==3.12.*と一致)を使う。
#   - .venv: 上記コンテナ内で `uv sync`(base依存のみ。torch等の重い
#     optional-dependenciesは今回インストールしていない)して作成済み。
#
# 投入前提: リポジトリのルートで
#   `mkdir -p logs/0019_20260806_pbs_test`
# してから
#   `module load apptainer/1.3.5 && qsub experiments/0019_20260806_pbs_test/run_slurm.sh`
# を実行する(runx相当の自動化はPBS未対応、README参照)。
# =====================================================
#PBS -N 0019_20260806_pbs_test
#PBS -q debug-g
#PBS -W group_list=gd43
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o logs/test/
#PBS -e logs/test/

# ジョブスクリプトは投入時のシェル環境(module load済み)を引き継がないため、
# ここで明示的にロードする(未ロードだと scripts/slurm_entry.sh の
# `apptainer exec` が `apptainer: command not found` で失敗することを確認済み)。
module load apptainer/1.3.5

# PBS_O_WORKDIR = qsub を実行したディレクトリ。リポジトリのルートでqsubする前提。
export PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
export EXP_NAME="0019_20260806_pbs_test"

# scripts/slurm_entry.sh の既定 SCRATCH_ROOT=/scratch はMiyabiには存在しない
# (`mkdir: cannot create directory '/scratch': Permission denied` で実際に失敗
# することを確認済み)。ノードローカルNVMe SSDの実際のマウント点は `/local`
# (`df -h`実測: /dev/nvme0n1p9, 906G, 書き込み可)なのでこちらを使う。
export SCRATCH_ROOT="/local"

# 共有コンテナを使用(このリポジトリ専用の env/env.sif は未ビルド)。
export SIF_PATH="/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif"

export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# =====================================================
# Storage
# 疎通確認のみでデータを読まないため、ローカルSSDコピーは無効のまま。
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1
DATA_SUBDIRS=()

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
