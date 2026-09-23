# 実行コンテナの同定情報（再現性のための記録）

> 記録日: 2026-08-29
>
> このリポジトリの学習ジョブは**自前ビルドではなく、サイト提供の共有SIFイメージ**で動いている。
> 共有イメージは我々の管理下にないため（更新・削除の可能性がある）、同定に必要な情報を
> ここに控えておく。イメージが消えても下記の NGC タグから同じものを取り直せる。

## 実際に使われているイメージ

```
/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif
```

`experiments/*/run_slurm.sh` の全11実験がこのパスを `SIF_PATH` に指定しており、
実際に投入されたジョブの `logs/<exp>/<jobid>/_multinode_launch.sh` でも
このイメージが使われていることを確認済み。

### 提供元について

`/work/share/` 配下は Miyabi のシステム管理アカウント `z30105` が管理する共有領域で、
cuda-13.1 / gcc / hpcx / miniforge3 / DeepSpeed 等のシステムソフトウェア一式が置かれている。
`ContainerImages-G`（GPU向け）/ `ContainerImages-C`（CPU向け）はサイトが用意した
コンテナカタログであり、個人の私物ではない。NGC PyTorch は 25.01〜26.06 の14世代が並んでいる。

## 同定情報

| 項目 | 値 |
|---|---|
| **元イメージ（最重要）** | `nvcr.io/nvidia/pytorch:26.06-py3` (docker bootstrap) |
| **sha256** | `6920d9a6206728968186a76b85d28f504d0afca66c37724d2742182b0eb1b2ef` |
| SIF UUID | `a3bd31b3-773d-4506-97b2-7a5b42dac570` |
| NVIDIA build id | `337426144` |
| NVIDIA build ref | `d557151f4c7ddca284cb5e8d5ce78cee4d80f7e5` |
| サイズ | 10,284,302,336 bytes |
| mtime | 2026-07-14 07:30:49 +0900 |
| SIFビルド日時 | Tuesday_14_July_2026_7:29:51_JST |
| ビルドツール | apptainer 1.3.5 |
| アーキテクチャ | arm64 (aarch64 / GH200) |
| ベースOS | Ubuntu 24.04 |

### 同梱コンポーネントのバージョン

| コンポーネント | バージョン |
|---|---|
| **PyTorch** | **2.13.0a0+8145d63** |
| cuDNN | 9.23.0.39 |
| NCCL | 2.30.4 |
| cuBLAS | 13.5.1.27 |
| cuBLASMp | 0.9.0.2693 |
| cuFFT | 12.3.0.29 |
| cuSOLVER | 12.2.2.18 |
| cuSPARSE | 12.8.1.7 |
| cuSPARSELt | 0.9.1.1 |
| TensorRT | 11.0.0.114+cuda13.2 |
| NVVM | 13.3.33 |
| Nsight Systems / Compute | 2026.3.1.117 / 2026.2.0.7 |

取得コマンド（再取得したい場合）:

```bash
module load apptainer/1.3.5
SIF=/work/share/ContainerImages-G/singularity/pytorch-ngc-26.06.sif
apptainer inspect "$SIF"          # ラベル一式
apptainer sif header "$SIF"       # SIF UUID
sha256sum "$SIF"                  # 中身が差し替わっていないかの照合（10GB、数分かかる）
```

## 実行時の二層構造

学習ジョブは「コンテナ + プロジェクトの venv」の二層で動く
（`scripts/slurm_entry.sh` の `_multinode_incontainer.sh` 生成部を参照）:

```
apptainer exec --nv <上記SIF>
  └─ source ${PROJECT_ROOT}/.venv/bin/activate
       └─ python -m torch.distributed.run ... train_tggate.py
```

- **torch / CUDA / NCCL はコンテナ側**（上表のバージョン）
- **プロジェクト固有の依存は `.venv` 側**（`pyproject.toml` / `uv.lock`）

パッケージを足したいときは `.venv` に入れればよく、コンテナを触る必要がない。

### ⚠️ 宣言と実体のズレ（2026-08-29時点）

`pyproject.toml` は optional-dependencies の `torch` extra で

```toml
"torch==2.11.0+cu130",
```

を宣言しているが、**`.venv` には torch が入っていない**（`.venv/lib/python3.12/site-packages/torch*` が存在しない）。
そのため実行時に使われるのは**コンテナ側の PyTorch 2.13.0a0+8145d63** である
（コンテナ内で実行したスクリプトが `/usr/local/lib/python3.12/dist-packages/torch` を
ロードすることを実機確認済み）。

つまり **`uv.lock` / `pyproject.toml` は実際に使われている torch を固定していない。**
再現性の観点では、torchのバージョンを決めているのは pyproject ではなく**SIFイメージの方**である。
aarch64/GH200 で torch を pip 導入するのは困難で、NGC の最適化ビルドを使うのは妥当な判断だが、
`pyproject.toml` の `torch==2.11.0+cu130` は実態と食い違うため、
将来 `uv sync --all-extras` が成功すると**意図せず 2.11.0 がコンテナの 2.13.0a0 を
シャドウする**可能性がある点に注意。

## ⚠️ `uv run` が `.venv` を作り直す事故（2026-09-19 発生）

**ログインノードでプロジェクト直下の `uv run` / `uv sync` を叩いてはいけない。**
`--no-project`（または `UV_PROJECT_ENVIRONMENT` で別の場所を指す）を必ず付ける。

### 何が起きるか

| | python3 |
|---|---|
| ログインノード `/usr/bin/python3` | **3.9.25** |
| コンテナ `/usr/bin/python3` | **3.12.3** |

学習用 `.venv` は**コンテナの 3.12.3 を土台**に作られている（`pyvenv.cfg` の
`home = /usr/bin`）。ところがログインノードから同じパスを引くと 3.9 に化けるため、
`pyproject.toml` の `requires-python = "==3.12.*"` を満たさない**壊れた環境に見える**。
そこで uv は「使えない」と判断し、**確認なしに `.venv` を作り直す**:

- 土台が uv 管理の standalone CPython（`~/.local/share/uv/python/...`）に変わる
- `include-system-site-packages = false` になり、**コンテナ側 torch への道筋が切れる**
- `--all-extras` が付かないので **timm / wandb も入らない**

学習ジョブは `source ${PROJECT_ROOT}/.venv/bin/activate` → `python -m torch.distributed.run`
という経路（`scripts/slurm_entry.sh`）なので、次の投入で全ノードが即死する:

```
/work/.../.venv/bin/python: Error while finding module specification for
'torch.distributed.run' (ModuleNotFoundError: No module named 'torch')
```

### 実際の事故

`scripts/analysis/plot_training_curves.py` の docstring にあった実行例

```bash
uv run --with matplotlib python scripts/analysis/plot_training_curves.py ...   # ← --no-project が無い
```

をログインノードで実行した結果、`.venv/pyvenv.cfg` が **15:36:16** に書き換わり、
グラフ `outputs/_analysis/curves_0025_0026_0028.png` が **15:37** に生成された。
その 8 時間後に投入した exp 0028 の resume（**job 3398346**）が、
4ノードすべて上記エラーで起動12分後に落ちた。**グラフを1枚描いただけなので、
venv を触った自覚は残らない。** docstring は `--no-project` 付きに修正済み。

### 直し方

```bash
bash tools/rebuild_venv.sh            # 点検のみ
bash tools/rebuild_venv.sh --apply    # 退避してから作り直す
```

やっていること（すべてコンテナ内）:

```bash
uv venv --python /usr/bin/python3 --system-site-packages .venv   # 土台をコンテナ python に固定
uv sync --active --inexact                                      # base 依存のみ(torch extra は入れない)
uv pip install --no-deps 'timm==1.0.28'                         # timm だけ個別。torch を引かせない
```

`--no-deps` が要るのは、`uv pip install timm` だと **uv が torch 2.14.0 を venv 側に
入れてしまい**、コンテナの 2.13.0a0 をシャドウするため（実機確認済み）。
`uv` は `--system-site-packages` 下でも既存 torch を「充足済み」と見なさない。

### 構成の根拠

壊れる前の `.venv` の中身は **job 3382307 自身の wandb 記録**から復元した:

```
wandb/offline-run-20260917_195710-0laflbix/files/requirements.txt   (350 パッケージ)
```

これとコンテナ側 `pip freeze`（217 パッケージ）の差分＝**venv 固有は 25 個だけ**で、
`pyproject.toml` の base 依存 + `timm` に一致する。
**torch / torchvision / wandb はすべてコンテナ側**から来ていた（上節のとおり）。

### 再発防止

`experiments/0028_20260917_dino_lr_half/resume.sh` の点検項目7 が、投入前に
`pyvenv.cfg` の `home` / `include-system-site-packages` と `timm` の有無を見て止める。
4ノード確保してから落ちるのは高くつくので、他の実験の投入スクリプトにも同じ検査を入れること。

## イメージが消えた/変わった場合の復旧

元が NGC の公開イメージなので、同じタグから作り直せる:

```bash
apptainer build pytorch-ngc-26.06.sif docker://nvcr.io/nvidia/pytorch:26.06-py3
```

同一性は上表の build id (`337426144`) / build ref (`d557151f...`) で照合できる。

## 補足: `.bashrc` の `SIF_PATH` は実験には効いていない

リポジトリの `.bashrc` は

```bash
export SIF_PATH="./env.sif"
```

としているが、`experiments/*/run_slurm.sh` が後から共有イメージのパスで上書きするため、
**学習ジョブには影響しない**。この値を見るのは `tools/uv_sync.sh` と
`tools/start_jupyter.sh` のみで、`env/env.sif` は未ビルドのため（`env/` には
`env.def` と `env.example` しかない）そのままでは動かない。

長期の再現性を重視するなら `env/env.def` から自前ビルドして `env/env.sif` を
実効化する選択肢もあるが、2026-08-29時点ではサイト提供イメージで運用する方針。
