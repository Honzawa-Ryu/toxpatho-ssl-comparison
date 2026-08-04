# マルチGPU・スパコン環境移行に向けた変更点

> 調査日: 2026-08-03
> 目的: 本リポジトリ（TG-GATE SSLベンチマーク基盤）を、シングルGPU・単一ノード運用から
> スパコン・複数GPU（将来的にはマルチノードDDP）環境へ移行する際に必要な変更を洗い出す。
> 実装前の棚卸しメモであり、このファイル自体は実装計画ではない（着手時は別途プラン化する）。

## 前提・現状認識

このコードベースは「将来のマルチGPU化」を見据えた設計が既にされている。
`lib/trainer/distributed.py` を中心に拡張ポイントが切り出されており、
`entry.py` / `model.py` からの呼び出し配線も済んでいる。
**ただし実装自体は全てno-opで、シングルGPU専用のまま。**
また、SSL手法側（Barlow Twins / SwAV / DINO）にはDDP対応コードが
書かれてはいるが、コメントアウトされて無効化されている箇所がある。

---

## 1. 分散学習ロジック本体（`lib/trainer/`）

| ファイル | 現状 | 必要な変更 |
|---|---|---|
| `lib/trainer/distributed.py` | `setup()`/`teardown()`/`rank()`/`world_size()`/`wrap()` が全てno-op（`rank()=0`, `world_size()=1`固定） | `torch.distributed.init_process_group` の呼び出し、環境変数（`RANK`/`WORLD_SIZE`/`LOCAL_RANK`）読み取り、`DistributedDataParallel` ラップを実装。呼び出し口は既に配線済みなので、ここを実装するだけで多くが繋がる |
| `lib/trainer/context.py:90` | `device = torch.device('cuda:0' if ...)` が固定 | `local_rank` に応じた `cuda:{local_rank}` への変更が必須（さもないと全プロセスがGPU0を奪い合う） |
| `lib/trainer/entry.py` (`_run()` 内 `wandb.init(...)`) | rank0ガードなしで全プロセスから呼ばれる | `distributed.is_main_process()`（定義済み・未使用）でガードしないと、重複run・重複ログが発生する |
| `lib/trainer/loop.py` (`torch.save(state, ...)`, `LOGGER.logger.info(...)`, `run.log(...)`) | 全プロセス実行前提 | チェックポイント書き込み・ログ・wandb.logをrank0限定にしないと、複数プロセスが同じファイルに競合書き込みする |
| `lib/trainer/data.py` | WebDatasetのshardをk-fold分割しているだけで、`split_by_node`/`split_by_worker`（webdatasetが提供）が未実装（コメントで明言済み） | rank単位でシャードを分割する実装が必須。今のままだと各GPUプロセスが同じシャードを重複して読む。`.with_epoch(1_000_000 // (batch_size * num_workers_train))` のepoch長計算もworld_sizeを考慮していないため再設計が必要 |
| `lib/trainer/optim.py` | world_sizeに応じた学習率スケーリング（linear scaling rule等）が未実装（コメントで明言済み） | global batch sizeが変わるSSL手法（特にBarlow Twins/SwAV/SimCLR）は要調整 |

---

## 2. SSL手法側のDDP対応（`lib/sslmodel/models/`）

見落としやすいポイント。GPUを増やしても、手法によっては**各GPU内のミニバッチだけで
統計量を計算すると数学的に誤った損失になる**ため、個別対応が必要。

- **`barlowtwins.py:126-132`**: cross-correlation行列を全GPUでall_reduceするコードが
  実装済みだが**まるごとコメントアウト**（`#if self.gather_distributed...`）。
  有効化しないと、各GPUが自分のローカルバッチだけでcross-correlationを計算する
  ＝実効バッチサイズが増えない。
- **`swav.py:156-176`**: Sinkhorn-Knopp反復のall_gather/all_reduceも同様に
  **コメントアウト**。SwAVはプロトタイプ割当がバッチ全体の分布に依存するため、
  GPU間gatherなしだと崩壊しやすい。
- **`dino.py:96-99` (`update_center`)**: teacher出力のcenterをローカルバッチの
  平均だけで更新している。マルチGPUでは全GPUのteacher出力をall_reduceして
  centerを揃える必要がある（未実装）。
- **BatchNorm**: ResNet50/DenseNet121バックボーンや各種ProjectionHeadが
  `nn.BatchNorm1d/2d` を使用（`lib/trainer/model.py` 参照）。DDPでは各GPUの
  ローカルバッチでBN統計が計算され、有効バッチサイズが小さいままになる
  （SimCLR系は特に影響大）。`torch.nn.SyncBatchNorm.convert_sync_batchnorm`
  の適用を検討する必要。

---

## 3. ジョブ投入・インフラ（Slurm/PBS周り）

- **`templates/run_slurm.sh:9`**: `#SBATCH --gres=gpu:1` `--nodes=1` 固定。
  GPU数・ノード数をパラメータ化する必要。
- **`RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"`**: 単一プロセス起動前提。
  1ノード内マルチGPUなら `torchrun --standalone --nproc_per_node=N`、
  複数ノードなら `srun ... torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --nnodes=$SLURM_NNODES --nproc_per_node=N`
  への変更が必要。`MASTER_ADDR`（`scontrol show hostnames` 等で先頭ノードを解決）
  や空きポートの決定ロジックも追加が要る。
- **`scripts/slurm_entry.sh` の `_run_single()`（578行目〜）**: `apptainer exec --nv ...`
  を1回だけ実行する設計。マルチノードでは各ノードで1回ずつコンテナ起動が必要なため、
  `srun apptainer exec --nv ...` のようにsrun配下でコンテナ起動する形へ変更が必要
  （現状はノード1台・プロセス1つの前提で書かれている）。
- **CPU/メモリ配分**: `num_workers_train = 16`（`lib/trainer/data.py:102`）と
  `--cpus-per-task=16` が1GPUプロセス分の想定。1ノードにGPU×N個載せてプロセスも
  N個立てる場合、`--cpus-per-task` をN倍にするかworker数を減らす調整が必要
  （オーバーサブスクリプション回避、`context.py:60` の `torch.set_num_threads(1)`
  と合わせて設計）。
- **`scripts/exp_common.sh: exp_resolve_partition`**: パーティション自動判定が
  実行時間のみに基づく（GPU数は考慮外）。対象スパコンのパーティション体系が
  GPU数で分かれている場合は拡張が必要。
- **PBS(Miyabi)対応の制約**: README/USAGE記載の通り、`runx`（ジョブ投入）は現状
  Slurm専用。対象スパコンがPBS系（Miyabiなど）でマルチGPU投入も自前ツールでやりたい
  場合、投入側（`tools/`, `.bashrc.d`）のPBS対応拡張が別途必要（現状PBSはキャンセル・
  監視のみ対応）。

---

## 4. コンテナ・依存関係

- **`env/env.def`**: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` ベース。
  NCCLはpip版torch wheelに同梱されるため単一ノード内マルチGPUは概ね問題ないはず。
  ただし**マルチノード**でInfiniBand/専用インターコネクトを使う環境では、
  サイト固有のNCCLプラグイン（例: libibverbs、UCX、ベンダー製OFIプラグイン等）が
  コンテナに入っているか要確認。対象スパコン側のドキュメント（推奨NCCL設定・
  `NCCL_SOCKET_IFNAME` 等の環境変数）に合わせて `env.def` の追加とイメージ再ビルドが
  必要になる可能性が高い。
- **`pyproject.toml`**: torch自体はcu130ホイールで問題なし。追加のマルチノード
  通信ライブラリ（例: `nccl-tests` での動作確認）が要る場合は別途検討。

---

## 5. 運用・検証観点

- wandb/Slack通知の重複防止（rank0限定化、上記1と連動）
- WebDatasetのシャード数がGPU数（world_size）に対して十分か
  （`data/shards` は現状20シャード → GPU数がこれを超えると空プロセスが出る）
- 有効バッチサイズが変わることによる、`Goal.yaml` 記載の「原論文設定準拠」方針との
  すり合わせ（LRだけでなくbatch/epoch設計全体の再検討）
- まずは1ノード内マルチGPU（DDP, `torchrun --standalone`）で動作・収束を検証し、
  その後マルチノードへ拡張する段階的アプローチを推奨
- `Makefile: test`/`tests/` にDDPのsmokeテスト追加、`scripts/preflight_check.py` に
  GPU数・torchrun起動設定の整合性チェック追加を検討

---

## 優先順位（推奨の着手順）

1. `lib/trainer/distributed.py` 実装 ＋ rank0ガード（entry.py / loop.py）
2. `lib/trainer/data.py` のシャードのrank分割
3. SSL手法別のgather/all_reduce有効化（barlowtwins.py / swav.py / dino.py）
4. `templates/run_slurm.sh` / `scripts/slurm_entry.sh` のマルチGPU起動対応
   （1ノード内 `torchrun --standalone` から）
5. マルチノード対応（`srun` + `torchrun` の rendezvous、コンテナ側ネットワーク設定）
