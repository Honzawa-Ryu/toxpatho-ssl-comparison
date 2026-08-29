# プロジェクト状況レポート: TG-GATE SSL ベンチマーク

> **重要: このファイルは毎回の作業後に必ず更新すること。**
> 次セッション開始時にはまずこのファイルを読んで状況を把握する。

最終更新: 2026-08-29（セッション8: DINOの崩壊要因切り分け。詳細は下記「🦕 DINO崩壊調査」）

> **やること・研究方針の一覧は [TODO.md](TODO.md) を参照**（試験間差除去テーマC章を含む）。

---

## 🦕 DINO崩壊調査（2026-08-29, セッション8）

### 現在走っているジョブ

| Job ID | 実験 | 内容 |
|---|---|---|
| 3225049 | `0023_20260829_dino_diag_momentum_paper` | teacher momentum のみ論文値 (cosine 0.996→1.0) |
| 3225052 | `0024_20260829_dino_diag_teacher_temp_paper` | teacher温度のみ論文値 (30ep warmup 0.04→0.07) |

いずれも4ノード・walltime 5h(≈45 epoch)。他は0017と完全同一。`--collapse_early_stop` 付き。

**⚠️ 結果が出るまでコード凍結中。** 下記「未対応の3点」を実装すると診断の前提が変わるため、
0023/0024 の完走・解釈が終わるまで lib/ 配下を触らないこと。

### 経緯（0017 → 0021 → 0022）

- **0017** (100 epoch完走): momentum 0.9995固定 / teacher温度 0.04固定 の
  「anti-collapse defaults」。epoch12〜33で train_loss が 9.00 前後
  (= ln(8192) = 出力が一様分布に潰れた値) に張り付き、epoch25で eff_rank 6.04 まで
  低下してから**自然回復**。最終 train_loss 4.32 でまだ下降中だった。
- **0021** (paper-faithful化を試行): momentum/温度を論文スケジュールへ + lr/wdのstep粒度化 +
  scratch統一の**3つを同時に変更** → epoch15で恒久崩壊 (loss 9.0109固定, grad_norm 1e-4)。
  原因は切り分けられていない。
- **0022** (0017からの継続学習): 未実行のまま。`--model_path` は重みのみでoptimizer状態を
  引き継がないため、継続ではなく新規学習で伸ばす方針に変更した。

### ⚠️ 判明した2つの事故

**1. DDP forward のバグ修正が stash に巻き込まれて消えていた（2026-08-29 復元, commit c782e58）**

`DINO.calc_loss` が DDPラップ済みモデルに対して `model.forward_student(views)` と
直接呼んでおり、DDPは任意属性を `.module` へ転送しないため AttributeError で即死する。
0017 は 2026-08-10 に実際にこれで2回失敗している (job 2513399, 全rank)。
Aug10→11に未コミットで修正され0017は完走したが、8/22に「0021 paper-faithful」を
stashした際にこの修正まで巻き添えで退避され、ツリーが壊れた状態に戻っていた。
**教訓: 修正とスケジュール変更を同じ未コミットツリーに混ぜない。**

**2. 0021の実装が部分的に失われており、何を検証したのか確定できない**

0021の `config.json` には `momentum_start: 0.996` 等が記録されているが、
それを適用する側のコード（entry.pyのCLI引数・loop.pyの毎step更新呼び出し）が
ツリーにもstashにも存在しない。配線が無かったなら 0021 は
「momentum 0.996 **固定**・温度 0.04 固定」で走ったことになり、論文準拠ではない。
→ 2026-08-29に独立トグルとして実装し直した (commit c9e39bb)。
   **毎epochのログに `teacher_momentum` / `teacher_temp` の実測値を出すようにしたので、
   今後は「有効にしたつもりで固定のまま」を必ず検出できる。まず最初にこれを確認すること。**

### epoch数は480が論文相当（100や300ではない）

論文の300 epochは ImageNet-1k (1,281,167枚) 基準。本プロジェクトの学習データは
**800,000枚**（1M中200k = 200 WSIはvalidation fold）で、同じ global batch 1024 では
**781 steps/epoch**（0017ログの実測値）にしかならない。

| | DINO論文 | 本プロジェクト |
|---|---|---|
| 学習画像数 | 1,281,167 | 800,000 |
| steps/epoch | 1,251 | 781 |
| 論文300 epoch | 375,300 steps | — |
| **同等にするには** | — | **480 epoch** |

**0017の100 epoch = 78,100 steps = ImageNet換算で約62 epoch相当**しかなく、
論文の約1/5。大幅に未収束だった（loss 4.32でまだ下降中だったのは当然）。
480 epoch ≈ 51.6時間（6.45分/epoch実測）なので walltime 48h を超え、`--resume` で2本必要。

### 論文（main_dino.py 既定値）との照合結果

**一致している**: `lr` 0.0005（linear scaling で実効2e-3）/ `warmup_epochs` 10 /
`min_lr` 1e-6 / `weight_decay` 0.04→0.4 / `optimizer` adamw / `batch` 1024 /
`local_crops_number` 8 / `freeze_last_layer` 1 / `norm_last_layer` True /
`student_temp` 0.1 / 2×224² + 8×96²

**注意**: 公式既定は `teacher_temp=0.04, warmup_teacher_temp_epochs=0` = 「0.04固定」。
つまり**現状の0.04固定は論文からの逸脱ではない**（0.04→0.07 warmupはREADMEの
"boosted" ViT-S 300ep レシピ側のオプション）。

**拡張(augmentation)の相違は意図的**: 回転 p=1.0 / grayscale 0.05 / crop scale (0.2,1.0) /
solarization無効 は、病理ドメイン由来の判断（`lib/sslmodel/utils.py:ssl_transform` に
根拠コメントあり: 224pxパッチ済みなので過度なcropは微細構造を壊す、H&Eの色は診断的に重要）。
かつ Goal.yaml が手法間の拡張統一を要求しているため、**DINOだけ論文値に戻してはいけない**。

### 未対応の3点（診断結果が出たらまとめて実装する）

| # | 項目 | 現状 | 公式 | 崩壊リスク |
|---|---|---|---|---|
| 1 | **勾配クリッピング** | **lib全体に1件も無い** | `clip_grad=3.0` を毎step適用 | 追加すると**崩壊しにくくなる**。実装漏れであり設計判断ではない。最優先 |
| 2 | `out_dim` | 8192 | 65536 | 大きい方が**崩壊しにくい**。ただしメモリ増(未実測)、崩壊時のloss基準が ln(8192)=9.01 → ln(65536)=11.09 に変わる |
| 3 | lr/wdスケジュール粒度 | epoch単位(階段状) | **iteration単位** | 僅かに安全側。ただし scheduler は全手法共通なので、DINO限定のopt-inにしないと他手法の既完走ランと比較不能になる |

1が欠けたまま崩壊調査をしている点に注意（0017のep25近傍崩壊・0021の恒久崩壊にも
寄与している可能性がある）。「クリッピングを入れれば論文設定でも崩壊しないのでは」は
診断後に検証する価値がある仮説。

### 崩壊の見分け方

- `eff_rank` < 5 が継続 / `train_loss` が **ln(out_dim)**（現状8192なら9.0109）に張り付く /
  `grad_norm` → 0
- ただし**0017もep25で eff_rank 6.04 まで落ちてから回復している**ので、
  一時的な低下だけで崩壊と即断しないこと。`CollapseMonitor` の既定は
  「rank_threshold 5.0 を patience 2回(=10 epoch)連続で下回る」。

---

## 🔀 マルチGPU移行（2026-08-04, セッション7: レビュー）

前セッション（セッション6以降のどこか）で `docs/multi_gpu_migration.md` の優先順位1〜3
（`lib/trainer/distributed.py` 実装／`data.py` のrank分割／SSL手法別all_reduce有効化）が
未コミットのまま実装されていた。本セッションはこれをレビューし、単一GPU後方互換性
（Goal.yaml必須要件）を静的に検証した。**agentパーティション capsule内はGPUなし・
python/torchrun等の直接実行がhookで禁止**のため、実行テストは不可。全て読解による
静的レビュー。

### レビュー結果
- **項目1（`distributed.py`実装 + rank0ガード）**: 実装済み・妥当。
  `RANK`/`WORLD_SIZE`未設定時は`setup()`/`wrap()`/`barrier()`等が全てno-opで
  従来のシングルGPU実行と同一に振る舞うことを確認（`is_distributed_env()`が
  `WORLD_SIZE`未設定または`<=1`ならFalseを返す設計）。
  `entry.py`のwandb.init、`loop.py`のcheckpoint保存/eff_rank監視/logger書き出しが
  `distributed.is_main_process()`でrank0限定化済み。
- **項目2（`data.py`のrank分割）**: 実装済み・妥当。
  `create_sharded_dataset(..., split_by_rank=True)`で`wds.split_by_node`を使用
  （world_size=1では`wds.shardlists.single_node_only`相当の挙動と同一のはず。
  webdataset自体がこのcapsuleに未インストールのため実行検証はできず、
  API存在は過去の知識ベース・コード上の使用パターンからの判断）。
  eval_dataset側は意図的に`split_by_rank=False`のまま（全rank同一valで
  early stopping判定を揃えるため）。epoch長計算もworld_size考慮済み。
- **項目3（SSL手法別all_reduce有効化）**: 実装済み・妥当。
  Barlow Twinsのcross-correlation、SwAVのSinkhorn-Knopp、DINOのteacher center更新、
  いずれも`dist.is_initialized()`ガード付きで有効化されており、`gather_distributed`
  系フラグは`lib/sslmodel/sslutils.py`で`distributed.world_size() > 1`から自動設定
  （単一GPUでは常にFalse = 従来通りコメントアウト時と同じ計算）。
- **見つけた問題（修正済み）**: 実装自体にバグはなかったが、3箇所のコメントが
  「将来DDPラップが入る場所」「REFACTOR_PLAN.md §6-4」等、**実装済みである現状と
  矛盾する内容のまま**だった（`entry.py`の`main()`docstring・`distributed.wrap()`
  呼び出し直後のコメント、`model.py`のモジュールdocstring）。実装内容に合わせて
  更新した。

### 未実施（項目4・5、次セッションへの申し送り）
- `templates/run_slurm.sh` / `scripts/slurm_entry.sh` の1ノード内マルチGPU対応
  （`torchrun --standalone --nproc_per_node=N`化）とマルチノード対応は**未着手**。
  本capsuleはGPUなし・python直接実行不可のため実機検証ができず、「既存の単一GPU
  投入は無変更で動作すること = 後方互換必須」（Goal.yaml）を壊すリスクを検証できない
  まま変更するのは避けた。実GPUで検証できるセッション（ログインノード経由、または
  GPU付きcapsule）で着手すること。
- 項目1〜3の**実機での動作確認**（実際に`torchrun --nproc_per_node=2`等でBarlow
  Twins等を数epoch回し、崩壊しないこと・checkpointが1つだけ書かれること・
  wandb runが重複しないことを確認）も未実施。単体テスト（pytest）もこのリポジトリの
  distributed周りには存在しない。次にGPUが使えるセッションでのsmoke testを推奨。
- `wds.split_by_node` / `wds.shardlists.single_node_only` のAPI存在・挙動は
  webdataset未インストール環境のため未検証（項目2参照）。

---

## 🔧 パイプライン断絶の修正（2026-08-01, セッション6）

`.agent-runs/20260801_155247_review_claude_.../artifacts/report.md` のレビューで、
「Goal.yaml の意図」と「実装・実行設定」の乖離が複数見つかった。対応内容:

1. **`scripts/train/train_tggate.py` が欠落**しており26実験全ての `run_slurm.sh` が
   起動不能だった → `lib/trainer/entry.py` を呼ぶ薄いシムとして復元。
2. **`scripts/analysis/methods_paper.yaml` が欠落**しており paper 版の表現比較
   （`0003`〜`0006`）が起動不能だった → 主比較4手法（Barlow Twins/DINO/MAE/SimSiam）
   分を作成。`model_path` は各 `20260714_paper_*/run_slurm.sh` の
   `PROJECT_ROOT=/workspace/andre01/honzawa/wsi-ad` に合わせた絶対パス。
   SwAV は方針通り除外（`20260714_paper_swav_vitb16` はチェックポイントとして残置、
   比較設定には含めない）。
   - `scripts/analysis/methods.yaml`（0001/0002が使う、paper以前の4手法+foundation版）は
     今回は未作成（DINOの版違い(`20260710_000002` vs `20260711_000001_dino_v2`)を
     ドキュメントから断定できず、ユーザー判断で見送り）。次に必要になった時点で要確認。
3. **`Goal.yaml` を実態に合わせて改訂**（ユーザー判断）:
   - 方針3（学習終了）: 「early stopping を基本方針」→ 主比較は**固定100エポック**が
     実態であるためそちらに合わせて修正。崩壊監視(`rank_monitor_interval`)は
     停止基準ではなく健全性チェックとして独立に継続。
   - constraints「公平性」: 「可能な限り統一」→ 各手法の**原論文設定に忠実**が実態
     であるためそちらに合わせて修正（`20260714_paper_*` はoptimizer/lr/batch_sizeが
     手法ごとに異なる paper-faithful 設定）。
   - SwAV除外の方針自体は変更なし。`20260714_paper_swav_vitb16` が比較対象に
     含まれない旨を明記して整理。

### 未実施・要フォローアップ
- `20260714_paper_*` の5ジョブはこのセッションでは未投入（シム復元後の初回実行検証がまだ）。
  次回、まず1手法（例: barlowtwins、既に07-04版で健全学習実績あり）を短時間ジョブで
  シムの動作確認してから本番投入することを推奨。
- 26件の旧`run_slurm.sh`が使う `PROJECT_ROOT`（`andre01`/`david01`/`filesrv02`混在）の
  一貫性は今回検証していない。少なくとも `20260714_paper_*` の5件は全て
  `andre01`で揃っており、同一パーティション(`x-large-andre01`)で完結する想定。

---

## 🔄 方針転換（2026-07-10） — 詳細は `Goal.yaml` を参照

プロジェクトの単一の真実源(source of truth)は **`Goal.yaml`**（本日新規作成）。要点:

1. **比較対象モデル**: 対照学習(SimCLR/SwAV)は負例前提で今回の関心と異なるため主比較から除外。
   **非対照学習4手法 = SimSiam / Barlow Twins / MAE / DINO** を主対象とする。将来 UNI を追加。
   - MAE / DINO を新規実装（`src/sslmodel/models/mae.py`, `dino.py`、timm ViT ベース）。
   - DINO は公平性のため **2 global crops(224) / batch 256**（SimSiam/BT と同一のビュー・バッチ構成）。
     multi-crop の local crops はオプション(既定 off)。→ SwAV の OOM 問題を回避。
   - MAE は単一ビュー・batch 256（マスク率0.75で軽量）。
2. **比較方法**: 下流タスク性能(Patho-Bench)より **学習表現そのものの性質** を重視。
   (a) CKA・埋め込み類似度、(b) クラスタリング一致度(ARI 等)、(c) プロトタイプパッチ可視化。
   → Patho-Bench は補助的位置づけに後退（既存準備は残置）。
3. **学習終了**: **early stopping を基本**（既存 `EarlyStopping` を使用、崩壊は eff_rank で監視）。
   新規ViT系ジョブは `patience 15`（学習損失の頭打ちで停止）。旧ResNet系は `patience 100`（実質無効）だった。

### 実行状況（2026-07-10 sbatch投入、node andre01 / 4GPU・同時最大3の運用）
| Job | 手法 | model | batch | lr | 状態 |
|---|---|---|---|---|---|
| 6194 | SwAV(旧・比較外) | ResNet50 | 64 | 1e-3 | Running（継続、キャンセルせず） |
| 6287 | **MAE** | ViTB16 | 256 | 1.5e-4 | Running（~3.1 it/s, OOMなし） |
| 6288 | **DINO** | ViTB16 | 128 | 2.5e-4 | Running（~1.5 it/s, OOMなし） |
| 6289 | **SimSiam(wd1e-4再学習)** | ResNet50 | 256 | 1e-3 | PENDING（`afterany:6194` 依存で待機） |

- **Barlow Twins**（20260704_000003）は前セッションで健全に完走済み → 再学習不要、比較にそのまま使用。
- 新規実装: `src/sslmodel/models/mae.py`, `dino.py`（timm ViT ベース、自己完結）。
  `sslutils.py` に `MAE`/`DINO` ラッパ追加、`src/utils.py` の `DICT_SSL` に `mae`/`dino` 登録。
  `train_tggate.py` は mae/dino のとき torchvision バックボーンの重みDLを回避（`weights=None`）。
- DINO は公平性のため 2 global crops(224)・EMA teacher（`update_moving_average` は既存train loopが呼ぶ）。
  `SSLEvaluator` 対応のため `DINO.forward`（student投影）を追加。MAEは単一ビューのためeval指標は空（recon lossで監視）。
### 表現比較パイプライン（2026-07-10 実装・検証済み）
Goal.yaml の比較方法(a)(b)(c)を実装。既存ResNet50チェックポイント3本(barlowtwins/旧simsiam/旧simclr)で
end-to-end検証済み（CKAが旧simsiamの崩壊を0.04として正しく検出）。
- `src/analysis/cka.py`: 線形/RBF CKA（次元非依存 → ResNet 2048d と ViT 768d を直接比較可能）。
- `scripts/analysis/extract_embeddings.py`: **共通の固定パッチ集合**（同一パッチ・同一順序）を各手法で埋め込み。
  欠損チェックポイントはスキップするので学習完了前でも実行可。サムネイル/メタも保存（プロトタイプ用）。
- `scripts/analysis/compare_representations.py`: (a) CKA行列、(b) k-means + ARI/NMI(手法間)、
  (c) クラスタ別プロトタイプパッチのモンタージュ、(d) t-SNE(PCA-50前処理)、
  (e) **任意**: `--label_csv key,label` を渡すと意味ARI/NMIとラベル重心間距離(A/B/C近接)を出力。
- `scripts/analysis/methods.yaml`: 4手法のチェックポイントパス。
- 実行: 全学習完了後に `runx 1`（extract）→ `runx 2`（compare）。欠損分はスキップされるので順次でも可。
  Phase 4 で `run_analysis_slurm.sh` を廃し、1フェーズ = 1実験ディレクトリに載せ替えた
  （REFACTOR_PLAN.md §7-6）。paper 版は `0003`〜`0006` の4実験に分割済み。
- ラベルは現状リポジトリに無い（findings CSVは分類名一覧のみ）。意味比較したい場合は
  Open TG-GATEsのスライド単位アノテーション(WSI→finding/dose)か外部ラベル付きH&Eパッチが別途必要。

### バグ修正（2026-07-10）
- `sslutils.py` の `SimSiam.prepare_featurize_model`: チェックポイント未ロード＋`.encoder`(存在しない)参照の
  2バグを修正（`model_path` をロードし `.backbone` を返す）。→ 特徴抽出/比較で simsiam が使えるようになった。

旧計画（SimCLR/SimSiam/BarlowTwins/SwAV の4本 + Patho-Bench主軸）の記述は以下に歴史的経緯として残す。

---

## プロジェクト概要

**目的**: Open TG-GATEs（肝臓H&E WSI）を用いてSSLアルゴリズムを事前学習し、Patho-Bench（95タスク）で体系的にベンチマークする。SSL手法固有のロジックが病理画像にどう作用するかを同一条件で検証・マッピングする。

**計算資源**: NVIDIA RTX A6000 (VRAM 48GB) × 1、混合精度（bfloat16）

| 設定 | 値 |
|------|-----|
| バックボーン | ResNet50 |
| SSL手法 | SimCLR, SimSiam, Barlow Twins, SwAV |
| 事前学習データ | Open TG-GATEs 肝臓H&E（224px パッチ、1,000,000枚） |
| バッチサイズ | 256（全手法統一） |
| エポック数 | 100 |
| 学習率 | 1e-3（Adam、cos annealing + warmup） |
| 評価 | Patho-Bench（H&E系タスク、臓器不問で代表タスクを選定。肝臓タスクは存在しないため） |

---

## ディレクトリ構成（主要ファイル）

```
/workspace/
├── src/sslmodel/
│   ├── models/
│   │   ├── simclr.py          ★ 新規: SimCLR + NT-Xent損失
│   │   ├── simsiam.py         ★ 修正: project_single() 追加
│   │   ├── barlowtwins.py
│   │   └── swav.py
│   ├── sslutils.py            ★ 修正: SimCLR クラス追加
│   ├── evaluation.py          ★ 修正: project_single フォールバック追加
│   └── utils.py               ★ 修正: ssl_transform 前処理修正
├── src/utils.py               ★ 修正: DICT_SSL に simclr 追加
├── scripts/
│   ├── train/train_tggate.py  メイン学習スクリプト
│   └── evaluate/
│       ├── abmil_eval.py
│       └── extract_features_pathobench.py  ★ 新規: Patho-Bench用H5出力
├── experiments/
│   ├── 20260704_000001_tggate_resnet50_simclr/run_slurm.sh     ★ 新規
│   ├── 20260704_000002_tggate_resnet50_simsiam/run_slurm.sh    ★ 新規
│   ├── 20260704_000003_tggate_resnet50_barlowtwins/run_slurm.sh ★ 新規
│   └── 20260704_000004_tggate_resnet50_swav/run_slurm.sh       ★ 新規
├── setup_environment.sh       ★ 新規: h5py追加 + Patho-Bench clone + venv構築（GPU不要ジョブ）
├── submit_benchmark.sh        ★ 新規: 4本一括投入（3台同時上限、4本目に依存設定）
├── PROJECT_STATUS.md          ★ このファイル（毎回更新）
├── tools/                     .gitignore済み（Patho-Bench clone先）
│   └── Patho-Bench/           setup_environment.sh 実行後に生成
└── data/shards/               shard_0000.tar 〜 shard_0019.tar（20シャード）
```

---

## 実装済み変更内容（2026-07-04 完了）

### 追加実装
- **SimCLR** (`src/sslmodel/models/simclr.py`): NT-Xent（InfoNCE）損失、2層プロジェクションヘッド
- **Patho-Bench特徴量抽出スクリプト** (`scripts/evaluate/extract_features_pathobench.py`): Trident互換H5形式（`features`/`coords` per WSI）

### バグ修正
- **SSLEvaluator**: SimSiam/BYOLの `forward(x1, x2)` シグネチャ非互換 → `project_single()` フォールバックで解決
- **SimSiam**: `project_single(x)` メソッド追加（backbone + projector の単一ビュー推論）

### 前処理修正（病理画像に最適化）
| 項目 | 修正前 | 修正後 | 理由 |
|------|--------|--------|------|
| `RandomResizedCrop` | `scale=(0.08, 1.0)` | `scale=(0.2, 1.0)` | 最小クロップ63px→100px、微細構造保持 |
| `RandomGrayscale` | `p=0.2` | `p=0.05` | H&E染色の色（核/細胞質）は診断的情報 |

---

## 現在の状況

### SSL学習（Phase 1）結果確認（2026-07-08）

4本とも投入から数日経過していたため結果を確認。**時間切れ（72h制限）は主因ではなく**（3本とも66〜69hで完走）、手法固有の問題が2件見つかった。

- [x] **BarlowTwins** (ResNet50) → `experiments/20260704_000003.../run_slurm.sh`
  - 完走（66.4h）。loss 18.8→0.13、eff_rank ~617〜698で安定。**崩壊なし、健全**。`outputs/20260704_000003_tggate_resnet50_barlowtwins/`（実体は `result/workspace/.../outputs/20260704_000003.../model_ssl.pt`）
- [x] **SimCLR** (ResNet50) → `experiments/20260704_000001.../run_slurm.sh`
  - 完走（66.6h）。loss 4.86→4.29とほぼ横ばい。eff_rank 712→994で見た目は健全だが、batch=256（原論文は4096〜8192）ゆえの負例不足の影響とみられる。要ダウンストリーム評価での確認。
- [x] **SimSiam** (ResNet50) → `experiments/20260704_000002.../run_slurm.sh`
  - 完走（69.2h）はしたが**表現崩壊**。loss -0.87→-0.98（理論下限-1に漸近）、eff_rank が序盤1.2〜1.8まで潰れ、その後も80〜200程度（本来の上限2048に対し4〜10%）までしか回復せず。
  - **原因**: stop-gradient等のロジック自体は正しい実装（`src/sslmodel/models/simsiam.py` / `src/sslmodel/sslutils.py` の `SimSiam.calc_loss` を確認済み、バグなし）。全SSL手法共通で `optim.Adam(..., weight_decay=0)`（`scripts/train/train_tggate.py` の `prepare_model`）を使っており、SimSiam原論文が前提とするweight decay（SGD, wd=1e-4）が欠けていたことが崩壊の主因と推定。
  - **修正**: `train_tggate.py` に `--weight_decay` 引数を追加（デフォルト0、既存実験には影響なし）。`experiments/20260708_132935_tggate_resnet50_simsiam_wd1e4/run_slurm.sh` で `--weight_decay 1e-4` を指定して再学習を用意（Adam/lr等は他手法と揃えたまま）。**要ログインノードからの手動sbatch投入**（コンテナ内からのsbatchは禁止のため）。
- [x] **SwAV** (ResNet50) → `experiments/20260704_000004.../run_slurm.sh`
  - **epoch1・batch1で即クラッシュ**（`CUDA out of memory`、48GB中47GB使用時に392MB確保失敗）。学習は一切進んでいない。
  - **原因**: `MultiCropsTransform`（`src/sslmodel/utils.py`）が1サンプルあたり2枚(224px)+6枚(96px)=**8ビュー**を生成。他手法は2ビューのため、batch=256だと実質フォワード数が他手法の4倍（2048 vs 512）になりOOM。`#SBATCH --mem=64G`はホストRAMでGPU VRAMとは無関係のため、以前のmem増量では解決しなかった。
  - **修正**: `experiments/20260704_000004_tggate_resnet50_swav/run_slurm.sh` の `--batch_size` を256→64に変更（64×8=512ビューで他手法と揃う）。同ディレクトリを直接修正済み（有効な結果がなかったため新規ディレクトリは作らず上書き）。**要ログインノードからの手動sbatch投入**。

### 次回投入コマンド（ログインノードから）
```bash
cd /workspace/andre01/honzawa/wsi-ad
sbatch experiments/20260708_132935_tggate_resnet50_simsiam_wd1e4/run_slurm.sh
sbatch experiments/20260704_000004_tggate_resnet50_swav/run_slurm.sh
```

### Patho-Bench準備（Phase 2、SSL学習と並行）
- [x] `setup_environment.sh` を sbatch で実行（h5py追加 + Patho-Bench clone + venv構築） **完了 2026-07-04**
  - ログ: `outputs/setup_environment_5727.log`、エラーなく完了
  - h5py==3.16.0 を学習環境(.venv)に追加済み、`tools/Patho-Bench/.venv/`（Python 3.12、81パッケージ）構築済み
- [x] HuggingFace でタスクカタログ確認 → H&E系タスクを選定 **完了 2026-07-04**
  - **重要な発見**: Patho-Bench公式カタログ（33データセット/95タスク、`available_splits.yaml`で確認）に**肝臓(LIHC/hepatocellular)タスクは存在しない**。当初計画のTCGA-LIHC優先方針は変更が必要。
  - **方針転換**: 臓器を問わず代表的なタスクを選定し評価する（TG-GATEでの学習→比較が主目的のため）。意味が薄そうならPatho-Bench外の肝臓データセットも検討（保留中の代替案）。
  - **選定タスク**:
    | タスク | データセット | 種別 | 入手方法 |
    |---|---|---|---|
    | `panda--isup_grade` | PANDA（前立腺癌グレーディング） | 分類(6クラス) | Kaggle公開、登録不要・即DL可（最優先） |
    | `cptac_coad--KRAS_mutation` / `TP53_mutation` | CPTAC大腸癌 | 変異予測 | GDCアカウント経由 |
    | `cptac_luad--EGFR_mutation` / `OS` | CPTAC肺腺癌 | 変異予測+生存予測 | GDCアカウント経由 |
    | `bracs--slidelevel_coarse` | BRACS（乳癌サブタイピング） | 分類 | **後回し**。ユーザーがcapsule外（ワークスペース外）に以前DL済みデータを保有。優先度は低、PANDA/CPTAC完了後に着手 |
- [ ] GDCアカウント登録 → CPTAC-COAD / CPTAC-LUAD WSIをダウンロード（旧: TCGA-LIHCから変更）
- [ ] PANDA（Kaggle）WSIをダウンロード
- [ ] BRACSデータ配置（後回し。ユーザーがcapsule外に保有、PANDA/CPTAC完了後に着手）
- [ ] Patho-Bench のタスクラベル・split CSV を取得（`SplitFactory.from_hf`で自動DL可）

### 特徴量抽出・評価（Phase 3、SSL学習完了後）
- [ ] 4手法 × 評価データセット数 分の特徴量抽出を実行
- [ ] Patho-Bench 評価実行
- [ ] 結果まとめ・比較表作成

---

## 学習ジョブの投入方法

GPU は4台あるが他ユーザへの配慮で **最大3台同時使用**とする。
4本目（SwAV）は最初に終わった1本の完了を待って投入する運用（または手動で後から投げる）。

`submit_benchmark.sh` に依存関係付きの一括投入スクリプトを用意してあるが、
個別に投げる場合は以下：

```bash
# ログインノード（コンテナ外）から実行
cd /workspace/andre01/honzawa/wsi-ad

# 3本を即時投入
sbatch experiments/20260704_000001_tggate_resnet50_simclr/run_slurm.sh
sbatch experiments/20260704_000002_tggate_resnet50_simsiam/run_slurm.sh
sbatch experiments/20260704_000003_tggate_resnet50_barlowtwins/run_slurm.sh
# SwAV は上記いずれかの完了後に投入（またはsubmit_benchmark.shで依存設定）
sbatch experiments/20260704_000004_tggate_resnet50_swav/run_slurm.sh

# 投入確認
squeue -u $USER
```

**注意事項**:
- SimSiamは `effective_rank` が1に近づいたら崩壊サイン → WandBで監視
- SwAVは最初の数エポックで損失が不安定になることがある（正常）
- BarlowTwinsの8192-dim cross-correlation行列: bfloat16で~128MB（48GB VRAMで問題なし）
- WandBはオフラインモードで記録 → 後で `wandb sync` が必要

---

## Patho-Bench セットアップ手順

### インストール（SLURMジョブで実行）
```bash
# ログインノードから投入するだけでOK（GPU不要）
sbatch setup_environment.sh
```
内部でやること: `uv add h5py` → `git clone Patho-Bench` → `uv venv && uv sync`（コンテナ内Python 3.12で実行）
- h5py は学習環境(.venv)に追加
- Patho-Bench は `tools/Patho-Bench/.venv/` に分離（pyproject.toml に追記しない。datasets==3.6.0 の固定ピンが競合するため）

### タスク選定（2026-07-04 更新）
- URL: https://huggingface.co/datasets/MahmoodLab/Patho-Bench
- **注意**: 公式カタログ（`available_splits.yaml`、33データセット/95タスク）に肝臓(LIHC)タスクは存在しない。TCGA-LIHC優先方針は撤回。
- 選定タスク（臓器不問、代表性重視）:
  - `panda--isup_grade`（前立腺グレーディング、Kaggle公開・登録不要、最優先）
  - `cptac_coad--KRAS_mutation` / `TP53_mutation`（大腸癌変異予測、GDC経由）
  - `cptac_luad--EGFR_mutation` / `OS`（肺腺癌変異予測+生存予測、GDC経由）
  - `bracs--slidelevel_coarse`（乳癌サブタイピング、後回し。ユーザーがcapsule外に既存データ保有、PANDA/CPTAC完了後に着手）
- 評価タイプ: 線形プローブ、生存予測（Cox、`cptac_luad--OS`のみ）

### WSIデータ取得
```bash
# PANDA: Kaggleから直接DL（登録不要）
# https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data

# CPTAC-COAD / CPTAC-LUAD: GDCポータル経由
# https://portal.gdc.cancer.gov/
# アカウント登録後、gdc-clientでダウンロード
gdc-client download -m gdc_manifest_coad.txt -d /data/cptac_coad/
gdc-client download -m gdc_manifest_luad.txt -d /data/cptac_luad/

# BRACS: 後回し（ユーザーがcapsule外に既存データ保有、PANDA/CPTAC完了後に着手）
```

### 特徴量抽出（SSL学習完了後）
```bash
# モデルごとに実行（例: BarlowTwins、データセットはPANDA/CPTAC等に読み替え）
uv run python scripts/evaluate/extract_features_pathobench.py \
    --wsi_dir /data/panda \
    --model_path result/20260704_000003.../model_ssl.pt \
    --model_name ResNet50 \
    --ssl_name barlowtwins \
    --model_tag resnet50_barlowtwins_tggate \
    --dataset_name panda \
    --output_dir /data/pathobench_features \
    --resume
```

---

## 既存の参考実験（ResNet18、BarlowTwins）

以前のセッションで ResNet18 × BarlowTwins の学習が走っている（または完了している可能性）。
`outputs/20260630_000001〜000005/` 以下を確認すること。
これらは前処理修正**前**の結果なので、ベンチマーク本番とは条件が異なる点に注意。

---

## 注意点・既知の制約

1. **SimCLRのバッチサイズ制約**: 原論文推奨は4096〜8192。256では負例が少なく結果が他手法より劣る可能性。論文では「固定計算資源下での公平な比較」として明記すること。
2. **WandBはオフラインモード**: 学習中はローカル保存 → 終了後 `wandb sync` でアップロード。
3. **Patho-BenchはTrident形式(.h5)を想定**: 独自の `extract_features_pathobench.py` で対応済み。
4. **sbatchはコンテナ内から実行不可**: ログインノードから投入すること。

---

## セッション再開時のチェックリスト

次回のセッション開始時:
1. このファイルを読む
2. `squeue -u $USER` でジョブ状況を確認
3. `ls outputs/20260704_*/` で出力を確認
4. WandBオフラインログを確認: `ls wandb/`
5. 状況に応じて次のPhaseに進む
