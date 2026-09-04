# 引き継ぎ: DINO の下流評価と安定化（2026-09-03）

> 別環境で作業する人向けの自己完結メモ。Miyabi 上の経緯は
> [PROJECT_STATUS.md](../PROJECT_STATUS.md)「🦕 DINO崩壊調査」を参照。
> 崩壊史のグラフ: https://claude.ai/code/artifact/f4a6e1f1-5b83-4660-a212-40562d82e363

## 3行まとめ

- DINO ViT-B/16 の事前学習は 6ラン中 6ランとも崩壊した（損失が `ln(out_dim)` に張り付き勾配が厳密にゼロになる吸収状態）。
- 主因は **weight decay を bias / LayerNorm ゲインにも掛けていた実装漏れ**で、これは修正済み（commit `581e53d`）。
- 修正後の 0026 は **ep85 で train_loss 1.61 / eff_rank 318** という本プロジェクト初のまともな DINO 表現に到達したが、ep90 で別要因により崩壊した。

---

## タスク1（最優先・別環境向け）: ep85 の表現を下流評価する

**目的**: 480 epoch の完走に日数を積む前に、「ep85 の表現は他手法と比較する価値があるか」を先に判定する。
本来の目的は DINO を完璧に学習させることではなく**手法比較**なので、ep85 で十分なら epoch 予算の前提自体を見直せる。

### 使うチェックポイント

```
outputs/0026_20260901_dino_clipgrad03/model_ep85.pt      832MB
```

- 中身は `lib/sslmodel/models/dino.py: DINO` の `state_dict()`（student + teacher の両方、466キー）。
- 特徴抽出に要るのは `student_backbone.*`（150キー、timm `vit_base_patch16_224`、`num_classes=0` の pooled 出力 768次元）だけ。
  転送量を減らすなら下記で剥がせる。

```python
import torch
sd = torch.load("model_ep85.pt", map_location="cpu")
torch.save({k[len("student_backbone."):]: v for k, v in sd.items()
            if k.startswith("student_backbone.")}, "dino_ep85_backbone.pt")
```

- **out_dim は 65536**（`student_head.last_layer.weight_v` が `(65536, 256)`）。
  `lib/sslmodel/sslutils.py: DINO.prepare_featurize_model` はチェックポイントの形状から
  out_dim を復元するので、そのまま `model_path` に渡せば読める（2026-08-31 に修正済み）。

### 比較対象

`scripts/analysis/methods_paper.yaml` に主比較4手法（Barlow Twins / DINO / MAE / SimSiam）の対応表がある。

- **DINO の行は上記 ep85 に差し替え済み**（2026-09-03）。`model_path` は
  リポジトリルートからの相対パス `outputs/0026_20260901_dino_clipgrad03/model_ep85.pt`。
  `lib/model/zoo.py: prepare_model_eval` 経由で読めること（466キー / out_dim 65536 が
  形状から復元され、768次元の pooled 特徴が出ること）を実機で確認済み。
- ⚠️ **ほかの3手法は現状すべて `[skip]` される。** `model_path` が別クラスタ（andre01）の
  絶対パス `/workspace/andre01/honzawa/wsi-ad/outputs/...` のままで、Miyabi には
  `/workspace` 自体が存在しないことを確認した（2026-09-03）。
  `embeddings.run` は存在しないチェックポイントを黙って飛ばして続行するので、
  **このまま回すと「4手法比較」ではなく DINO 単独になる**。実行後は
  `extract_summary.json` の `embeddings` に何が入ったかを必ず確認すること。
- したがって当面は DINO 単独評価（ep20 / ep50 / ep85 の推移比較）から始めるのが現実的。
  その場合は `methods_paper.yaml` を複製し、`name` を `dino_ep20` などに分けて
  同じ `ssl_name: dino` で3行並べればよい（`name` は出力ファイル名 `emb_{name}.npy` になるだけ）。

### 実行

```python
from lib.analysis import embeddings, compare
embeddings.run("scripts/analysis/methods_paper.yaml",
               data_dir=..., n_patches=2000,
               output_dir="outputs/representation_analysis")
compare.run("outputs/representation_analysis", k=10)
```

`compare.run` が CKA 行列 / k-means の ARI・NMI / クラスタ別プロトタイプパッチ / t-SNE を出す。

### ⚠️ 既知のブロッカー（先に潰す必要あり）

**解析パイプラインが読むデータ形式が、いまの学習データと食い違っている。**

| | 形式 | 場所 |
|---|---|---|
| 学習（`lib/trainer/data.py`） | `patches.memmap` + `index.csv`（uint8, shape `(N, 224, 224, 3)`） | `data/ssl_patches/` (141GB) |
| 解析（`lib/analysis/embeddings.py: load_fixed_patches`） | **webdataset の `.tar` シャード** | `data/shards/`（**存在しない**） |

`load_fixed_patches` は EXP8 の memmap 移行より前に書かれたもので、そのままでは動かない。
対応は「`load_fixed_patches` を memmap 版に書き換える」のが素直（`lib/trainer/data.py` の
`_read_index` / memmap Dataset がそのまま流用できる）。**全手法で同一パッチ集合・同一順序を使う**
のがこの比較の前提なので、`index.csv` の row 番号で決定的に選ぶこと。

---

## タスク2: 安定化ラン（Miyabi 側、2本並列を推奨）

0026 は ep57 の 0.9 から ep88 の 6.0 へ**クリップ前の勾配ノルムが30 epoch かけて単調増大**し、
lr 1.87e-3 の高原部で破綻した。`ln_gain` 0.605 / `eff_rank` 318 と、崩壊直前まで表現自体は健全だった。

0023 / 0024 と同じく **4ノード×2本を同時投入して1軸ずつ切り分ける**。ベースは
`experiments/0026_20260901_dino_clipgrad03/run_slurm.sh`。

### A: peak lr を半分（本命）

```diff
-    --optimizer adamw --lr 5e-4 --ddp_linear_scale_lr    # 実効 2e-3
+    --optimizer adamw --lr 2.5e-4 --ddp_linear_scale_lr  # 実効 1e-3
```

根拠: 崩壊が lr 高原で起きており、定常 lr のもとで勾配が単調増大している。論文の 2e-3 は
ImageNet-1k での linear scaling 値だが、本データは **800枚のWSI由来80万パッチ**で多様性が桁違いに低い。
論文からの逸脱になるので、論文には「安定性のため」と明記する。

### B: 損失計算を fp32 にする（実装済み・投入待ち）

公式 `main_dino.py` の `use_fp16` help に「**loss が不安定なとき、大きい ViT を使うときは
mixed precision を切ることを推奨**」とある。現状は bf16 autocast（仮数部8bit）で、DINO の損失は
`t - center` という**近い値どうしの差**を取ってから `÷0.04` で25倍に増幅する（桁落ち）。

backbone は bf16 のまま、**ヘッドと損失だけ fp32** にすれば速度低下は小さい。

**→ `--dino_fp32_head` として実装済み（2026-09-03）。** run_slurm.sh の
`RUN_COMMAND` にこのフラグを足すだけでよい（既定は従来どおり off なので、
付けなければ既存ランと同一挙動）。

```diff
     --dino_out_dim 65536 --dino_drop_path 0.1 \
+    --dino_fp32_head \
```

実装は `DINO._forward_views` でヘッドを `torch.amp.autocast(enabled=False)` の
スコープに入れ、`DINOLoss` 側も同じスコープで計算する（`lib/sslmodel/models/dino.py`）。
テストは `lib/sslmodel/tests/test_dino_fp32_head.py`。

補足（実測）: timm ViT の末尾は LayerNorm で、LayerNorm は autocast の fp32 ポリシー
対象なので **backbone の出力はフラグに関係なく元から fp32**。実際に効くのは
ヘッド内の Linear（bf16 → fp32）と、`t - center` の引き算のほう。backbone 内部の
行列積は bf16 のままなので速度への影響は小さい。

### 3番手以降（A・Bが効かなかったら）

1. `out_dim` を 8192 に戻す（公式 help は 65k を「複雑で大規模なデータセット向け」と条件付きにしている）
2. `--dino_freeze_last_layer 3`（公式の助言は「loss が下がらないとき」向けで、今回の症状とは合わない）
   **→ CLI に公開済み（2026-09-03）。** 公式 `main_dino.py` の `--freeze_last_layer` と
   同義・同既定（1）。epoch は0始まりなので `3` を渡すと epoch 0/1/2 の3エポック分、
   ヘッド最終層の勾配を捨てる。

---

## タスク3: MAE / SimSiam の投入（未実行）

`experiments/0013 / 0016 / 0018`（MAE）と `0014`（SimSiam）は**一度も実行されていない**。
weight decay の param group 修正はこれらにも効くので、修正後の設定で投入すればよい。

- MAE: adamw wd 0.05 で、純wd予測ではゲインが ep100 に初期値の 0.22 まで削られる計算だった（修正済み）
- SimSiam: `fix_pred_lr` 分岐も同じ穴だった（修正済み）。原論文の ResNet レシピは bias/BN を除外しないが、
  本プロジェクトは ViT-B なので MoCo v3 準拠（除外）でよい。原論文どおりにするなら `--wd_apply_to_bias_norm`
- Barlow Twins（0015）は LARS の除外フラグで元から影響なし

---

## 監視のルール（次に学習を回す人へ）

| 指標 | 見方 |
|---|---|
| `train_loss` | **`ln(out_dim)` に張り付いたら死**（8192 → 9.0109 / 65536 → 11.0904）。勾配が厳密に0の吸収状態なので回復しない |
| `grad_norm` | `1e-4` 台に落ちたら崩壊済み。逆に**数エポックかけて増大していたら破綻の前兆** |
| `ln_gain` | 毎epochログに出る。1.0 付近から**単調に下がり続けたら weight decay の設定を疑う**（`lib/trainer/model.py:_wd_groups`） |
| 起動直後 | stdout の `exempt (bias & ndim<=1) M tensors` の **M が 0 なら除外が効いていない** |
| `eff_rank` | **使わない。**0024 は36 epoch 崩壊し続けても閾値5.0に掛からなかった |
| `alignment` | Wang & Isola の alignment **loss**。**小さいほど良い。**崩壊時も0に近づくので単独では判定不能 |
| `uniformity` | 0 に近いほど悪いが、**尺度が out_dim に依存する**。out_dim を変えたら閾値を較正し直すこと |

`--resume` の落とし穴: abort や walltime 切れで `model_ssl.pt` が書かれていると
「training complete, nothing to resume」で即終了する。継続したいときは退避してから再投入する。
また **2026-08-31 より前の `state.pt` からは再開できない**（param group 構成が変わったため、
`lib/trainer/entry.py` が理由を明示して停止する）。

---

## 関連コミット

| commit | 内容 |
|---|---|
| `581e53d` | 原因修正（wd の param group 分離）＋ clip_grad / ln_gain ログ / 崩壊検知の刷新 |
| `99feeeb` | exp 26（clip_grad 0.3） |
| `6dd33a9` | 崩壊検知の誤検知修正（補助指標を単独発火させない） |
| `9a091ae` | `--dino_fp32_head` / `--dino_freeze_last_layer` の追加、`methods_paper.yaml` の DINO 行差し替え |
