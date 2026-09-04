# プロジェクト状況レポート: TG-GATE SSL ベンチマーク

> **重要: このファイルは毎回の作業後に必ず更新すること。**
> 次セッション開始時にはまずこのファイルを読んで状況を把握する。

最終更新: 2026-09-04（セッション10: **ep85 の下流評価を実施 = HANDOFF タスク1 完了**。
480 epoch 完走の根拠は出なかった。次は他3手法の重みを揃えること）

> **崩壊史のグラフ（全6ラン）: https://claude.ai/code/artifact/f4a6e1f1-5b83-4660-a212-40562d82e363**
>
> **0026 の epoch 推移（exp 0027 の結果）: https://claude.ai/code/artifact/2eadc29f-373e-4157-847f-6d25e62d2b80**

> **やること・研究方針の一覧は [TODO.md](TODO.md) を参照**（試験間差除去テーマC章を含む）。

---

## 📊 0026 の表現を epoch 推移で評価（2026-09-04, セッション10, exp 0027 / job 9672）

**HANDOFF タスク1「ep85 の表現は他手法と比較する価値があるか」への回答。
結論: この指標群で見る限り 480 epoch 完走に日数を積む根拠は無い。**

全19チェックポイント（ep5〜ep95）を**同一の2,000パッチ**（fold 0 の val = 事前学習で重みを
更新していない 200 WSI × 10枚、`index.csv` の行番号で決定的に選択、RNG不使用）で埋め込み、
CKA と `lib/analysis/batch_color.py` の η² を測った。

| epoch | 5 | 10 | 20 | 30 | 45 | 60 | 80 | 85 | **90** | **95** |
|---|---|---|---|---|---|---|---|---|---|---|
| η²_slide | 0.661 | 0.785 | 0.712 | 0.685 | 0.651 | 0.645 | 0.638 | **0.639** | 0.398 | 0.393 |
| η²_color | 0.674 | 0.422 | 0.278 | 0.250 | 0.209 | 0.196 | 0.173 | **0.177** | 0.131 | 0.130 |
| cohesion | 0.426 | 0.171 | 0.066 | 0.106 | 0.307 | 0.340 | 0.367 | 0.365 | **0.000** | **0.000** |
| AMI(clust,slide) | 0.207 | 0.315 | 0.366 | 0.390 | 0.415 | 0.411 | 0.438 | 0.428 | 0.066 | 0.062 |

**1. 「バッチは色ではなく構造由来」の直接的な裏づけ（C章の中心仮説）**

色不変性は ep5 の 0.674 から ep80 の 0.173 まで**単調に改善し続ける**のに、スライド依存は
**ep45 の 0.651 から ep85 の 0.639 へ 40 epoch かけて 0.012 しか動かない**（偶然水準 ≈0.07）。
「白黒化してもη²_slideはほぼ下がらない」という以前の観察と同じ結論に、今度は単一手法の
学習軌跡から到達した。しかも `cohesion` (0.066→0.367) と `AMI(clust,slide)` (0.366→0.438) は
**増加**しており、学習が進むほどクラスタ構造はスライドIDに寄っていく。

**2. 表現は収束していない。それでも追加学習の見返りが無い**

隣接 epoch 間の CKA は ep15 以降ずっと 0.98〜0.99 で、ep80→85 でも ep30 付近と同じ速度で
変化している（＝CKAの意味では未収束）。「収束したから止めてよい」ではなく
**「動いてもバッチ問題には効いていないから投資の見返りが薄い」**が正しい読み。
ep85 は ep45 に対し η²_slide でほぼ差が無く（0.639 対 0.651）、優位なのは η²_color だけ。

**3. ⚠️ η²_slide は崩壊検出に使えない（「使ってはいけない指標」に追加）**

ep90 の吸収状態でも η²_slide は 0.398 で、崩壊後の表現でも4割がスライドIDで説明される。
崩壊を明確に示すのは `within_slide_cohesion` **0.365 → 0.000ちょうど**、
`AMI(clust,slide)` 0.428 → 0.066、**隣接CKA 0.984 → 0.166**（ep90→95 は 0.978 で、
崩壊状態が安定＝吸収状態であることも確認できる）。

**この結果が言えないこと**: すべてラベル不要の幾何指標であり、**下流タスク性能が epoch と
ともに伸び続ける可能性は否定できない**。また既存記録の η²_slide 0.55〜0.87 は ResNet 時代の
別パッチ集合の値なので、今回の 0.64 と直接比較はできない。対等な比較には他3手法
（Barlow Twins / MAE / SimSiam）を同一2,000パッチで測り直す必要がある。

### 解析パイプラインの復旧（このセッションで実施）

`lib/analysis/embeddings.py: load_fixed_patches` が `data/shards/*.tar`（webdataset）前提の
まま EXP8 の memmap 移行に取り残されていた問題を解消した（commit `577bd02`）。

- `lib/trainer/data.py` の `load_index_table` / `split_wsi_ids_by_fold` を再利用し、学習と
  同じ fold 分割の val 側だけを使う。**分割関数は必ず import して使うこと**: `wsi_id` は str で
  読まれるため `np.sort` が辞書順になり（4桁ID 99件・5桁ID 901件が混在）、数値ソートで
  書き直すと学習時と違う val 集合になる
- `shard_idx` / `n_patches` を廃し `per_slide` / `fold_idx` / `num_folds` に統一、webdataset 依存を除去
- memmap のサイズ整合チェックを追加（224px を 112px と誤ると行数が4倍で reshape が通り、
  中身だけ静かにずれるため）
- 実験 0027 を新設（`experiments/0027_20260904_repr_dino_epoch_compare`）。extract → compare を
  1ジョブで回す。epoch 掃引の対応表は `scripts/analysis/dino_epochs.yaml`

### この環境で学習・解析を動かすのに必要だったもの

Miyabi 以外の環境（andre01, RTX A6000 × 3）で動かすのに、リポジトリの clone だけでは足りず
以下が必要だった。`.gitignore` 対象なので clone には含まれない。

| 必要物 | 入手方法 |
|---|---|
| `data/ssl_patches/patches.memmap` (140.2GB) + `index.csv` | Miyabi から rsync |
| `outputs/0026_.../model_ep*.pt` (19本, 各832MB) | 同上 |
| `env.sif` (6.2GB) | 同上（`env/env.def` は 01-toxpatho 側とバイト単位で一致） |
| `.venv` | `make uv_sync p=<partition>`（uvのキャッシュが効き7秒で完了） |

---

## 🦕 DINO崩壊調査（2026-08-31 決着, セッション9）

### 結論: 原因は **weight decay を bias / LayerNormゲインにも掛けていたこと**

teacher momentum でも teacher 温度でもなかった。診断ラン 0023/0024 の**両方に共通する
optimizer の設定漏れ**が主因で、`lib/trainer/model.py` が全パラメータを単一 param group
のまま `AdamW(weight_decay=0.04→0.4)` に渡していた。DINO公式 `utils.get_params_groups()`
は `len(shape)==1 or name.endswith(".bias")` を **wd=0 の別グループ**にしている
（MAE・Barlow Twins・SwAV の公式実装も同様）。

**崩壊の機序（すべて実測で確認済み）**

1. LayerNormゲインには wd に対抗する勾配がほとんど無いため、毎step `γ ← γ(1 − lr·wd)`
   で単調に削られる。実測の減衰率は純wd予測 `∏(1 − lr·wd)` と **数%以内で一致**
   （0024はep9以降 grad_norm≈4e-4＝勾配ゼロなので、純粋なwdの効果だけを取り出せた）。
2. backbone出力が入力に依存しない定数に潰れる。
3. `DINOLoss` の center EMA がその定数に収束 → `(t − center) → 0` →
   teacher softmax が**厳密に一様** → 勾配が `1/K − 1/K = 0` で**厳密にゼロ**。
4. loss は ln(8192) = 9.0109 に固定。**吸収状態なので原理的に脱出不可能**。

**チェックポイント実測（`student_backbone.norm.weight` の平均、初期値1.0）**

| epoch | 5 | 10 | 20 | 30 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| 0017 (num_epoch=100) | 0.960 | 0.763 | 0.317 | 0.073 | 0.022 | 0.0085 | **0.0019** |
| 0023 (num_epoch=480) | 0.962 | 0.771 | 0.373 | 0.216 | 0.138 | — | — |
| 0024 (num_epoch=480) | 0.961 | 0.788 | 0.415 | 0.205 | 0.130 | — | — |

**0017「健全に100 epoch完走」は幻だった。** ep100時点で最終LayerNormのゲインは初期値の
0.2%、`blocks.11.norm2` は 0.0004。「train_loss 4.32 でまだ下降中」は、backboneが消えて
いく裏で head が重みを膨張させて（`mlp.0` の重みノルム 24→159）loss だけ下げていた過程。
**`outputs/0017.../model_ep100.pt` は事前学習済みモデルとして使用不可**（0022の継続学習案も無効）。
0017が0023/0024より速く壊れたのは `num_epoch=100` で wd のcosineが 0.4 まで5倍速く
上がるため。**wdスケジュールの速さと崩壊の速さが一致する**ことも独立した裏付けになっている。

### 診断2軸（0023/0024）の正しい位置づけ

| ラン | 早期崩壊 | 回復 | 最終 |
|---|---|---|---|
| 0023 (momentum 0.996) | ep4 | **ep7に回復**、ep41でloss 1.7228まで低下 | **ep42に突然崩壊 → ep43から9.0109固定** |
| 0024 (teacher温度warmup) | ep9 | **回復せず** | 36 epoch 9.010x のまま |

- 0023 の ep42 の崩壊は **lr一定・grad_norm 0.83→0.55（スパイク無し）** の平穏な状況で
  起きており、「warmupのlr上昇＋クリッピング欠如で吹き飛ぶ」では説明できない。
- **teacher温度**: sharpening（一様解から離れる唯一の力）を弱めるので、
  「早期崩壊から回復できるか」を決める。0024が回復できなかった理由。
- **teacher momentum**: 崩壊の**タイミング**のみ。副作用として、EMA teacherは
  「減衰中の重みの古いコピー」なので teacher/student にスケール差が出る
  （0024: teacherのLayerNormゲインが最大 **+23%**、0023: 3%以内）。
- どちらも恒久崩壊の主因ではない。

### 実施した修正（2026-08-31）

| # | 修正 | 場所 |
|---|---|---|
| 1 | **bias / ndim<=1 を weight decay から除外**（既定ON。`--wd_apply_to_bias_norm` で従来挙動） | `lib/trainer/model.py:_wd_groups` |
| 2 | wdコサインスケジュールが除外グループを上書きしないようガード | `lib/trainer/loop.py` |
| 3 | **勾配クリッピング `--clip_grad`**（DINO公式と同じパラメータ毎クリップ、公式値3.0。既定0=無効） | `lib/trainer/loop.py:train_epoch` |
| 4 | **LayerNormゲイン平均 `ln_gain` を毎epochログ出力**（今回の主因を即座に検知できる） | `lib/trainer/loop.py:norm_gain_mean` |
| 5 | 崩壊判定を3指標のORへ（`train_loss ≥ ln(out_dim)×0.999` / `uniformity > −0.05` / eff_rank） | `lib/sslmodel/utils.py:CollapseMonitor` |
| 6 | 旧 `state.pt` からの `--resume` は原因を明示して停止（param group構成が変わったため） | `lib/trainer/entry.py` |
| 7 | 回帰テスト追加（torch不要・純Python、19項目。`make test` に組込済） | `tests/test_collapse_guards.py` |
| 8 | **`--dino_out_dim`**（公式65536。既定8192は据え置き） | `lib/trainer/entry.py` / `sslutils.py` |
| 9 | **`--dino_drop_path`**（公式0.1のstochastic depth。studentのみに適用） | `lib/sslmodel/models/dino.py` |
| 10 | `DINOLoss` の `log_softmax` 重複計算を解消（teacherビューごとに再計算していた → student ビューごとに1回。out_dim 65536 では約0.7GB/回のメモリ直撃） | `lib/sslmodel/models/dino.py` |
| 11 | 特徴抽出時の `out_dim` をチェックポイント形状から復元（65536のランを読めるように） | `sslutils.py:DINO.prepare_featurize_model` |

**⚠️ `eff_rank` は崩壊検出に使えない。** 0023 ep5 は loss=ln(8192)ちょうど・feat_std 0.055
の壊滅状態で eff_rank 31.99、0024 は36 epoch崩壊し続けても発火しなかった。
また **`alignment` は Wang & Isola の alignment *loss*（小さいほど良い）** であり、
崩壊時も0に近づくので単独では判定に使えない（2026-08-29のログで誤読していた）。
崩壊の一次指標は **`train_loss → ln(out_dim)`** と **`uniformity → 0`**。

### 他手法への影響

| 実験 | optimizer | wd | 影響 |
|---|---|---|---|
| 0017/0021/0023/0024 DINO | adamw | 0.04→0.4 | **致命的（実測確認）。全ラン破棄** |
| 0013/0016/0018 MAE | adamw | 0.05 | **影響あり**。純wd予測でep100にゲイン ~0.22（未実行なので損失は無し） |
| 0014 SimSiam | sgd | 1e-4 | 影響あり（`fix_pred_lr` 分岐も同じ穴だった）。ただしSimSiam原論文のResNetレシピはbias/BNを除外しない。ViT-Bで走らせる以上はMoCo v3準拠（除外）が妥当 |
| 0015 Barlow Twins | lars + `--lars_exclude_bias_bn` | 1.5e-6 | **影響なし**（元から除外済み・wdも極小） |
| 20260704_* の旧ResNetラン | adam wd=0 | 0 | 影響なし |

DINO以外が致命傷にならないのは、**厳密に勾配ゼロの吸収状態（loss = ln K）を持つのが
DINOの目的関数だけ**だから。MAEの画素再構成やBTの冗長性低減はゲインが小さくなっても
勾配が残るので、消滅ではなく平衡に落ち着く（実際 Barlow Twins は健全に完走している）。

### 過去に判明した2つの事故（記録・再発防止）

**1. DDP forward のバグ修正が stash に巻き込まれて消えていた（2026-08-29 復元, commit c782e58）**

`DINO.calc_loss` が DDPラップ済みモデルに対して `model.forward_student(views)` と直接呼んで
おり、DDPは任意属性を `.module` へ転送しないため AttributeError で即死する。0017 は
2026-08-10 に実際にこれで2回失敗している (job 2513399, 全rank)。Aug10→11に未コミットで
修正され0017は完走したが、8/22に「0021 paper-faithful」をstashした際にこの修正まで巻き添えで
退避され、ツリーが壊れた状態に戻っていた。**教訓: 修正とスケジュール変更を同じ未コミット
ツリーに混ぜない。**

**2. 0021の実装が部分的に失われており、何を検証したのか確定できない**

0021の `config.json` には `momentum_start: 0.996` 等が記録されているが、それを適用する側の
コード（entry.pyのCLI引数・loop.pyの毎step更新呼び出し）がツリーにもstashにも存在しなかった。
2026-08-29に独立トグルとして実装し直し (commit c9e39bb)、毎epochのログに
`teacher_momentum` / `teacher_temp` の実測値を出すようにしたので、今後は
「有効にしたつもりで固定のまま」を必ず検出できる。

### 0025 の結果（2026-08-31, job 3271936）— wd修正は成功、別要因の崩壊が残存

**weight decay の修正は効いた（確認済み）**

```
weight decay 0.04->0.4: decayed 55 tensors / exempt (bias & ndim<=1) 102 tensors
```

| epoch | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| **0025（修正後）の ln_gain** | 1.0003 | 0.9866 | 0.9951 | **0.9955** |
| 0023（修正前）の ln_gain | 0.9620 | 0.7711 | 0.5379 | **0.3731** |

LayerNormゲインの侵食は完全に停止。head の重み膨張も無し（`mlp.0` の重みノルムが
0023では 24→101→159 と暴走したのに対し 0025 は 25→35→25）。**恒久崩壊の主因は解消。**

新しい崩壊検知も設計どおり動作し、ep19 で abort した:

```
Collapse detected for 2 consecutive checks at Epoch: 19 — aborting to save compute
[train_loss 11.0904 >= ln(out_dim) x 0.999 (= 11.0793) — uniform collapse;
 uniformity -0.0000 > -0.05 — all samples collapsed onto one point]
```

旧 `eff_rank` 基準（ep20時点で21.24）なら発火せず48h回し切っていた。

**しかし別の崩壊が残っている**

ep11→12 で train_loss が ln(65536)=11.0904 に張り付き、grad_norm 1e-4 /
feat_std 0.018 / uniformity 0.0000。**このとき ln_gain は 0.9692 で固定されたまま**
なので weight decay 起因ではない別要因。loss は ep1-11 で
10.79 / 8.74 / 10.81 / 10.65 / 10.03 / 10.62 / 10.11 / 9.47 / 9.08 / 10.21 / 10.42 と
振動するだけで、**一度も下降トレンドに乗らなかった**（0023 は out_dim 8192 で
ep2 に 4.55 まで下降し、最終 1.72 まで到達していた）。

`outputs/0025.../model_ssl.pt` は abort 時に復元された checkpoint.pt
（val最良 = ep9, val_loss 8.60）で、事前学習済みモデルとしては使用不可。

**公式 main_dino.py の help に該当する記述**（2026-09-01 に確認）

| 引数 | 公式の help | 0025 の設定 |
|---|---|---|
| `clip_grad` | 「Clipping with norm **.3 ~ 1.0** can help optimization for **larger ViT architectures**」 | **3.0**（argparse既定）。実測 grad_norm 0.25〜3.15 なので**ほぼ発火していなかった** |
| `freeze_last_layer` | 「Try increasing this value **if the loss does not decrease**」 | 1（公式既定）。0025 の症状そのもの |
| `use_fp16` | 「loss が不安定なとき、**大きいViTを使うとき**は mixed precision を切ることを推奨」 | bf16 autocast。bf16 は fp16 より仮数部が3bit少なく、centering の `t - center` に不利 |
| `out_dim` | 「**複雑で大規模なデータセット**では 65k のような大きい値が良い」 | 65536。TG-GATE の H&E パッチは ImageNet ほど多様ではない |

### 0026 の結果（2026-09-01, job 3277479）— clip_grad 0.3 で崩壊が止まった

**0025 からの変更は `--clip_grad 3.0 → 0.3` の1点のみ。これで崩壊しなくなった。**

```
ep  1: 10.7873  gn 2.01  ln_gain 1.0001
ep  5:  9.3731  gn 1.41  ln_gain 0.9942   eff_rank 120.19  feat_std 0.7258
ep 10:  5.3949  gn 1.30  ln_gain 0.8978   eff_rank 117.91  feat_std 0.2734
ep 15:  3.4830  gn 1.08  ln_gain 0.8572   eff_rank 195.46  feat_std 0.1996
ep 20:  2.8539  gn 0.88  ln_gain 0.8095   eff_rank 237.18  feat_std 0.1545  val_loss 3.8944
```

| | 0025（clip 3.0） | 0026（clip 0.3） |
|---|---|---|
| ep11 の loss | 10.42 → **ep12で11.0904に張り付き** | 5.17 → 順調に下降 |
| ep20 の loss | 11.0904（崩壊） | **2.8539** |
| ep20 の grad_norm | 1e-4（勾配消失） | 0.88（正常） |
| ep20 の eff_rank | 21.24（低下） | **237.18（上昇）** |

`eff_rank` が上昇し、val_loss も 3.89 まで下降。本プロジェクトで最良の軌道
（0023 の ep20 は loss 6.59 で、しかも当時は崩壊コース上だった）。
`ln_gain` の 1.00→0.81 は weight decay ではなく勾配由来（wd除外済みなので
純wd予測は 1.0 のまま）。学習が実際に進んでいる証拠。

**根拠**: 公式 main_dino.py の help「Clipping with norm **.3 ~ 1.0** can help
optimization for **larger ViT architectures**」。0025 で使った 3.0 は argparse の
既定値であって ViT-B 向けの推奨値ではなく、実測 grad_norm 0.25〜3.15 に対して
**ほぼ一度も発火していなかった**（＝実質クリッピング無しで走っていた）。

### ⚠️ 崩壊検知の誤検知で ep19 に abort させてしまった（2026-09-01, 修正済み commit 6dd33a9）

上記の健全なランを `uniformity -0.0108 > -0.05` だけで止めた。

| 指標 | ep20 | 判定 |
|---|---|---|
| `train_loss` | 2.8539 | 天井 ln(65536)=11.0904 の **26%** → 該当せず |
| `effective_rank` | 237.18 | 閾値5.0の遥か上 → 該当せず |
| `uniformity` | −0.0108 | 閾値 −0.05 を超過 → **これだけで発火** |

`uniformity` はヘッドの生出力(out_dim次元)に対して計算されるため**尺度が out_dim に
強く依存する**。65536本のプロトタイプは256次元ボトルネックに詰まっているので
サンプル間のロジットは構造的にほぼ平行になり、健全でも −0.01 程度まで0に寄る。
閾値 −0.05 は out_dim 8192 のログから決めた値で、65536 にした時点で無効になっていた。

**修正**: 補助指標(uniformity / effective_rank)は「損失が天井の90%以上」のときだけ
有効にする。一次指標 `train_loss >= ln(out_dim)` は代数的事実に基づきout_dim非依存に
効くので単独発火のまま。**out_dim や損失定義を変えたら uniformity の閾値は
必ず較正し直すこと。**

計算資源のロスは無し。`state.pt` に ep20 の状態（重み・optimizer・scheduler・
early_stopping）が残っていたので、`model_ssl.pt` を `model_ssl_aborted_ep20.pt` へ
退避して ep20 から再開した（job 3280970）。

### 0026 の最終結果（2026-09-02, job 3280970）— ep90 で崩壊

`clip_grad 0.3` は **ep12 → ep89 まで寿命を伸ばした**が、完走には至らなかった。

```
ep 50: loss 2.0612  gn 0.84   ← 安定
ep 65: loss 2.0136  gn 1.19
ep 66: loss 5.3178          ← 1回目の警告。自力で回復した
ep 80: loss 1.5993  gn 1.09
ep 85: loss 1.6132  gn 5.67  ← 勾配ノルムが急上昇
ep 88: loss 1.6051  gn 5.98
ep 89: loss 6.4304          ← 破綻
ep 90: loss 11.0904 gn 0.0002 ← 吸収状態。ep94 で abort
```

**前回(0025)とは機序が違う。** クリップ前の勾配ノルムが ep57 の 0.9 から ep88 の 6.0 へ
**30 epoch かけて単調増大**し、lr 1.87e-3 の高原部で臨界を越えた。`ln_gain` 0.605 で安定、
`eff_rank` 318 と、**崩壊直前まで表現自体は健全**だった（weight decay 由来ではない）。

**得られたもの**: `model_ep85.pt` は train_loss 1.61 / val_loss 4.17 / eff_rank 318 で、
**本プロジェクト初のまともな DINO 表現**。崩壊前スナップショットは ep85 まで5 epoch刻みで残っている。

**⚠️ 論文設定そのものは、このデータでは成立していない。**
`clip_grad` は 0025 で論文既定の 3.0 を使って ep12 で死んでおり、0026 の 0.3
（公式が「大きいViTには .3〜1.0」と書いている範囲）に外したことで ep90 まで伸びた。
「論文準拠だから崩壊した」ではなく「**論文から外したから伸びた**」が正しい。

### 次の一手

**→ 詳細な手順は [docs/HANDOFF_dino_next.md](docs/HANDOFF_dino_next.md) に分離した（別環境で作業する人向けの自己完結メモ）。**

1. **最優先: `model_ep85.pt` の下流評価**（別環境で実施）。480 epoch の完走に日数を積む前に、
   「この表現は他手法と比較する価値があるか」を先に判定する。
   ⚠️ 解析パイプラインは `.tar` シャード前提で、いまの学習データ（memmap）を読めない。要修正。
2. 安定化ランを4ノード×2本で並列（A: peak lr を 2e-3 → 1e-3 / B: ヘッドと損失を fp32）。
   B は `--dino_fp32_head` として実装済み（2026-09-03）。run_slurm.sh に足すだけで投入できる。
3. MAE（0013/0016/0018）と SimSiam（0014）は未実行のまま。wd 修正が効くので投入してよい。
4. 0017/0021/0023/0024/0025 の重みは事前学習済みモデルとして使用不可。

#### 2026-09-03 の作業（下流評価の準備）

- `scripts/analysis/methods_paper.yaml` の **DINO 行を `0026/model_ep85.pt` に差し替えた**。
  旧値は崩壊済みラン（`20260714_paper_dino_vitb16/model_ssl.pt`）を指していた。
  `lib/model/zoo.py: prepare_model_eval` で読めること（out_dim 65536 を形状から復元し、
  768次元の pooled 特徴が出る）を実機確認済み。
- **ほかの3手法（barlowtwins / mae / simsiam）は Miyabi では読めない**ことを確認した。
  `/workspace/andre01/...` という旧クラスタの絶対パスのままで、`/workspace` 自体が存在しない。
  `embeddings.run` は見つからないチェックポイントを `[skip]` して続行するので、
  **このまま回すと DINO 単独の解析になる**（沈黙して4手法比較にならない）点に注意。
- CLI に `--dino_fp32_head`（ヘッドと損失だけ autocast を外す）と
  `--dino_freeze_last_layer`（公式 `--freeze_last_layer` と同義・既定1）を追加した。
  どちらも既定は従来挙動のままなので、既存ランとの比較可能性は保たれる。
  テスト: `lib/sslmodel/tests/test_dino_fp32_head.py`（コンテナ内で `python -m unittest`）。

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

### 公式(main_dino.py, `--arch vit_base` 既定)との照合 — 2026-08-31 に公式リポジトリで確認

**公式のViT-B/16レシピは「`--arch vit_base` を指定するだけの全部デフォルト、2ノード×8GPU×batch 64 = global batch 1024」**。
本プロジェクトの global batch 1024 と同一条件。

| 項目 | 公式 | 本プロジェクト | 状態 |
|---|---|---|---|
| `out_dim` | 65536 | 8192 → **65536** | 0025で是正 |
| `clip_grad` | 3.0 | 無し → **3.0**(0025) → **0.3**(0026) | ◆ **論文は言及なし**（コード固有の軸） |
| `momentum_teacher` | 0.996 → cosine 1.0 | 0.9995固定 → **0.996→1.0** | 0025で是正 |
| `drop_path_rate` | 0.1 | 0 → **0.1** | 0025で是正 |
| `teacher_temp` | **0.04固定**(`warmup_teacher_temp_epochs=0`) | 0.04固定 | ◆ **論文本文とは食い違う**（下記） |
| `local_crops_number` | 8 | 8 | ◆ **論文 Appendix E は 6**（下記） |
| lr / warmup / min_lr / wd / optimizer / batch / freeze_last_layer / norm_last_layer / use_bn_in_head / student_temp | — | — | ✅ 一致 |
| `use_fp16` | True (fp16+GradScaler) | bf16 autocast | ✅ 実質同等以上 |
| `global_crops_scale` / `local_crops_scale` | (0.4,1.0) / (0.05,0.4) | (0.2,1.0) / (0.05,0.2) | ⚠️ **意図的な逸脱**（下記） |
| lr/wdスケジュール粒度 | iteration | epoch | ⚠️ schedulerが全手法共通のため据え置き |

**認識の訂正（重要）**

- **teacher温度: 論文本文と公式コードが食い違っている**（2026-09-04 に論文本体で再確認し、
  それまでの記述を訂正）。論文 §3.2 Implementation details は
  「The temperature τs is set to 0.1 while we use a **linear warm-up for τt from 0.04 to
  0.07 during the first 30 epochs**.」と明記しており、アーキテクチャ別の但し書きは無い。
  一方リリースされたコードの既定は `teacher_temp=0.04, warmup_teacher_temp_epochs=0`
  ＝ **0.04固定**で、公式ヘルプも「0.07 を超えると多くの実験で不安定。既定の 0.04 から
  始めることを推奨」としている。本プロジェクト(0017/0025/0026)は 0.04固定＝**コード側**。
  **したがって「0024 は論文準拠ではなく逸脱だった」という以前の記述は誤り。**
  正しくは **0024 は論文本文には忠実で、公式コード既定から逸脱していた**。
  0024 の崩壊は「論文が書いている設定がこのデータでは成立しない」証拠として読むべきもので、
  0025/0026 で得た「論文から外したから伸びた」という結論と同じ向きを指している。
  ⚠️ 論文に書くときは「論文準拠」と一語で済ませず、**論文本文準拠かコード既定準拠かを
  明示**すること。この2つは teacher温度・local crops数・clip_grad の3点で食い違う。
- **momentum 0.9995 は batch 256 向けの値。** 公式ヘルプの原文は
  「小さいバッチではより高い値を推奨。例えば batch 256 では 0.9995」。
  本プロジェクトは global batch 1024 なので**既定の 0.996 が論文設定**にあたる。
  0023 が loss 1.72 まで順調に下がったことも 0.996 が悪者でなかった裏づけ。
- **「out_dim が小さい方が崩壊しにくい」も誤り**（コード内コメントを修正済み）。
  プロトタイプ数が多いほどサンプルを散らせるので一様解に落ちにくい。

**◆ 論文本文と公式コードが食い違う3点**（2026-09-04、論文 arXiv:2104.14294 §3.1-3.2 /
Appendix C・E と `main_dino.py` を突き合わせて確認）

| 項目 | 論文本文 | 公式コード既定 | 本プロジェクト |
|---|---|---|---|
| `teacher_temp` | 0.04 → 0.07 の線形warmup(30 epoch) | **0.04固定** | 0.04固定（コード側） |
| local crops 数 | **6**（Appendix E） | 8 | 8（コード側） |
| `clip_grad` | **言及なし** | 3.0 | 3.0(0025) / 0.3(0026) |

`clip_grad` は論文に存在しないパラメータなので、**0026 が動かしたのは論文には無い軸**に
あたる。crop scale も論文は `(s,1)` / `(0.05,s)` と s を可変パラメータとして書くだけで
固定値を置いておらず、コード既定の s=0.4 が実質的な基準値になっている。

**意図的な逸脱（論文に明記すること）**: augmentation。公式の crop scale
(0.4,1.0)/(0.05,0.4) に対し本プロジェクトは (0.2,1.0)/(0.05,0.2)、回転 p=1.0、
grayscale 0.05、solarization無効。224pxパッチ済みなので過度なcropは微細構造を壊す・
H&Eの色は診断的に重要、という病理ドメイン由来の判断（`lib/sslmodel/utils.py:ssl_transform`
に根拠コメント）。かつ Goal.yaml が手法間の拡張統一を要求しているため、
**DINOだけ論文値に戻してはいけない。**

### 崩壊の見分け方（2026-08-31 更新）

- **一次指標**: `train_loss` が **ln(out_dim)**（out_dim 8192 なら 9.0109）に張り付く／
  `uniformity` が 0 に近づく（全サンプルが1点に潰れた）／`grad_norm` → 0。
- **`ln_gain`（毎epochログ）が初期値1.0から単調に下がり続けていたら weight decay の
  param group 設定を疑う。** 今回の主因はこれで、崩壊するずっと前から検知できた。
- **使ってはいけない指標**:
  - `eff_rank` は鈍すぎる。0023 ep5 は loss=ln(8192)ちょうど・feat_std 0.055 の壊滅状態で
    eff_rank 31.99。0024 は36 epoch崩壊し続けても閾値5.0に掛からなかった。
  - `alignment` は Wang & Isola の alignment **loss**（正例ペア間の距離）で **小さいほど良い**。
    崩壊時も0に近づくため単独では判定不能（2026-08-29のログで誤読した）。
- **一時的な低下だけで崩壊と即断しない**: 0017はep25で eff_rank 6.04 まで落ちてから回復し、
  0023もep4で崩壊してep7に回復している。ただし**一度 loss が厳密に ln(out_dim) で固定され
  grad_norm が 1e-4 台になったら吸収状態＝回復しない**。

### 生ログ（0023 / 0024、参考）

| epoch | 0023 (momentum 0.996) | 0024 (温度warmup) |
|---|---|---|
| 1 | 8.4604 / gn 1.99 | 8.9436 / gn 0.76 |
| 2 | 4.5532 / gn **5.42** ⚠️ | 8.7132 / gn 0.82 |
| 4 | **9.0109** 崩壊 / gn 0.005 | 8.9059 / gn 0.13 |
| 7 | 8.9576 回復開始 / gn 0.09 | 8.7700 / gn 0.15 |
| 9 | 7.9934 回復継続 / gn 0.34 | **9.0108** 崩壊 / gn 0.002 |
| 30 | 2.9739 / gn 0.77 | 9.0108 / gn 0.0004 |
| 41 | **1.7228** / gn 0.83 | 9.0102 / gn 0.0005 |
| 42 | 4.6083 / gn 0.55 | 9.0106 / gn 0.0019 |
| 43-44 | **9.0109 固定** / gn 0.0002 | 9.0109 固定 / gn 0.0014 |

スケジュール配線が実際に効いていたことは確認済み（0024の teacher_temp が ep5 で 0.0450 =
理論値 `0.04 + 0.03 × (4×781+780)/(30×781)` と一致、0023の momentum は 0.996 から微増、
それぞれ他方は固定）。0021で疑われた「有効にしたつもりで固定のまま」ではない。

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
