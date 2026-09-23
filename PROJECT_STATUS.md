# プロジェクト状況レポート: TG-GATE SSL ベンチマーク

> **重要: このファイルは毎回の作業後に必ず更新すること。**
> 次セッション開始時にはまずこのファイルを読んで状況を把握する。

> ## ⚠️ 実行環境の使い分け（2026-09-05 決定）
>
> **Miyabi は従量課金。ここでは実行を極力避ける。**
> **学習のみ Miyabi、評価・解析はすべて andre01（ユーザーの計算機）で行う。**
>
> 特徴抽出・表現比較（CKA / η²）・下流評価（線形プローブ / ABMIL / 用量反応）は
> Miyabi では回さない。コードは書いてコミットするところまでがこちらの仕事で、
> 実行は向こう。**コミットに「実装済み・未実行」が含まれるのは想定どおり。**
> 詳細は [CLAUDE.md](CLAUDE.md)。

最終更新: 2026-09-23（セッション13: **exp 0028 が 480 epoch を完走した**。崩壊せず、
投入前に固定した判定基準も満たしたので **lr 仮説は支持された**。
途中 `uv run` による `.venv` 破壊で1本無駄にしたが、復旧手順と再発防止を実装済み）

> ## 🚨 いま最初に読むこと（2026-09-23）
>
> **exp 0028 は 480/480 完走。`model_ssl.pt` あり。次は下流評価（andre01 で）。**
> `model_ep5`〜`model_ep480` のスナップショットが5刻みで **96本**揃っている。
> **ログインノードでプロジェクト直下の `uv run` / `uv sync` を叩かないこと**（`--no-project` を付ける）。
> 学習用 `.venv` が壊れて4ノードジョブが即死する。経緯は下の「💥 job 3398346 の即死」と
> `env/CONTAINER.md`「⚠️ `uv run` が `.venv` を作り直す事故」。

> **崩壊史のグラフ（全6ラン）: https://claude.ai/code/artifact/f4a6e1f1-5b83-4660-a212-40562d82e363**
>
> **0026 の epoch 推移（exp 0027 の結果）: https://claude.ai/code/artifact/2eadc29f-373e-4157-847f-6d25e62d2b80**
>
> **0025 / 0026 / 0028 の loss・grad_norm・eff_rank 比較（2026-09-19, ep383 時点）:
> https://claude.ai/artifact/E4CyXVWumi9dnaP8yJA4Ha**
> 再生成は `scripts/analysis/plot_training_curves.py`（学習ログのパースのみ。計算ノード不要）。

> **やること・研究方針の一覧は [TODO.md](TODO.md) を参照**（試験間差除去テーマC章を含む）。

---

## 🎯 exp 0028: peak lr 半減 — ✅ **480/480 完走**（job 3382307 → 3398346 → 3429521）

### 🏁 完走した（2026-09-23 22:20, job 3429521）— 崩壊せず、lr 仮説は支持された

```
Epoch: 480, train_loss: 1.0225, lr: 2.29e-06, grad_norm: 4.9905, ln_gain: 1.3000
eff_rank: 557.80  feat_std: 2.7957  val_loss: 4.82
model_ssl.pt 871MB (22:20)      pbsdsh task 0x0〜3 exit status 0      status: COMPLETED
```

15:04 → 22:20 の約7時間15分（61 epoch, 7.1分/epoch）。walltime 10h に収まった。
`Traceback` / `abort` / 崩壊検知・Early Stopping のヒットはいずれも無し。

> ⚠️ 正常終了したこのジョブのログ末尾にも `Cgroup memsw limit exceeded:`（コロンの後が空）が
> 出ている。**これはエピローグの定型出力でエラーではない**ことが、これで確定した。
> 1本目の walltime 切れのときに同じ行を見て OOM と誤読しないこと。

| | 0026（peak lr 2e-3） | **0028（peak lr 1e-3）** |
|---|---|---|
| 到達 epoch | ep89 で崩壊、ep94 abort | **480 完走** |
| train_loss 最小 | 1.5993 | **1.0225** |
| eff_rank 最良 | 318 | **557.8** |
| ep89 前後の grad_norm | 5.67 → 破綻 | 0.82（平坦） |

**投入前に固定した判定基準**は「ep100 前後まで grad_norm が 1.0〜1.5 圏内なら lr 仮説は
支持される」だった。実測は **ep100 で 0.8477** と基準より良い。変更は `--lr` の1点のみ
（非コメント行の差分は `EXP_NAME` / `--note` / `--lr` の3行だけ）なので、
**0026 の崩壊の原因が lr にあった**と切り分けられる。

grad_norm 自体は最後まで登り続けて 5.0 に達したが、0026 と違い **loss が下がり続けながらの
増大**（ep419 1.1927 → ep480 1.0225）で破綻しなかった。
「grad_norm の増大」と「lr の減衰」のレースは **lr の減衰が勝った**。

### 📦 成果物と次の一手

- `outputs/0028_20260917_dino_lr_half/model_ssl.pt`（最終）
- `model_ep5` 〜 `model_ep480` を5刻みで **96本**（この実験だけで **82GB**）

**特徴抽出・下流評価は Miyabi では回さない**（CLAUDE.md）。`toxpatho-uni` 側の
`scripts/extract_features.py --encoder dino --dino-ckpt <ckpt> --name <名前>` を andre01 から使う。
セッション11 で用意した epoch 別下流評価は、当初想定の ep20/45/65/85 だけでなく
**ep5〜480 の全軌跡**に掛けられるようになった。これで「480 epoch の予算が下流性能を
買っているか」（exp 0027 が幾何評価では平坦と示した論点）に決着をつけられる。


`experiments/0028_20260917_dino_lr_half/run_slurm.sh`。**0026 からの変更は
`--lr 5e-4 → 2.5e-4`（実効 peak lr 2e-3 → 1e-3）の1点のみ**で、非コメント行の差分は
`EXP_NAME` / `--note` / `--lr` の3行だけ（`diff` で確認済み）。HANDOFF タスク2-A。

### なぜ lr が第一候補なのか（0026 の実測ログから）

`logs/0026_20260901_dino_clipgrad03/3280970.opbs.OU` の epoch 行を並べると、崩壊の形は
「lr 高原で loss が床に達したあと、勾配が単調増大して臨界を越える」だった。

| epoch | train_loss | lr | grad_norm(クリップ前) | ln_gain |
|---|---|---|---|---|
| 21 | 2.7608 | 2.00e-3 | 0.88 | 0.800 |
| 50 | 2.0612 | 1.97e-3 | 0.84 | 0.657 |
| 66 | **5.3178** | 1.94e-3 | 1.00 | 0.633 | ← 1回目の暴発、自力回復 |
| 75 | 1.6647 | 1.91e-3 | 1.78 | 0.625 |
| 80 | 1.5993 | 1.90e-3 | 1.09 | 0.615 |
| 85 | 1.6132 | 1.89e-3 | **5.67** | 0.611 |
| 88 | 1.6051 | 1.88e-3 | **5.98** | 0.610 |
| 89 | **6.4304** | 1.87e-3 | 3.78 | 0.605 |
| 90 | 11.0904 | 1.87e-3 | 0.0002 | 0.605 | ← 吸収状態 |

1. **loss は ep75→88 で 1.66→1.60 とほぼ床**なのに、同区間で grad_norm が **1.0→6.0**。
   学習が進まないのに勾配だけ増える＝より鋭い領域で跳ね回っている。
   `ln_gain` は 0.61 で平坦なので **weight decay 起因ではない**（wd 修正が効いている裏づけ）。
2. **このランは実質「定常 lr 2e-3」だった。** `CosineLRScheduler(t_initial=480, warmup_t=10)`
   では ep89 は cosine の (89−10)/470 = **16% 地点**にすぎず、lr は 2.00e-3 → 1.87e-3 と
   **6.5% しか落ちていない**。lr アニールによる後半の安定化を一度も受けていない。
   **「lr の高原」は 480 epoch という予算設定の副作用**である点に注意。
3. **クリッピングに下げ代が無い。** `--clip_grad 0.3` は公式 help の推奨レンジ（.3〜1.0）の下限。
   勾配テンソルは student 約157本なので、全体ノルム 0.88(ep21) は1本あたり平均 **0.07**
   でほぼ未発火、5.98(ep88) は1本あたり平均 **0.48** で常時発火していた。それでも破綻した。
   加えて **AdamW の更新量 `lr·m/(√v+ε)` は勾配の定数倍に不変**（1座標あたり実質 ~lr で頭打ち）
   なので、勾配側を抑えても step size は lr でしか変わらない。
   **クリップが飽和した後に残るつまみは lr だけ。**
4. **2e-3 の出自は公式の linear scaling**（5e-4 × 1024/256）。これが成り立つのは critical
   batch size 以下という条件つきで、800枚のWSI由来80万パッチでは同一スライド内パッチの勾配が
   強く相関し critical batch size は ImageNet-1k より小さい。**論文値そのものではなく
   「論文の外挿式をドメイン外に当てた箇所」を動かしている**のが、この逸脱の正当化。

同じ形の破綻は **out_dim 8192 の 0023 でも起きている**（loss 1.72 まで下げた直後 ep42 に
lr 一定・grad_norm スパイク無しで突然崩壊）。out_dim を変えても同じ lr 高原で再現している。

### 投入記録

`qsub` 済み（2026-09-17, **job 3382307.opbs**）。routing queue `regular-g` → 実行キュー
`small-g`（4ノード / walltime 48h / 予約トークン 192.0）。480 epoch は 48h を超えるので
`--resume` で2本目が必要になる見込み。

監視は毎 epoch のログ1行で足りる:

```bash
grep "Epoch: " logs/0028_20260917_dino_lr_half/3382307.opbs.OU | tail -20
```

### ✅ 1本目の結果（2026-09-19 19:45 終了, ep419 / 480）— 崩壊せず、アニールで伸びた

**終了理由は walltime であってエラーではない。** ログ末尾に PBS の kill 通知が入っている。

```
=>> PBS: job killed: walltime 172916 exceeded limit 172800
```

その直後の `❌ Error on line 862 (exit code: 0)` は `scripts/slurm_entry.sh` の ERR トラップが
`_run_multi_node` の失敗を拾っただけ、最後の `Cgroup memsw limit exceeded:`（コロンの後が空）は
エピローグの定型出力で**該当プロセス無し**。OOM ではない。所要 2865.97 分 = 47.8h で
**6.84分/epoch**、投入前の見積り「48h 枠で ep415 前後」はほぼ当たった。

| epoch | train_loss | lr | grad_norm | ln_gain | eff_rank |
|---|---|---|---|---|---|
| 99 | 1.5629 | 9.18e-4 | 0.85 | 0.689 | — |
| 161 | 1.5840 | 7.78e-4 | 1.17 | 0.659 | 440 |
| 240 | 1.5788 | 5.36e-4 | 1.79 | 0.762 | — |
| 300 | 1.5127 | 3.43e-4 | 2.14 | 0.922 | — |
| 360 | 1.3843 | 1.74e-4 | 2.94 | 1.108 | — |
| **419（最終）** | **1.1927** | 5.54e-5 | 4.13 | 1.258 | **540** |

**予想どおり最終アニールで loss が再び下がった。** セッション11 に書いた「実効減衰圧 `lr×wd` は
ep250 付近でピークを打って以降は下がるので、最終アニールで loss は再び下がる見込み」という
予想が実測で確認できた（ep161 の 1.5840 から **−24.7%**）。これは本プロジェクトで測った事実に
なったので、予想扱いを解除してよい。

**eff_rank は 0026 の最良 318 に対して 540。** ep161 の 440 からさらに +23% 伸びており、
アニール区間は表現の次元数の面でも買い物をしている。`feat_std` 0.665 → 2.614。

### ⚠️ grad_norm のランプは最後まで止まらなかった（ただし 0026 とは別物）

ep59 の 0.733 で底を打ってから ep419 の 4.13 まで単調増大（+464%）。0026 の ep75→88 の
1.0 → 6.0 に匹敵する倍率まで来ている。**それでも崩壊しなかった**のは、

- 0026 は **loss が床に張り付いたまま** grad_norm だけ伸びた（ep75→88 で 1.66→1.60）。
- 0028 は **loss が下がり続けながら** grad_norm が伸びた（ep300→419 で 1.51→1.19）。

つまり 0028 の grad_norm 増大は「鋭い領域で跳ね回っている」のではなく、lr が 2 桁落ちる区間で
step size が縮んだぶん勾配が大きいまま釣り合っている状態。`ln_gain` の 0.66 → 1.26 も
wd スケジュール（0.04 → 0.4）の下で LayerNorm ゲインが伸びている挙動と整合する。
**「grad_norm の増大」と「lr の減衰」のレースは lr の減衰が勝った**、というのがこのランの結論。

### 🔁 resume は投入できる状態（点検6項目すべて通過, 2026-09-19 確認）

`experiments/0028_20260917_dino_lr_half/resume.sh` の点検項目を実機で確認した結果:

| # | 項目 | 結果 |
|---|---|---|
| 1 | 同じ実験のジョブが走っていないか | ok（`qstat` は `No unfinished job found`） |
| 2 | `model_ssl.pt` が無いか | ok（walltime 切れなので書かれていない） |
| 3 | `state.pt` の鮮度 | ok（1.7GB, 09/19 19:44 = ep419 の直後） |
| 4 | 最終 epoch | ok（ログ `Epoch: 419` = `loop.py:206` の `epoch+1` なので 419 完了、残り **61**） |
| 5 | 必要 walltime | **10:00:00**（61 × 6.9分 + 30%。実測は 6.84分/epoch なので実時間 ~7.0h） |
| 6 | 吸収状態でないか | ok（train_loss 1.1927。11.0904 への張り付き無し） |

`run_slurm.sh` は既に `--resume` を渡しており（162行目）、`entry.py:213` が
`start_epoch = state['epoch'] + 1 = 419` で再開するので、**ep419〜479 の 61 epoch** を走る。

```bash
bash experiments/0028_20260917_dino_lr_half/resume.sh --submit
# 実体: qsub -l walltime=10:00:00 experiments/0028_20260917_dino_lr_half/run_slurm.sh
```

コストは 4ノード × 約7h = **約28ノード時間**。ディスクは 0028 だけで既に 70GB
（`outputs/` 全体 145GB）、`--save_interval 5` なので resume 分で +12本 ≈ 10GB 増える。

### 💥 job 3398346 の即死（2026-09-19 23:50）— 学習ではなく環境の事故

**1 epoch も進んでいない。** 23:38 起動 → 23:50 に4ノードすべて `exit status 1`。
`state.pt` は ep419（09/19 19:44）のままで、**失った計算は約0.8ノード時間だけ**。

```
/work/.../.venv/bin/python: Error while finding module specification for
'torch.distributed.run' (ModuleNotFoundError: No module named 'torch')
```

生成スクリプト（`_multinode_incontainer.sh` / `_multinode_launch.sh` / `command.sh`）は
**1本目とジョブID・ホスト名以外が完全に同一**（`diff` 確認済み）。SIF も同じ。
つまり**変わったのは `.venv` だけ**である。

| | 1本目 3382307（成功） | 2本目 3398346（失敗） |
|---|---|---|
| venv の土台 python | コンテナ `/usr/bin/python3`（**3.12.3**） | uv 管理 standalone CPython **3.12.13** |
| `include-system-site-packages` | true | **false** |
| torch | コンテナの `/usr/local/lib/python3.12/dist-packages`（2.13.0a0） | **見えない** |
| timm / wandb | あり | **無し** |

原因は `uv run`。詳細と再発防止は **`env/CONTAINER.md`「⚠️ `uv run` が `.venv` を
作り直す事故」** に書いた。要点だけ:

- ログインノードの `/usr/bin/python3` は **3.9.25**、コンテナは **3.12.3**。
  学習用 venv はコンテナ python 土台なので、**ログインノードからは壊れて見える**。
- `pyproject.toml` の `requires-python = "==3.12.*"` を満たさないと判断した uv が、
  **確認なしに `.venv` を作り直した**（`--all-extras` 無しなので timm も消える）。
- 引き金は `scripts/analysis/plot_training_curves.py` の docstring にあった実行例
  `uv run --with matplotlib python ...`（`--no-project` 無し）。
  `.venv/pyvenv.cfg` が **15:36:16**、グラフ `outputs/_analysis/curves_0025_0026_0028.png`
  が **15:37** に生成されており、時刻が一致する。
  **グラフを1枚描いただけなので venv を触った自覚は残らない。**

### 🔧 復旧手順（2026-09-23 実装・スクラッチで検証済み、未適用）

```bash
bash tools/rebuild_venv.sh            # 点検のみ
bash tools/rebuild_venv.sh --apply    # 既存 .venv を .venv.bak.<日時> へ退避してから作り直す
```

壊れる前の中身は **job 3382307 自身の wandb 記録**から復元した
（`wandb/offline-run-20260917_195710-0laflbix/files/requirements.txt`, 350パッケージ）。
コンテナ側 `pip freeze`（217パッケージ）との差分＝**venv 固有は25個だけ**で、
base 依存 + `timm` に一致する。torch / torchvision / wandb はコンテナ側から来ていた。

`/tmp` のスクラッチで手順を検証した結果（`.venv` には触れていない）:

```
torch   2.13.0a0+8145d630e8.nv26.06  /usr/local/lib/python3.12/dist-packages/torch   <- ep1〜419 と同一
timm    1.0.28                                                                      <- 記録と一致
h5py    3.16.0 / numpy 2.3.5                                                        <- 記録と一致
OK: venv に torch なし（コンテナ版を使用）
```

⚠️ `uv pip install timm`（`--no-deps` 無し）だと **uv が torch 2.14.0 を venv 側に入れて
コンテナの 2.13.0a0 をシャドウする**ことを実機で確認した。`env/CONTAINER.md` が
2026-08-29 時点で警告していた事象が実際に起きる。**必ず `--no-deps` を付ける。**

### 🛡️ 再発防止（実装済み）

| 場所 | 内容 |
|---|---|
| `experiments/0028_.../resume.sh` | **点検項目7** を追加。`pyvenv.cfg` の `home` / `include-system-site-packages` と `timm` の有無を見て、壊れた venv なら投入前に止める |
| `scripts/analysis/plot_training_curves.py` | docstring の実行例に **`--no-project`** を追加し、外した場合に何が壊れるかを明記 |
| `tools/rebuild_venv.sh` | 復旧手順をスクリプト化（退避 → 作り直し → 本番と同じ経路での import 検証まで） |
| `env/CONTAINER.md` | 事故の機序・根拠・直し方を記録 |

点検項目7 が現に止まることを確認済み:

```
NG: .venv の土台がコンテナの python ではない。uv に作り直された疑い。
    bash tools/rebuild_venv.sh --apply で直すこと
```

**この検査は他の実験の投入スクリプトにも入れること。** 4ノード確保してから落ちるのは高い。

### ✅ venv を復元して3本目を投入した（2026-09-23, **job 3429521.opbs**）

`bash tools/rebuild_venv.sh --apply` を実行。旧 venv は `.venv.bak.20260923_150300`（1.1GB）へ退避。
本番と同じ経路（コンテナ内で `.venv` を activate）での検証が全通過した:

```
torch   2.13.0a0+8145d630e8.nv26.06  /usr/local/lib/python3.12/dist-packages/torch
timm    1.0.28
h5py    3.16.0 / numpy 2.3.5 / wandb 0.27.2
lib.* の import OK
torch.distributed.run OK
```

`timm` / `wandb` / `h5py` / `numpy` の4つは **job 3382307 の requirements.txt と版まで一致**。
`pyvenv.cfg` も `home = /usr/bin` / `version_info = 3.12.3` /
`include-system-site-packages = true` と、壊れる前の構造に戻った。

新しい `.venv` は **67MB**（壊れていた方は 1.1GB）。ほとんどのパッケージが
コンテナ側から供給される設計どおりの姿で、venv 固有は25個だけという差分解析とも整合する。

`resume.sh` の点検は**項目7を含めて全通過**し、`qsub -l walltime=10:00:00` で投入した。
routing queue `regular-g` → 実行キュー `small-g`（4ノード / 予約トークン 40.0）。

#### ✅ 起動確認（2026-09-23 15:23）— resume は成功している

```
[20260923-151710] --ddp_linear_scale_lr: lr scaled by world_size=4 -> 0.001   (1本目と同じ。0.002 なら直し忘れ)
[20260923-151720] Resumed from state.pt (epoch 418) -> continue at epoch 419/480   (4ランクすべて)
[20260923-152310] Epoch: 420, train_loss: 1.1864, lr: 5.40e-05, grad_norm: 4.0004, ln_gain: 1.2601
```

`ModuleNotFoundError` は出ていない（前回はここで12分後に死んだ）。
**状態の復元は完全**で、1本目 ep419 の各値から滑らかに続いている:

| | 1本目 ep419 | 3本目 ep420 |
|---|---|---|
| train_loss | 1.1927 | **1.1864** |
| lr | 5.54e-05 | 5.40e-05 |
| grad_norm | 4.1278 | 4.0004 |
| ln_gain | 1.2584 | 1.2601 |

`traceback` / `abort` / 崩壊検知のヒットは無し（`collapse` の12ヒットは
`Execution Arguments` が `collapse_early_stop` 等を4ランク分エコーしているだけ）。
ep420 の所要は約5.8分なので、残り60 epoch は **約6時間**、walltime 10h に十分収まる。

### 🚀 2本目を投入した（2026-09-19, **job 3398346.opbs**）— 即死したので無効

ユーザー確認のうえ `resume.sh --submit` を実行。routing queue `regular-g` → 実行キュー
`small-g`（4ノード / walltime 10h / 予約トークン 40.0）。投入直後は QUEUED。

起動したらまず次の3行を確認すること:

```bash
# 1. state.pt から再開できたか（fresh start になっていないか）
grep -a 'Resumed from state.pt' logs/0028_20260917_dino_lr_half/3398346.opbs.OU
#    期待: Resumed from state.pt (epoch 418) -> continue at epoch 419/480

# 2. lr の掛け直しが 1本目と同じか（0.002 なら --lr 直し忘れ）
grep -a 'lr scaled by world_size' logs/0028_20260917_dino_lr_half/3398346.opbs.OU

# 3. 以後の監視
grep -a 'Epoch: ' logs/0028_20260917_dino_lr_half/3398346.opbs.OU | uniq | tail -20
```

見るべき点は **ep419 の再開直後に train_loss が 1.19 付近から続くか**。ここが跳ねるなら
optimizer/scheduler の復元に問題がある。残り 61 epoch で lr は 5.5e-5 → 2e-6 まで落ちるので、
`grad_norm` のランプ（現在 4.13）はここで頭打ちになるはず。ep480 到達時に `model_ssl.pt` が
書かれ、以後 `--resume` は「training complete」で即終了するようになる。

### 経過（2026-09-18 14:20 時点, ep161 / 480）— 崩壊せず、判定基準は満たしている

18.6時間で ep161（6.9分/epoch）。`traceback` / 崩壊検知のヒットは無し。起動時チェック2つ
（`lr scaled by world_size=4 -> 0.001` / `exempt (bias & ndim<=1) 102 tensors`）も通過。

**0026 が死んだ epoch 帯を、0028 は grad_norm 0.75 前後で平坦に通過した。**

| epoch | 0026 の grad_norm | 0028 の grad_norm |
|---|---|---|
| 50 | 0.84 | 0.76 |
| 65 | 1.19 | 0.75 |
| 75 | **1.78** | 0.79 |
| 85 | **5.67** | 0.82 |
| 88/90 | **5.98** → ep89 破綻 | 0.82 |

```
ep  99: loss 1.5629  lr 9.19e-4  gn 0.84  ln_gain 0.690   <- train_loss の最小値
ep 161: loss 1.5840  lr 7.78e-4  gn 1.17  ln_gain 0.659
eff_rank 440 / feat_std 0.665 / uniformity -0.0056 / val_loss 4.75
```

**0026 より良い**: loss の床 1.5629（0026 は 1.598）、`eff_rank` 440（0026 の最良 318）。
lr を半分にして表現の質が落ちてはいない。

### ⚠️ ランプ自体は出ている（速度が約28分の1になっただけ）

grad_norm は ep59 の 0.733 で底を打ってから ep161 の 1.17 まで単調増大（+60%）。
30 epoch あたりの増分も +0.096 → +0.109 → +0.139 と緩やかに加速している。
0026 は同じ区間で 1.0 → 6.0（+500%）だったので速度は全く違うが、**機序は同じ**。
「grad_norm の増大」と「lr の減衰」のレースであり、まだ決着していない。
次の確認点は **ep250 前後**で grad_norm が 1.5 を超えていないか。

### ◆ ep99 以降の train_loss 上昇は weight decay スケジュールで説明できる

`loop.py` の wd は `wd = wd_end + 0.5(wd_start − wd_end)(1 + cos(π·epoch/num_epoch))` で
**0.04 → 0.4 へ単調増加**する。loss 最小の ep99 から現在までに **wd は 1.7 倍**になっている。

| epoch | wd | lr | lr×wd（実効減衰圧） |
|---|---|---|---|
| 99（loss 最小） | 0.076 | 9.18e-4 | 7.0e-5 |
| **161（現在）** | **0.131** | 7.78e-4 | **1.02e-4** |
| 250（lr×wd のピーク） | 0.232 | 5.01e-4 | **1.16e-4** |
| 350 | 0.339 | 1.96e-4 | 6.7e-5 |
| 415（walltime 到達見込み） | 0.384 | 6.00e-5 | 2.3e-5 |
| 480 | 0.400 | 2.07e-6 | 8.3e-7 |

つまり train_loss の +1.3%（1.5629 → 1.5840）は**設計どおりの正則化圧の増加**であり、
病的な現象ではない。実効減衰圧 `lr×wd` は **ep250 付近でピーク（現在比 +14%）を打って
以降は下がる**ので、最終アニール（lr が 6e-5 → 2e-6 に落ちる区間）では loss は再び
下がる見込み。~~ただしこれは cosine スケジュールの一般的な挙動からの**予想**であって、
本プロジェクトで測った事実ではない。~~ → **ep419 まで走って実測で確認した**（1.5840 → 1.1927）。
上の「1本目の結果」を参照。

### 📉 「計算資源の無駄」への対処 — 止める根拠は下流評価でしか作れない

上の通り loss の上昇そのものは無駄の証拠ではない。一方 exp 0027 は
**表現（η²_slide）が ep45 以降ほぼ動かない**ことを示しており、
「ep160 → ep480 が下流性能を買っているか」は依然として未測定のまま。

→ **epoch 別下流評価の対象を 0026 から 0028 に切り替えるべき。** 0028 は ep5〜ep160 の
チェックポイントが5刻みで32本（28GB）揃っており、これが**現行レシピそのもの**の軌跡である。
コストは約2ノード時間で、もし平坦だと分かれば **resume（約30ノード時間）を見送る判断**や
**次ランの cosine 周期短縮**の根拠になる。2 ノード時間で 30〜100 ノード時間の判断ができる。

### 🔁 resume の準備（2026-09-18 実装・点検済み）

`experiments/0028_20260917_dino_lr_half/resume.sh`。6.9分/epoch なので **48h 枠では
ep415 前後で切れる**（480 に 65 epoch 足りない）。

```bash
bash experiments/0028_20260917_dino_lr_half/resume.sh            # 点検のみ(既定)
bash experiments/0028_20260917_dino_lr_half/resume.sh --submit   # 点検して qsub
```

点検する6項目:

1. 同じ実験のジョブがキュー/実行中でないか（二重投入すると同じ `state.pt` を両方が書く）
2. `model_ssl.pt` が無いか — **あると `entry.py:179` が即終了する**（退避コマンドを表示）
3. `state.pt` の有無と鮮度
4. ログから最終 epoch と残り epoch 数
5. 必要 walltime の見積り（残り × 6.9分 + 30%、48h で打ち切り）を `qsub -l walltime=` に渡す
6. `train_loss` が 11.09 に張り付いていないか（**吸収状態から resume しても無意味**）

`state.pt` は rank0 が**毎 epoch** 書いており（`loop.py:266`）、`--dir_result` が
プロジェクト側 `outputs/` を直接指している（scratch 経由ではない）ことを実機で確認済みなので、
walltime による強制終了でも直前の epoch から再開できる。現在は項目1で正しく止まる
（実行中のため）ことと、項目1を外せば2〜6が通ることの両方を確認した。

### 判定基準（投入前に固定した）

- 本命は **クリップ前 grad_norm のランプが出るか**。0026 は ep55 付近からノイズが増え
  ep75 以降で単調増大した。**ep100 前後まで 1.0〜1.5 圏内なら lr 仮説は支持される。**
- **loss の下降が 0026 より遅いのは当然**（lr 半分）。ep50 で 0026 の 2.06 に届かないこと
  自体は失敗ではない。見るのは「床に達してから grad_norm が伸び始めるか」。
- 起動直後に `lr scaled by world_size=4 -> 0.001` が出ること（0.002 なら `--lr` 直し忘れ）。

### ◆ 精度（fp32 head）を第一候補にしなかった理由

HANDOFF タスク2-B（`--dino_fp32_head`）は実装済みだが、**0026 の症状とは噛み合わない**。

- `center` は `register_buffer(torch.zeros(1, out_dim))` = **fp32 バッファ**なので、
  `t - center` の**引き算自体は fp32 で行われる**。効いているのは減算の桁落ちではなく、
  その手前で `t` が bf16 に丸められる誤差。`dino.py` のコメントの「桁落ち」は言い過ぎ。
- `norm_last_layer=True` かつ入力 L2 正規化なので `t` は実質コサイン（|t| ≤ 1）。
  bf16 の丸めは |t|~0.1 で ≈5e-4、|t|~1 で ≈4e-3。`÷0.04` で 25 倍して
  **logit 上 0.01〜0.1**、teacher softmax の確率にして ±10% 程度。しかも `detach()` 済み。
- **ゼロ平均の丸めノイズは 30 epoch の単調ランプを作らない。** ep1〜85 は同じ bf16 で健全。
- `--dino_fp32_head` が fp32 化するのは**ヘッドの Linear と損失だけ**で、ViT-B の 12 ブロックは
  bf16 のまま。「精度が原因」を本気で検証するなら full fp32 か fp16+GradScaler であって、
  head-only は**安いが弱い検証**にあたる。
- ⚠️ 用語: 精度を上げるのは**論文・公式既定から遠ざかる**方向（公式は `use_fp16=True` =
  fp16+GradScaler、論文本文に精度の記述は無い）。「論文に近づける」と書いてはいけない。
  公式 help のトラブルシュート助言に従う**逸脱**である。

→ 別軸の対抗馬として残す。A が外れたら投入する。

---

## 🔬 所見ラベルによる epoch 別下流評価（2026-09-17, セッション11）— 実装済み・未実行

### まず訂正: セッション10 の「ep85 の下流評価」は下流評価ではない

`TODO.md` A節に `[x] DINO ep85 の下流評価` と書かれていたのは exp 0027 のことだが、中身は
**CKA / η²_slide / η²_color / k-means = ラベルを一切使わない幾何評価**。0027 自身が
「**下流タスク性能が epoch とともに伸び続ける可能性は否定できない**」と断っている。
**「現状の DINO が UNI などと比べてどのレベルか」はまだ一度も測っていない。**
TODO.md の項目名を `ep85 の表現評価（幾何）` に訂正した。

### この評価が決めること

exp 0028（案1: lr のみ変更・480 epoch 維持）と、案2（lr 半減 + cosine 周期短縮）の切り分け。

| 下流 AUROC の epoch 推移 | 読み | 次のラン |
|---|---|---|
| ep45 → ep85 で横ばい | 0027 の幾何評価と一致。epoch 予算は何も買っていない | 案2（cosine 周期短縮）に切り替える根拠が固まる |
| 単調に上昇 | 幾何は平坦でも下流は伸びている | 案1 のまま（480 維持） |
| どの epoch も ViT-ImageNet 未満 | lr でも epoch でもなくレシピ/データ側の問題 | A/B より前に設計を疑う |

0027 は「幾何は平坦」までしか示しておらず、**案2 を選ぶ根拠としては不十分**。

### 実装したもの（`toxpatho-uni` 側）

| ファイル | 役割 | 実行場所 |
|---|---|---|
| `scripts/extract_dino_epochs.pbs` | ep20/45/65/85 を**1ジョブ**で特徴抽出 | ⚠️ Miyabi（GPU） |
| `scripts/eval_dino_epochs.sh` | スコア化 → 所見別 AUROC をベースラインと並べる | andre01（CPU のみ） |

既存資産の確認結果（`toxpatho-uni`）:

- `scripts/extract_features.py: build_dino` は **本リポジトリのチェックポイント形式に既に対応済み**
  （466キーから `student_backbone.` を剥がして timm ViT-B/16 に `strict=True`、768次元を返す）。
  `--name` の help にも「同じ dino で epoch 違いを並べるときに `dino_ep85` 等と分ける」とあり、
  epoch 掃引が想定済み。
- ベースラインは `outputs/features/{uni,uni2h,vit_imagenet}.h5` と `outputs/eval/by_finding.csv`
  に既にある。スコアは**対照群セントロイドからの Mahalanobis 距離で学習を伴わない**ので、
  `TODO.md` B節の **ABMIL の val リーク問題を踏まずに**順位を比較できる。
- プーリングは所見の空間分布で使い分ける（病巣性 → top16 / びまん性 → slide_mean）。
  epoch 推移が両者で食い違う可能性があるので**両方出して別表にする**。

### ⚠️ 特徴抽出だけは Miyabi で回すしかない

入力の **1.1TB パッチ集合**（`/work/gd43/share/tggates/sample_patch_agg`, 512パッチ/スライド ×
11,295スライド）が Miyabi の共有ストレージにしか無く、andre01 へ移すコストの方が大きい。
`eval_dose_response_patch.py` は `patch_feat` を要求するので `--no-patch-feat`（34MB に縮む）も
使えない。生成される h5 は **1 epoch あたり 11GB**。

コスト比較で引き合うと判断した:

| | ノード時間 |
|---|---|
| 特徴抽出 4 epoch（ViT-B/16, 1ノード, 30分弱/本） | **約 2** |
| 判定対象の 0028（4ノード × 48h × 2本） | 約 384 |

**0.5% 程度**で「480 epoch 積む価値があるか」を先に決められる。スコア化以降は CPU のみなので
h5 を andre01 へ持っていって `eval_dino_epochs.sh` を回す。**qsub はユーザー承認後。**

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
