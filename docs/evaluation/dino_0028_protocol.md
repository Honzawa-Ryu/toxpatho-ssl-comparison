# DINO exp 0028 の評価手順

2026-10-01 のユーザー合意: まず学習の良否を評価し、その後に CKA 等で学習自体を分析する。
作業ブランチは `eval/model-evaluation`。学習コード・学習出力は読み取りのみとし、
評価結果はユーザー指示により、元プロジェクト直下の `outputs/evaluation_dino_0028/` に保存する。
最初の5点の CKA 結果は `outputs/evaluation_dino_0028/cka_20261001/`。
評価用 worktree の同名出力パスは、その保存先へのシンボリックリンクにしてある。
追加の全96点は `outputs/evaluation_dino_0028/dense_20261001/` に保存した。

同日の追加指示: ベンチマークはユーザーが用意し、その間に実行できる CKA 等を先に行う。
旧 `wsi-ad/data/shards/` の画像を使った結果は探索的な表現解析として区別する。
学習時の memmap index が見つかっていないため、事前学習との WSI 重複は未確認。
ベンチマークによる第1段階の性能評価は引き続き必要。

## 追加実行: 5 epoch刻みと後半のEffrank（完了）

ユーザーの追加指示により、保存済みep5〜480の全96本を同じ2,000枚で比較した。
全体CKAとスライド平均除去後CKAの96×96行列、隣接5 epochのCKA、中心化スペクトル、
L2正規化と最終LayerNormの倍率・バイアス除去後のrank、学習ログとの対応を出力。
スライド平均除去後CKAは全パッチをまとめて計算し、スライド別CKAの平均とは区別する。

特異値entropy rankはep320の413.1からep440の504.2へ増加し、ep480は509.1。
増加はおおむね300〜440に続き、450以降は横ばい。L2正規化・最終LayerNorm除去後も残る。
90%分散の成分数はep320の119からep480の216へ増加した。
後半の代表点と大きな増分の両端、計12点をFP32（TF32無効）で照合し、rank差は最大0.0324%。
この操作は内部層のLayerNormを除かない。学習率・weight decay等の因果効果も分離していない。
rankの増加を病理所見の識別向上と同一視しない。

[詳細レポートと図](/workspace/andre01/honzawa/02-playground/toxpatho-ssl-comparison/outputs/evaluation_dino_0028/dense_20261001/report.md)。
入力・特徴・最終LayerNorm・ログのハッシュを保存し、前回5点の特徴とCKAの一致を確認した。

### 再実行手順

評価用worktreeで実行する。下記は保存済みの結果を保つため別フォルダへ出力する例。
既存環境を使い、GPU1枚で抽出、CKA・スペクトルはCPUで解析する。

```bash
EVAL_PYTHON=/workspace/andre01/honzawa/01-toxpatho/wsi-ad/.venv/bin/python3
EVAL_ROOT=/workspace/andre01/honzawa/02-playground/toxpatho-ssl-comparison
EVAL_SHARD=/workspace/andre01/honzawa/01-toxpatho/wsi-ad/data/shards/shard_0000.tar
EVAL_OUT="$EVAL_ROOT/outputs/evaluation_dino_0028/dense_repeat"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0

"$EVAL_PYTHON" -u scripts/analysis/analyze_dino_epochs.py \
  --config scripts/evaluate/dino_0028_dense.yaml \
  --shard "$EVAL_SHARD" --checkpoint-root "$EVAL_ROOT" \
  --same-sample-as "$EVAL_ROOT/outputs/evaluation_dino_0028/cka_20261001" \
  --output "$EVAL_OUT" --extract-only --device cuda:0 --batch-size 64

"$EVAL_PYTHON" -u scripts/analysis/analyze_dino_dense.py \
  --output "$EVAL_OUT" --training-dir "$EVAL_ROOT/outputs/0028_20260917_dino_lr_half"

"$EVAL_PYTHON" -u scripts/analysis/analyze_dino_epochs.py \
  --config scripts/evaluate/dino_0028_precision.yaml \
  --shard "$EVAL_SHARD" --checkpoint-root "$EVAL_ROOT" \
  --same-sample-as "$EVAL_OUT" --output "${EVAL_OUT}_fp32" \
  --extract-only --precision fp32 --device cuda:0 --batch-size 64

"$EVAL_PYTHON" -u scripts/analysis/analyze_dino_epochs.py \
  --config scripts/evaluate/dino_0028_precision_jumps.yaml \
  --shard "$EVAL_SHARD" --checkpoint-root "$EVAL_ROOT" \
  --same-sample-as "$EVAL_OUT" --output "${EVAL_OUT}_fp32_jumps" \
  --extract-only --precision fp32 --device cuda:0 --batch-size 64

"$EVAL_PYTHON" -u scripts/analysis/analyze_dino_dense.py \
  --output "$EVAL_OUT" --precision-only \
  --fp32-output "${EVAL_OUT}_fp32" --fp32-output "${EVAL_OUT}_fp32_jumps"
```

## 第1段階: 学習の良否

対象は `0028_20260917_dino_lr_half` の ep20 / ep85 / ep160 / ep320 / ep480。
現時点では評価未実施。記録にある train_loss や effective rank から下流性能を推定しない。

### 入力と比較条件

- 全 epoch と比較モデルに同一の画像・パッチID・順序・分割を使う。
- 既存抽出コードは DINO の student backbone を使う。まずこれに統一し、
  teacher の評価を追加する場合は別条件として記録する。
- 比較対象は ImageNet 学習済み ViT と UNI。パッチの解像度・選択を共通にし、
  各モデルの推論前処理・重みのバージョンを記録する。
  既存の比較スコアは、画像集合・分割・正常モデルの推定方法が一致する場合のみ再利用する。
- UNI との比較にはモデル規模・学習データの差が含まれる。DINO レシピだけの効果と解釈しない。
- 入力画像・メタデータ・抽出重みの対応を確定するまで、旧実験の埋め込みを代用しない。

### 分割とラベル

- スライドの所見ラベルはスライドの評価に使い、全パッチに病変があるとは扱わない。
- 同じ動物・スライドが正常モデルの推定、検証、最終テストをまたがない分割を使う。
  動物IDが無い場合は、その制約を記録し、スライド分割だけで動物独立と主張しない。
- 事前学習に使用した WSI と評価 WSI の重複を確認する。`lib/trainer/data.py` と同じ
  fold 分割で保持された WSI と、事前学習に使った WSI の結果を混ぜない。
  必要な index が無い間は、事前学習との独立性は未確認とする。
- 未知の試験への汎化は試験単位の保持で別に測る。スライド間の分割と区別して報告する。
- epoch・集約方法等の選択は検証集合で行い、最終テストで選ばない。
  テストを見た後の変更は探索評価として記録し、独立した性能推定と区別する。

### 健全性

固定した評価パッチ集合で NaN/inf、特徴ノルム、各次元の分散、中心化後の特異値分布、
effective rank、異なるパッチ間のコサイン類似度を確認する。
未中心化の特徴と中心化した特徴の指標を区別する。
effective rank や η² の一指標だけで崩壊・学習成功と判定しない。

### 所見別異常検出

エンコーダを固定し、正常参照群の特徴から Mahalanobis 距離を計算する。
平均・共分散を推定する処理は統計的な適合を伴うため、参照群と評価対象を分ける。
PCA、正規化の統計、共分散の正則化や次元数の調整も、参照群・検証集合だけで決める。
同じ対照スライドを参照群と陰性の評価対象の両方に使わない。
対照群の所見情報があれば確認し、所見なしを勝手に仮定しない。

- スライドごとに `slide_mean` と `top16` を両方出す。
  `top16` はパッチ異常スコア上位16枚の平均として定義し、空間的に局所的な変化を見る。
  `slide_mean` は全パッチ異常スコアの平均として定義し、びまん性変化を見る。
  既存コードの定義が異なる場合は確認して明示し、同じ名前で混ぜない。
- パッチ数を揃え、16枚未満のスライドはエラーまたは明示的除外とする。
- 所見の陽性・陰性の定義、重複所見・欠損ラベルの扱いを評価コードと照合して固定する。
  未記録の所見を自動的に陰性とみなさない。
- 各所見の AUROC と AUPRC（実装上は average precision か積分法かを明記）、
  陽性・陰性スライド数、陽性割合を報告する。両クラスが無い所見は推定不可とする。
- 不確実性は動物単位、試験汎化なら試験単位で bootstrap し、
  モデル差は同じ resample による対応あり比較にする。パッチを独立標本とみなさない。
- 所見別結果を主とし、全所見の単純平均だけで成功と判定しない。

### スライド・色依存と画像確認

スライドごとのパッチ数を揃えた固定集合で `lib/analysis/batch_color.py` の
η²_slide / η²_color とクラスタ・スライドIDの対応を測る。
η² の偶然値は集合のスライド数・パッチ数に依存するため、ラベル置換で参考値を作る。
スライド依存には本来の生物学的な差も含まれ、η²_slide が低いだけで良いモデルとしない。

クラスタ代表パッチと、クエリとは異なるスライドから検索した類似パッチを並べる。
類似検索では同じ動物も除外できるなら除外する。
スライドID・所見・スコアを添え、形態が似ているのか、染色やフォーカスが似ているのかを確認する。
代表画像の選択は固定規則にし、良く見える例だけを選ばない。

### 出力と判断

1. 入力・重み・分割・前処理を記録した実行メタデータ。
2. 各 epoch の特徴健全性の表。
3. 所見別・集約方法別スコア表と epoch 推移図、比較モデルとの差・不確実性。
4. スライド・色依存の表と代表・類似画像集。
5. 改善した所見、横ばいの所見、依存が強まった条件、次に調べる仮説。

健全な特徴が出ること、所見を識別すること、スライドに依存しにくいことを別々に判断する。
少数陽性や信頼区間の重なりが大きい場合は優劣を保留する。

## 第2段階: 学習中の変化の分析

第1段階を確認した後、同じ固定パッチ集合で epoch 間 CKA を計算する。
全スライドをまとめた CKA と、スライド内 CKA を併記し、スライド差の寄与を切り分ける。
必要なら所見別の集合・層別 CKA を追加する。
下流性能の変化点と表現の変化を対応づけるが、CKA の高低だけで良否や因果関係を決めない。

## 現時点の入力確認

2026-10-01、ホスト名は `andre01`。5本のチェックポイントは元フォルダの
`outputs/0028_20260917_dino_lr_half/` に存在する。
評価用 worktree には Git 管理外の outputs は自動コピーされないため、
元フォルダを読み取り元として明示する。

記録にある `toxpatho-uni` の `data/manifest.csv` と評価スクリプトは、
現在確認した `/workspace/andre01/honzawa/{01-toxpatho,02-playground}` には見つからない。
場所をユーザーに照会中。入力が確定してから既存コードの分割・正常モデル・集約定義を確認する。
ホスト Python には torch / timm / h5py / scikit-learn が無い。
評価用の環境を選ぶ際は、学習側の環境を変更せずに別環境を使う。

チェックポイントと評価資産の配置を確認するコマンド（特徴抽出・評価は実行しない）:

```bash
python3 scripts/evaluate/check_dino_0028_inputs.py \
  --checkpoint-root /workspace/andre01/honzawa/02-playground/toxpatho-ssl-comparison \
  --evaluation-root /path/to/toxpatho-uni
```
