# wsi-ad TODO / ロードマップ

最終更新: 2026-07-10
関連: [Goal.yaml](Goal.yaml) ・ [PROJECT_STATUS.md](PROJECT_STATUS.md) ・ [related_work.md](related_work.md)（先行研究サマリ）

---

## 背景（なぜこのTODOなのか）

非対照SSL手法の「表現の質」を比較する過程で、**試験間差（バッチ＝スライド/試験レベルの差）が
支配的**であることが分かった（2026-07-10 の検証）:

- 埋め込み分散の **55〜87% がスライド identity で説明**される（η²_slide、偶然値≈0.07）。
- **白黒化してもη²_slideはほぼ下がらない**（0.70→0.66）→ バッチは色ではなく**構造由来**
  （スキャナ・フォーカス・切片厚・テクスチャ・圧縮）。
- **単一スライドに絞る（バッチ固定）と手法間CKAが跳ね上がる**（RN50·BT↔DN121·BT 0.43→0.76）
  → 多スライドで見えた「手法差」の大半は**バッチ符号化の差**で、組織の見方の差ではない。

→ 帰結: (1) 公平な手法比較にはバッチ統制が必須。(2) さらに進んで
**「試験間差を除去する表現学習」自体を研究テーマにできる**（★C章）。

---

## A. いま動いていること
- [ ] **ViT-L MAE（backbone規模プローブ）**: `experiments/20260713_000001_tggate_vitl16_mae/run_slurm.sh` 準備済み・投入待ち。
      ViT-B MAE(20260710_000001)と同一レシピでencoderのみViT-L/16化（`--model_name ViTL16`→mae_vit_large_patch16, 329.5M）。
      目的=UNIとの差のうち「アーキ規模由来」を単離。VRAM実測 bs=256で約24GB/48GB（余裕）。
      注意: 約2〜2.5倍/ep→100epは72h枠超過の可能性、state.pt毎ep保存でresume前提。投入コマンド:
      `sbatch experiments/20260713_000001_tggate_vitl16_mae/run_slurm.sh`
- [ ] 本命4手法（SimSiam / Barlow Twins / MAE / DINO）の同一条件・事前学習
      - 実行中: MAE(job 6287) / DINO(6288) / SimSiam-wd(6289, SwAV完了待ち) / BT=完了済み
      - [ ] **DINOのeff_rank崩壊監視**（epoch3でloss≈ln(K)に戻る挙動あり。epoch5監視、崩壊確定なら
        last-layer freeze / lr低減 / teacher-temp warmup で対処）

## B. 比較パイプライン（実装済み → 拡充）
- [x] CKA（線形/RBF, 次元非依存）、k-means ARI/NMI、プロトタイプパッチ、t-SNE
- [x] バッチ/色影響の定量: η²_slide・η²_color・within-slide cohesion（`batch_color_influence.py`）
- [x] ablation: 白黒 `--grayscale` / 単一スライド `--filter_wsi`
- [ ] 本命4手法完了後に `sbatch scripts/analysis/run_analysis_slurm.sh` で本比較
- [ ] **バッチ統制版を標準パネル化**: within-slide CKA/ARI、slide-balancedサンプリングを既定に
- [ ] **[要修正/方法論] ABMIL評価のval集合リーク** ([scripts/evaluate/abmil_eval.py](scripts/evaluate/abmil_eval.py))
      `train_abmil`が各foldのval AUCでbest-epochを選び、`eval_abmil`が**同じval**で最終AUCを報告
      している（[abmil_eval.py:166-212](scripts/evaluate/abmil_eval.py#L166-L212) / [:317-319](scripts/evaluate/abmil_eval.py#L317-L319)）。
      報告AUCが実質「エポック中のval AUC最大値」になり楽観バイアス。val枚数が少ないほど影響大。
      対処: **A案** 早期停止を外して固定エポックで学習しvalは報告のみ（最小改修） / **B案** trainを
      train/inner-valに分けて内側で早期停止、外側val(=test)は報告のみ（nested CV, 厳密）。
      → 手法間ベンチマークの順位を議論する前に対応。

## C. ★研究テーマ: 試験間差（バッチ）除去の表現学習
wsi id = 無料のバッチラベル。成功＝η²_slideを偶然(≈0.07)へ下げつつ、形態/生物シグナルは保持。

### C1. データ / 前処理
- [ ] 染色正規化（Macenko / Vahadane） … ベースライン。色は主因でないので単独効果は限定的の見込み
- [ ] スキャナ / JPEG圧縮のハーモナイズ

### C2. 学習でバッチ不変にする（augmentation / 正例設計）
- [ ] **DINOv2測光オーグを寄せてablation**（UNIが色に強い理由の検証／低コスト第一歩）
      - 動機: UNI(=DINOv2レシピ)は η²_color=0.087・AMI_clust_color=0.113 と最も色非依存。
        現行オーグ([src/sslmodel/utils.py](src/sslmodel/utils.py))はColorJitter(0.4,0.4,0.2,0.1)p=0.8で既にDINOv2と一致。
      - 差分は3点のみ: (1) **RandomGrayscale 0.05→0.2**（現行は意図的に低く抑えている）、
        (2) GaussianBlurを**非対称化**（片ビューp=1.0/もう片方p=0.1）＋(3) **Solarization p=0.2をview2に追加**。
        ※ DINOv2のlocal 96px/scale(0.05,0.32)マルチクロップは**真似しない**（病理パッチで微細構造が消えるため既に(0.2,1.0)採用）。
      - 実装: 既存デフォルトは壊さず新オーグプロファイルとして追加 → preview構成で1ラン → η²_slide/η²_color/CKAを比較。
      - **予想と検証点**: 白黒化でη²_slideがほぼ不変だった＝バッチは構造由来なので、**η²_colorは下がるがη²_slideは動きにくい**はず。
        これが確認できれば「測光オーグはUNIの*色*耐性は再現するが*構造バッチ*耐性は再現しない」を実証でき、本丸(下記 染色/スキャナaug)へ論拠づけできる。
      - 留意: grayscale強化はH&E色の形態シグナルも捨てる方向 → E章のバッチ除去↔生物シグナル保持トレードオフに直撃。
- [ ] 強い染色aug（RandStainNA, HEDジッター）
- [ ] **クロススライド正例**: 同一スライドの2ビューではなく、別スライドの類似パッチ or
      stain-transferで正例を作り、slide不変性を強制する
- [ ] batch/stain-transfer augmentation（別スライドの染色プロファイルを転写）

### C3. 損失 / アーキで明示的に除去
- [ ] **ドメイン敵対（DANN / 勾配反転）**でエンコーダがslideを識別できないようにする
- [ ] 特徴とslide idの独立性正則化（HSIC / 相関ペナルティ）
- [ ] 条件付きSSL（slide条件を頭に入れて表現から除去）

### C4. 事後補正（再学習不要・まず試せる）
- [ ] 学習済み埋め込みに **Harmony / ComBat / scVI流のバッチ補正**を適用（single-cell由来手法の転用）
      → η²_slideがどこまで下がるかを即測定できる、最小コストの第一歩

### C5. 評価
- [ ] 成功指標: η²_slide↓（→偶然水準）かつ形態シグナル保持
- [ ] バッチ混合指標 **kBET / iLISI**（scRNA-seq由来）を導入
- [ ] 「生物シグナル保持」の確認にはラベルが要る（→ D章）

## D. データ / ラベル（意味評価とバッチ除去の"保持"検証に必須）
- [ ] **Open TG-GATEs スライド単位アノテーション（WSI→finding / compound / dose）を入手**
      → 意味ARI・A/B/C近接、及び「バッチ除去後も生物差が残るか」の評価に使う
- [ ] or 外部ラベル付きH&E（NCT-CRC-HE-100K 等）で転移プローブ

## E. レポート / 発信
- [ ] Artifactレポートに「バッチ / 色の影響」章を追記（白黒・単一スライド・CKA多vs単の対比図）
- [ ] 本比較（4手法）の最終レポート
- [ ] （まとまれば）バッチ除去の結果を独立レポートに

## 未解決の問い
- 構造バッチの正体（スキャナ / 切片 / 圧縮）の切り分け・寄与度
- バッチ除去 ↔ 生物シグナル保持のトレードオフはどこまで両立するか
- **UNI等の基盤モデルは試験間差にどれだけ頑健か**（同じη²_slideで測れる。大規模事前学習が
  バッチ依存を下げているかの検証は面白い）

## とりたいデータ
- 各モデルの学習時のロスの減少・Effrankの挙動
- 各モデルの学習時の内部表現クラスタリング
      - バッチ効果の評価をしたいのでスライドごとに色分けする
      - 5EPに一回モデルを保存しているのでそれで取得する
- 各モデルの内部空間比較（CKA・ARI）
