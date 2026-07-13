# 関連研究サマリ（wsi-ad: 非対照SSLの表現比較）

最終更新: 2026-07-13
関連: [Goal.yaml](Goal.yaml) ・ [TODO.md](TODO.md) ・ レポート `result/representation_analysis_preview/report.html`

WebSearch/WebFetch で精査した先行研究を「条件 / 比較対象 / 手法 / 結論 / 本研究との差分」で整理。
この分野は 2024-2025 で急速に動いており、「病理で初」ではない。新規性は **統制（同一データ・backbone・optimizer）** と
**因果分解（色 vs バッチ vs 形態）** にある。

---

## ① Park, Kim, Heo, Kim, Yun — "What Do Self-Supervised Vision Transformers Learn?" (ICLR 2023)
- **URL**: https://arxiv.org/abs/2305.00729 ・ code: https://github.com/naver-ai/cl-vs-mim
- **条件**: 自然画像(ImageNet)、ViTバックボーン。対照学習(CL: MoCo系) vs マスク画像モデリング(MIM: MAE系)。
- **手法**: 周波数(Fourier)分析、自己注意パターン/注意距離、層別の線形分離性・表現多様性。
- **結論**: **CL＝形状・大域・低周波・後段層重視／MIM＝テクスチャ・局所・高周波・前段層重視**。両者は相補的で「単純な組合せでも両者の利点を活かせる」。
- **本研究との関係**: 「何を学習するか」の基準系。**MAEの色主導は、MIMのテクスチャ/高周波バイアスが病理で"染色色"として顕在化した姿**と解釈できる。→ 引用で足り、再学習不要。問い＝「自然画像のこの二分法は、バッチ交絡のある病理でも成り立つか」。

## ② Kang et al. (Lunit) — "Benchmarking Self-Supervised Learning on Diverse Pathology Datasets" (CVPR 2023)
- **URL**: https://lunit-io.github.io/research/publications/pathology_ssl/
- **条件**: TCGA 20,994 + 内部(TULIP) 15,672 WSI = **3260万パッチ**、20x/40x、200ep、V100×64。
- **比較対象**: **MoCo v2 / Barlow Twins / SwAV（ResNet50）+ DINO（ViT-S/16, /8）** の4手法。
- **手法**: 下流タスク（BACH/CRC/MHIST/PCam分類 + CoNSeP segmentation）を linear probe / fine-tune。
- **結論**: **「明確な勝者なし」**。傾向: BT=linear probe良／MoCo=fine-tune良／DINO=分類でしばしば最良。手法選択より**ドメイン整合データ・aug**が重要。
- **本研究との差分**: **下流精度のみ・表現レベル分析なし・バッチ観点なし**。最も近い"SSL手法比較"だが、目的関数が表現に何を刻むかは見ていない。

## ③ "Do Histopathological Foundation Models Eliminate Batch Effects? A Comparative Study" (2024)
- **URL**: https://arxiv.org/abs/2411.05489
- **条件**: **9基盤モデル**（Ciga/HIPT/RetCCL/Kang-DINO/CTransPath/Phikon/**UNI**/Prov-GigaPath/**Virchow**）。TCGA-LUSC(5施設, 25万パッチ) + CAMELYON16(2施設, 10万パッチ)、256px/20x。
- **比較対象**: 施設(TSS=tissue source site)＝染色/スキャナ/固定の違い＝「病院signature」。
- **手法**: 施設予測精度（NCC/kNN/線形プローブ）、下流精度の劣化、特徴空間距離、PCA。
- **結論**: **全FMが施設signatureを保持**（線形プローブで施設予測90%超、CAMELYONほぼ完全）。**特徴距離は生物差より施設差に支配**。**染色正規化では消えない**（正規化後も施設予測80-97%）。大規模事前学習でもバッチは消えない。
- **本研究との差分**: **既存FMの相関的分析**。「どのSSL目的関数がバッチを吸収するか」をアルゴリズム統制していない。※本研究の「白黒でもη²_slide不変」と整合。

## ④ "Comparing Computational Pathology Foundation Models using Representational Similarity Analysis" (2025) ★手法論が最接近
- **URL**: https://arxiv.org/abs/2509.15482
- **条件**: **6基盤モデル**（視覚言語対照: CONCH/PLIP/KEEP ／ 自己蒸留: UNI2/Virchow2/Prov-GigaPath）。TCGAのH&Eパッチ。
- **比較対象**: モデル間の表現構造類似性、slide依存 vs disease依存、染色正規化の効果、学習パラダイム(視覚のみ/視覚言語)の影響。
- **手法**: **表現類似性分析(RSA、神経科学由来、≒CKA的)** + 内在次元。
- **結論**: **全モデルが高slide依存・低disease依存**（＝本研究のη²_slide支配と同結論）。**染色正規化はslide依存を5.5〜20.5%しか下げない**（＝本研究の白黒でほぼ下がらない、と整合）。学習パラダイムだけでは表現の類似性は決まらない。Prov-GigaPathは他と似、UNI2/Virchow2は独特。
- **本研究との差分**: **異種の既存FM同士**（データ/規模/レシピがバラバラ）の比較。**非対照アルゴリズムを揃えて学習した比較ではない**。色/構造/形態の**因果分解(白黒・単一スライド)は無い**。

## 補助的に確認した文献
- **"Understanding Masked Autoencoders From a Local Contrastive Perspective"** (MAEの再構成目的を局所対照として解釈): https://arxiv.org/html/2310.01994v2
- **"Towards robust foundation models for digital pathology"** (Nature Communications, バッチ頑健性): https://www.nature.com/articles/s41467-026-73923-2

---

## 総合：確立済み vs 本研究の新規性

**もう確立されている（＝新規性として主張できない）:**
- 「病理表現はslide/施設バッチに支配される」→ ③④で2024-2025に定量報告済み。
- 「染色正規化では消えない」→ ③④で報告済み（本研究の白黒結果と一致）。
- 「CL vs MAEで学習内容が違う」→ ①で自然画像既出。

**まだ手薄で、本研究が主張できる差分:**
1. **統制されたアルゴリズムレベル比較** — 先行(③④)は異種の既存FM比較で目的関数を分離できない。本研究は**同一データ・同一ViT・同一AdamW**で非対照4手法を揃え、**目的関数そのものの効果を単離**。
2. **色 vs バッチ vs 形態の因果分解** — η²_slide/**白黒**/**単一スライド**で支配要因を切り分ける機構的分析。先行は相関的(施設予測精度/RSA)に留まる。
3. **非対照ファミリー(SimSiam/BT/MAE/DINO)特化** + **TG-GATE毒性病理(肝・薬剤誘発)** という薄いドメイン。

**「一般画像との比較は要るか」:**
- 新規性のためには**不要**（①を引用すれば足りる。バッチはImageNetに無い病理固有の軸）。
- 位置づけとして「**自然画像の CL/MIM 二分法が、バッチ交絡のある病理でどう崩れる/保たれるか**」と framing すると強い。査読対策の"強化材料"として軽い自然画像コントロールは有効だが必須ではない。
