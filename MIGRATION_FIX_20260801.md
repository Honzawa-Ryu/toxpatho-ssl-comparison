# 移植merge 誤配置修正レポート（2026-08-01）

対象リポジトリ: `toxpatho-ssl-comparison`（ブランチ `260727_dev`）
関連: `wsi-ad` の `.agent-runs/20260801_132644_campaign_claude_migrated-.../artifacts/report.md`
（フェーズB研究コード移植の元レポート）

## 経緯

元レポートは `git subtree split` で作成した2つのブランチ
（`migrate/lib-only`, `migrate/experiments-only`）を
`git merge --allow-unrelated-histories` で取り込めば移植完了、としていた。
ユーザーが実際に取り込み後のリポジトリを確認したところ
「移植されているように見えない」との指摘があり、検証したところ
**mergeコマンド自体は成功していたが、内容が壊れていた**ことが判明した。

## 発見した不具合

`git subtree split` で作られるブランチは、指定した prefix（`lib/` や
`experiments/`）を含まない相対パスでツリーが記録される。そのため
`--allow-unrelated-histories` で素直に merge すると、本来
`lib/` 配下・`experiments/` 配下に入るべきファイルが
**リポジトリのルート直下にそのまま展開されてしまう**。

merge直後の状態:

| 項目 | 内容 |
|---|---|
| ルート直下に誤って出現 | `analysis/`, `evaluate/`, `model/`, `sslmodel/`, `trainer/`, `wsi.py`（本来 `lib/` 配下）、実験ディレクトリ40件（本来 `experiments/` 配下） |
| 重複ファイル | ルート直下の `output_utils.py`（`lib/output_utils.py` と内容完全一致）、空の `__init__.py` |
| `lib/`, `experiments/` の中身 | ほぼ変化なし（`lib/` は既存の4ファイルのみ、`experiments/` は空のまま） |

コンフリクトが一切発生しなかったのは、パスが `analysis/foo.py` と
`lib/analysis/foo.py` のようにそもそも異なっていたため、
git の衝突判定が働かなかったことによる。

## 修正内容（コミット `448f28b`）

`git mv` で以下を正しい位置へ再配置し、重複ファイルを削除した。

- `analysis/`, `evaluate/`, `model/`, `sslmodel/`, `trainer/`, `wsi.py`
  → それぞれ `lib/` 配下へ
- ルート直下の実験ディレクトリ40件（`0001_...`〜`0009_...`,
  `20260507_...`〜`20260714_...`）→ `experiments/` 配下へ
- ルート直下の重複 `output_utils.py`, `__init__.py` → 削除
  （`lib/output_utils.py`, `lib/__init__.py` は既存のものを維持）

git のレンダーする `R  __init__.py -> lib/analysis/__init__.py` のような
rename表示は、内容が空の `__init__.py` 同士をgitの類似度検出が
たまたま対応付けただけの表示上の癖であり、実害はない
（対応する add/delete は個別に正しいパスへ行われている）。

## 修正後の検証結果

```
lib/
├── analysis/ (tests/含む)
├── evaluate/
├── model/
├── sslmodel/ (models/含む)
├── trainer/
├── __init__.py
├── output_utils.py
├── test_output_utils.py
├── test_preflight_check.py
└── wsi.py

experiments/  … 40件（0001〜0009、20260507〜20260714の各実験ディレクトリ）
```

- ルート直下に想定外のディレクトリ・ファイルが残っていないことを確認済み
- `git status` はクリーン
- `lib/output_utils.py` は移行前の内容と一致（diffなし）

## 未実施・要フォローアップ

1. `origin` への push は未実施（ローカルで `260727_dev` が
   `origin/260727_dev` より30コミット進んだ状態）
2. `data/`・`result/`・`outputs/`・`wandb/` の扱い方針は元レポート同様
   未決（symlink / 実体rsync / 環境変数化）
3. 移行後の `wsi-ad` リポジトリの扱い（アーカイブ/並行稼働等）も未決
4. `wsi-ad-local` の一時remoteは削除済み（副産物なし）
