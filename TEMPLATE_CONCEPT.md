# 実験管理テンプレート設計コンセプト (TEMPLATE_CONCEPT.md)

本テンプレートは、SlurmおよびPBS (Miyabi) 環境において、研究実験の**「高い再現性」「資源の最適化」「実験サイクルの高速化」**を両立し、AIエージェント（Antigravity/Claude Code等）と人間が最も効率よく協調して研究開発を進めるためのプラットフォームです。

---

## 1. 設計理念 (Design Philosophy)

本テンプレートは、以下の4つの原則に基づいて設計されています。

### ① 「1実験 = 1フェーズ」の単一責任原則
ひとつの実験ディレクトリに前処理・学習・評価・集計などの複数フェーズを混在させません。
*   実験をフェーズ単位（例: `fetch_patients` → `preprocess_sequences` → `train_mamba` → `eval_mamba`）に細分化することで、コードをシンプルに保ち、中間データの再利用性を最大化します。

### ② 「1ジョブ = 1原子単位」の再現性原則
`experiment.py` 内でモデル、シード、プロンプトなどの可変パラメータをループで回して複数回実行することを禁止します。
*   1ジョブは必ず「パラメータの1つの組み合わせ」のみを処理する「一原子実行」とします。
*   複数組み合わせの探索は、スクリプト内ループではなく、`run_slurm.sh`側でarray/seqモードを
    有効化した上でのGRID直積展開で解決します（詳細は`USAGE.md`参照）。

### ③ 「Soft Rules + Hard Hooks」の二段構えの防衛
ルールをドキュメント（capsule側の `common/AGENTS.md`）に記載するだけの「ソフトルール」だけでは、エージェントや人間の遵守率は低くなります。
*   本テンプレートは agent 設定を持ちません。ルールは capsule 側（`slurm-agents-capsule` の `common/AGENTS.md` と各 hook）が担い、`make preflight` のような静的解析コマンドはユーザーが手動で呼ぶか、agentがcapsule内から実行します。

### ④ 「YAGNI (You Aren't Gonna Need It)」と「Surgical Changes」
余計な抽象化や「将来使うかもしれない」コードの実装を排除し、今解くべき課題に対する最小限の実装に留めます。
*   実験コードが肥大化・重複した場合は、速やかに共通ライブラリ `lib/` へ統合（`consolidate`）します。

---

## 2. なにができるか (Capabilities)

本テンプレートによって実現される主な機能・自動化は以下の通りです。

| 機能 | コマンド / 仕組み | 役割 |
|---|---|---|
| **実験自動作成** | `make create_exp name=<名前>` | テンプレートから自動採番（4桁ID）で実験ディレクトリを作る（Git操作は行わない。ブランチは daily/YYMMDD 方式でユーザーが `gstart` で作成済みのものを使う）。 |
| **自動事前検証** | `make preflight` | `config.yml` との実装キー不整合、シード固定漏れ、ハードコード書き込みの有無を一撃で静的解析する。 |
| **安全なジョブ投入** | `runx <id>`（capsule内では `sbatch` 直接投入が禁止されているため、agentは `artifacts/proposed_train.sbatch` を書いてユーザーに委ねる） | ログインノードでの誤実行防止、二重実行警告、GRID組み合わせ数が上限（50件）を超える場合の過剰投入警告を自動で行う。 |
| **失敗・ゾンビログ消去** | `make clean_failed` | `FAILED`や`TIMEOUT`のジョブフォルダだけでなく、24時間以上動きのないハングした`RUNNING`ジョブや、リネーム等で孤立した実験ログを自動で検出・削除する。 |

---

## 3. どうしたいか / 今後の展望 (Direction)

本システムが目指す究極のゴールは、**「思考スピードでの研究検証サイクルの実行」**です。

### ① 再現性 100% の実験エコシステム
*   すべての実験結果が `completion.json` および Parquet 形式で保存されることで、1年後であっても特定の実験結果を1コマンドで再現できる状態を目指します。

### ② クリーンなコードベースの維持
*   実験が進むにつれて肥大化するゴミログや、使われなくなった共通コード（ゾンビ関数）を、Makefileのターゲットやクローラーを通じて常に自律的に排除し、いつでも見通しの良いコードベースを保ちます。

### ③ エージェントとの高速なサイクル
*   人間が「仮説」を立て、エージェントが「計画書（plan.md）作成 → 事前検証（preflight） → 実行（runx、投入自体はユーザー） → 結果確認 → 共通化（consolidate-lib）」の一連のサイクルを回すことで、人間は科学的主張や仮説のブラッシュアップに集中できます。

---

## 4. 実験ごとの注意事項を持たせる（Antigravity向け・テンプレート直下インデックス方式）

### 4-1. `EXPERIMENT_NOTES.md`（テンプレート直下）

```
template-daily-research-experiments/
├── EXPERIMENT_NOTES.md   ← 全実験の固有注意事項をここに集約
├── experiments/
│   └── 0007_20260701_my-exp/
├── lib/
└── ...
```

実験IDごとに見出しを立てて追記していく、単純な追記型のファイルです。

```markdown
# EXPERIMENT_NOTES.md

実験固有の注意事項をIDごとに記録する。plan-next-experimentでplan.mdを書く際、
review-expで結果を集約する際、debug-experimentで調査する際は、対象実験のIDに
該当する節があればまず読むこと。

## 0005_20260615_preprocess_sequences

- 出力する`variant_key`は正規化済みID前提。upstream側でこのIDが変わったら再実行が必要。

## 0007_20260701_train_mamba_ehr

- upstream の 0005 は正規化済みIDを前提にしている。前処理を変更した場合は upstream 側の再実行が必要。
- 評価指標は accuracy ではなく macro-F1 を使う（クラス不均衡のため）。

## 0012_20260710_run_ablation

- Ablation実行手順: `run_slurm.sh`のRUN_COMMANDに`--ablation-mode`を追加する。
  詳細な組み合わせ表は `experiments/0012_.../ablation_matrix.md` を参照。
```

新しい実験に取り掛かる際、その実験固有の申し送り事項があれば、実験ディレクトリの中にファイルを
増やすのではなく、この`EXPERIMENT_NOTES.md`に節を追加してください。特殊な手順(旧`.claude/skills/`
相当)も、専用スキルファイルとしてではなく、この中に手順を文章で書く形に統一します。


### 4-2. 注意点

- 1ファイルに全実験分が集約されるため、実験数が増えると肥大化します。古い実験の節は
  `consolidate-lib`のタイミングで要点だけ残して圧縮する、または`archive/`セクションに
  まとめて移動するなど、定期的な整理を想定してください。


## 5. GitHubでの管理方針 (Pattern2 / daily branch方式)

Pattern1の`rules/git.md`が定める「型A＋固定ブランチ(`machine/<hostname>`)」方式とは異なり、
Pattern2では**`daily/YYMMDD`の日次ブランチ**でGitを運用します。エージェントはcapsule内で
作業するため、ブランチの切り替え・マージ・pushはすべて人間がホスト側で行い、エージェントは
「今日のdailyブランチの上でコミットを積む」ことだけを行います。

### 5-1. 1日のサイクル

```
朝: gstart          mainをpullし、daily/YYMMDD ブランチを作成/checkout（ユーザーがホスト側で実行）
日中: agentrun assist ... daily/YYMMDD ブランチ上でagentが作業・コミット
夜: gpush            daily/YYMMDD をpush → main へ --no-ff マージ → main をpush（ユーザーがホスト側で実行）
```

`gstart`/`gpush`はagentが実行するコマンドではありません(`.bashrc.d/2-git.sh`のシェル関数)。
エージェントはこの2つの存在を知っている必要はありますが、呼び出しません。

### 5-2. エージェントの権限境界(hookで物理的に強制)

`git_branch_guard.py`(全エージェント共通)により、以下はcapsule内から常にブロックされます。
`bash x.sh`のようにローカルスクリプト経由で実行しようとした場合も、スクリプトの
中身を読んで同様にブロックされる(Write→Bash実行による迂回はできない)。

- `main`/`master`へのcheckout/switch
- 新規ブランチの作成(`checkout -b`/`switch -c`)
- `main`/`master`上での`commit`
- `git merge`/`git rebase`
- `git push`
- ブランチの強制削除(`git branch -D`)
- `.git`ディレクトリへの直接操作

Claudeの`settings.json`にも同種のdeny listがあるが、`--dangerously-skip-permissions`
起動時はこのdeny listが実質的に機能しないため、実効的な防衛線は上記hookのみと考える。

エージェントに許可されているのは、**人間が`gstart`で用意した`daily/YYMMDD`ブランチの上で
コミットを積むところまで**です。ブランチの作成・切り替え・push・mainへのマージは全て人間の役割です。

### 5-3. 並行作業の防止

同一リポジトリに対して複数の`agentrun`を同時に起動することはできません。`submit.sbatch`/
interactive起動のいずれも、リポジトリ単位のロック(`<repo>/.agentrun.lock`)を`flock`で取得し、
既に他のrunが実行中の場合は起動を拒否します。1つのdailyブランチに複数エージェントが
同時にコミットして競合する事態を防ぐための仕組みです。

### 5-4. 誤ったリポジトリへのコミット防止

git hooksの`core.hooksPath`にpre-commit hookを仕込み、`remote.origin.url`のowner名が
`mizuno-group`の場合はコミット自体をブロックします(`git commit --no-verify`で明示的に
バイパスした場合のみ許可)。これは`gstart`/`gpush`と同じくホスト側の個人設定として管理し、
テンプレート・capsuleどちらのリポジトリ本体にも含めません。

### 5-5. PRレビューについて

`gpush`のデフォルト挙動は、`daily/YYMMDD`を一度originへpushした後、**ローカルでmainに
マージしてからmainを直接originへpushします**。GitHub上のPull Requestを経由した
レビューのステップはデフォルトでは含まれていません。

コードレビューを必須にしたい場合は `gpush -gh` を使います。ローカルマージの代わりに
`gh pr create`でPRを作成し、URLを表示してy/n確認、`y`なら`gh pr merge`でマージする
流れに切り替わります(要`gh` CLIのインストールと`gh auth login`)。`gh`はagentが実行する
コマンドではなく、`gpush`と同様にユーザーがホスト側で実行するものです。