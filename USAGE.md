# USAGE

このテンプレートを実際に手を動かして使うための実践ガイドです。設計思想は
[`TEMPLATE_CONCEPT.md`](./TEMPLATE_CONCEPT.md)、関数・スクリプトの一覧リファレンスは
[`FUNCTIONS.md`](./FUNCTIONS.md)、セットアップ手順は [`README.md`](./README.md) を参照してください。
本ドキュメントは「日々の運用でどう使うか」を、特にGit運用・実験管理・実行方法にフォーカスして
手順ベースで説明します。

---

## 1. Git運用

### 1-1. ブランチモデル（daily/YYMMDD方式）

このテンプレートは **daily/YYMMDD** 単位のブランチで運用します。1実験ディレクトリごとの
専用ブランチは作りません（旧方式からの変更点。`make create_exp` はディレクトリを作るだけで
Git操作は一切行いません）。

```
朝: gstart          main を最新化し、今日の daily/YYMMDD ブランチを作成/checkout
日中: agentrun ...   daily/YYMMDD ブランチ上でagentが作業・コミット
夜: gpush            daily/YYMMDD を push → main へ --no-ff マージ → main を push
```

`gstart`・`gpush` は **ユーザーがホスト側で実行するシェル関数**です。実体は
[`.bashrc.d/2-git.sh`](./.bashrc.d/2-git.sh) にありますが、このファイル自体は
テンプレート/capsuleリポジトリ本体には含めない設計です（冒頭コメント参照）。
自分の `~/.bashrc` など、テンプレート外の個人設定に `source` して使ってください。
agentはこの2つの存在を知っている必要はありますが、呼び出すことはありません
（後述のhookで物理的にブロックされます）。

### 1-2. `gstart` の中身（朝の作業）

```bash
gstart
```

1. 未コミットの変更があれば失敗する（先に commit/stash が必要）
2. `git checkout main && git pull origin main` で main を最新化
3. `daily/YYMMDD`（今日の日付、`%y%m%d`）ブランチを作成、または既存なら checkout
4. 他マシン/runで既に同名の `daily/YYMMDD` が origin にあれば `--ff-only` で取り込む
   （fast-forwardできない場合は警告のみで、手動対応が必要）

### 1-3. 日中の作業（agentが行うこと）

`gstart` 済みの `daily/YYMMDD` ブランチの上で、`agentrun` を起動して作業します。

```bash
agentrun assist --repo /path/to/this/repo "..."
```

**同一リポジトリへの並行run（同時に複数の `agentrun` を起動すること）はできません。**
`<repo>/.agentrun.lock` を `flock` で取得するロックが働き、既に他のrunが実行中なら起動を拒否します。

agentができるのは、**既に daily/YYMMDD ブランチ上にいる状態でコミットを積むこと**までです。

| 操作 | 可否 | 備考 |
|---|---|---|
| `git status` / `git diff` / `git log` | ✅ | 状況確認 |
| `git add` / `git commit` | ✅（daily/YYMMDD上のみ） | main/master上でのcommitはhookでブロック |
| `git checkout main` / `git switch main` | ❌ | `git_branch_guard.py` でブロック |
| `git checkout -b <new>` / `git switch -c <new>` | ❌ | 新規ブランチ作成はユーザー（`gstart`）の役割 |
| `git merge` | ❌ | `--abort`/`--continue`/`--quit` のみ許可 |
| `git rebase` | ❌ | `--abort`/`--continue`/`--quit`/`--skip` のみ許可 |
| `git push` | ❌ | `gpush` はユーザーが実行 |
| `git branch -D` | ❌ | 削除はユーザーに依頼 |
| `.git/` への直接操作 | ❌ | 常にブロック |

これらは `slurm-agents-capsule/common/hooks/git_branch_guard.py` により hook レベルで
強制されます（ドキュメントに書くだけの「お願い」ではありません）。agentが上記操作を
必要とする場面（ブランチを切り替えたい、pushしたい等）に遭遇したら、ユーザーに依頼して
待ちます。

### 1-4. `gpush` の中身（夜の作業）

```bash
gpush
```

前提: 現在のブランチが `daily/*` であること。未コミットの変更があると失敗します。

1. `git push origin <daily/YYMMDD>`
2. `git checkout main && git pull origin main`
3. `git merge --no-ff <daily/YYMMDD> -m "merge <daily/YYMMDD> into main"`
4. `git push origin main`

デフォルトでは**PRを経由したレビューのステップを含まず**、ローカルでmainにマージして
そのままpushします。

#### `gpush -gh`: PR経由でレビューしたい場合

```bash
gpush -gh
```

`git checkout main && merge && push` の代わりに、`gh` CLI（[cli.github.com](https://cli.github.com/)、
事前に `gh auth login` が必要）でPRを作成する経路に切り替わります。

1. `git push origin <daily/YYMMDD>`
2. `gh pr create --base main --head <daily/YYMMDD> --fill`（既存PRがあれば流用）
3. PRのURLを表示
4. y/n確認
5. `y` なら `gh pr merge --merge --delete-branch=false` で main へマージ、`n` ならマージせず終了（後でGitHub上から手動マージ可能）

`gh` が未インストール・未認証の場合はエラーで停止します（`git push`自体は実行済みなので、
daily/YYMMDDブランチはoriginに残ります）。

### 1-5. `mizuno-group` への誤コミット防止

`_setup_git_config`（`.bashrc.d/2-git.sh` の一部）が対話シェル起動時に自動セットアップする
`core.hooksPath` 付き pre-commit hook です。リモートのownerが `mizuno-group` の場合、
コミット自体をブロックします（`git commit --no-verify` で明示的にバイパスした場合のみ許可）。
これはホスト側の個人設定として一度だけセットアップされ、テンプレート/capsuleリポジトリ
本体には含まれません。

### 1-6. 実験作成時のuncommitted diffの扱い

`make create_exp` 実行時、その時点で未コミットの変更があれば
`experiments/{id}_{date}_{name}/uncommitted_changes.diff` として自動保存されます
（`git diff HEAD` の出力）。また `metadata.yaml` に `git_commit`（実行時のHEAD）が記録されます。
これにより「その実験を作った時点のコード状態」を後から追跡できます。

### 1-7. `runx` のdirtyチェック

`runx` は投入前に `git diff` / `git diff --cached` が空でないかを確認し、**未コミットの
変更があると投入を拒否します**。実装が終わったらコミットしてから投入するのが基本です。
コミットせずにどうしても投入したい場合のみ `runx ... --allow-dirty` を使えますが、
再現性が失われるため通常はコミットを優先してください。

---

## 2. 実験管理

### 2-1. ディレクトリ構造

```
experiments/{id}_{date}_{name}/
    experiment.py     # エントリポイント（データロード・学習・評価ロジック）
    config.yml         # ハイパーパラメータ
    run_slurm.sh        # 完全なsbatchスクリプト（#SBATCHヘッダー + 実行コマンド + GRID定義）
    metadata.yaml       # 自動生成: exp_id, exp_name, created_date, git_commit
    uncommitted_changes.diff  # 作成時にdirtyだった場合のみ生成
experiments/latest -> experiments/{最新}/    # symlink

outputs/{id}_{date}_{name}/{variant_key}/    # experiment.py が書き込む先
    completion.json     # status: running/completed, guard_skipped_at等
logs/{id}_{date}_{name}/{job_id}/            # command.sh, run_metadata.yaml, slurm.out
```

**1実験＝1フェーズ、1ジョブ＝1原子単位**が原則です（詳細は `TEMPLATE_CONCEPT.md`）。
可変パラメータ（model/seed/prompt等）は `experiment.py` 内でループせず、
`run_slurm.sh` の `GRID_ARGS`/`GRID_VALUES` の直積として外部展開します。

### 2-2. 実験を作成する

```bash
make create_exp name=<exp_name>
```

`tools/create_exp.sh` が実行され、以下を行います（**Git操作は一切行いません**）。

1. `experiments/` 内の既存ディレクトリから4桁IDを走査し、最新+1を採番（初回は `0001`）
2. 実験名を `英数字・アンダースコア・ハイフンのみ` にサニタイズ
3. `{ID}_{YYYYMMDD}_{sanitized_name}/` ディレクトリを作成し、`templates/` から
   `run_slurm.sh`・`experiment.py`・`config.yml` をコピー。`run_slurm.sh`は以下の
   プレースホルダを実際の値に置換した上でコピーされる:
   - `__EXP_NAME__` → 実際のディレクトリ名
   - `__PROJECT_ROOT__` → リポジトリのルート絶対パス
   - `__PARTITION__` → テンプレートの`#SBATCH --time`初期値からowner+time-scaleを
     自動解決した値（例: `x-large-david01`）
   - `__SIGNAL_MARGIN__` → 同じ初期時間から計算したtime-limit警告用マージン（秒）

   **これらは作成時に一度だけ焼き込まれる。** 後で`run_slurm.sh`の`--time`を
   大きく変えてscaleが変わる場合、`--partition`/`--signal`は自動追従しないので
   手動で合わせて編集すること。
4. `metadata.yaml` を生成（`exp_id`/`exp_name`/`created_date`/`git_commit`）
5. dirtyな変更があれば `uncommitted_changes.diff` に保存
6. `experiments/latest` / `outputs/latest` symlink を更新

命名規則は `{動詞}_{対象}`（スネークケース・英語）。`test_xxx`（pytestと紛らわしい）・
`new_xxx`/`retry_xxx`（意味がない）・`experiment001`（内容が分からない）は避けてください。
詳細な動詞例は `TEMPLATE_CONCEPT.md` の命名規則表を参照。

作成後、`experiment.py`・`config.yml`・`run_slurm.sh` を編集します。役割分担:

| ファイル | 置くもの |
|---|---|
| `config.yml` | 固定ハイパーパラメータ（探索しないもの） |
| `run_slurm.sh` | リソース設定（`#SBATCH`ヘッダー）と、探索する次元（`GRID_ARGS`/`GRID_VALUES`、必ず `--arg` 経由で渡す）。single/array/seqの切り替えもここで行う |
| `experiment.py` | 実験ロジック本体。ハイパーパラメータの直書きは禁止、`config.yml` から読む |

### 2-3. 出力の書き方（`lib/output_utils.py`）

```python
from lib.output_utils import get_run_dir, write_run_metadata, complete_run

run_dir = get_run_dir(project_root, __file__, variant_key)   # completedガード + mkdir
write_run_metadata(run_dir, model=model_id, seed=seed, ...)  # status=running
# ... 処理 ...
complete_run(run_dir)                                          # status=completed
```

`variant_key` には結果に影響するすべての次元（model/seed/prompt等）を含めます。
`get_run_dir` は `outputs/{exp}/{variant}/completion.json` の `status` が
`completed` なら `sys.exit(0)` で即座に再実行をブロックします（`completed`
ディレクトリの中身は不変）。再実行したい場合はディレクトリを手動削除してから
再投入してください。

`USE_LOCAL_SSD_OUTPUT=1`（`templates/experiment.py`がデフォルトで対応）の場合、
`OUTPUT_ROOT`環境変数（`scripts/slurm_entry.sh`が`SCRATCH_DIR/outputs`にセット）が
`get_run_dir`の第4引数`output_root`として渡され、実際の書き込み先はscratchに
なります。ただしcompletedガードの判定自体は常に`project_root`（`/workspace`直下）側の
`completion.json`を見るため、scratchが空でも誤って未実行扱いにはなりません。
出力はジョブ終了時に`scripts/slurm_entry.sh`が`outputs/{exp_name}/`へrsyncで
回収します。

`variant_key` に `/`（スラッシュ）を含めると、`get_run_dir` が内部で呼ぶ
`sanitize_variant_key` が自動的に最後の`/`以降だけを使います（例:
`"google/gemma-4-31b-it"` → `"gemma-4-31b-it"`）。これは `outputs/{exp}/{variant}/` を
1階層のフラット構造に保つための挙動で、`review-exp`/`roadmap` が使う
`outputs/*/*/completion.json` というglobパターンが前提にしています。自動変換された
場合は警告が出るので、可能であれば`model_short`側で最初からスラッシュを除去した
短縮名を使う方が確実です（元のキーは `run_dir/.variant_key_original.txt` に記録されます）。

**時間制限によるデータ消失を防ぐため、結果はメモリに溜め込まず逐次書き出してください。**

### 2-4. 実験固有の注意事項（`EXPERIMENT_NOTES.md`）

実験ディレクトリの中に個別ファイルを増やすのではなく、テンプレート直下の
`EXPERIMENT_NOTES.md` に実験IDごとの見出しで追記していきます。

```markdown
## 0007_20260701_train_mamba_ehr

- upstream の 0005 は正規化済みIDを前提にしている。前処理を変更した場合は再実行が必要。
- 評価指標は accuracy ではなく macro-F1 を使う（クラス不均衡のため）。
```

対象実験のIDに該当する節があれば、計画作成・結果集約・デバッグの際に必ず確認してください。
実験数が増えたら、古い節は要点だけ残して圧縮する等、定期的な整理を想定しています。

### 2-5. 投入前の静的検証（`make preflight`）

```bash
make preflight
```

最新の実験ディレクトリに対し、以下を一括チェックします（`scripts/preflight_check.py`）。

- `config.yml` のトップレベルキーと `experiment.py` 内の参照キーの不整合
- `run_slurm.sh` の `GRID_VALUES` 直積数が上限（50件）を超えていないか
- `GRID_ARGS`が定義されているのに`RUN_MODE`が`array`/`seq`になっていない
  （single modeのまま投入すると1組しか実行されない、というミスの検出）
- `RUN_MODE="array"`のとき、`#SBATCH --array=0-N`のNがGRID直積数と一致しているか
- `experiment.py` 内で `outputs/`/`data/` へのハードコード書き込み（`to_parquet`/`to_csv`/`open`）
  がないか

全てOKなら `✅ preflight OK`、NGがあれば理由付きで非ゼロ終了します。`runx` の前に一度
通しておくと、投入後に気づくミスを減らせます。

### 2-6. リネーム

```bash
make rename_exp name=<exp_id_or_name> new=<new_name>
```

### 2-7. 失敗マーク

```bash
make mark_fail name=<exp_id_or_name> reason=<reason>
```

`experiments/{...}/` と対応する `outputs/{...}/` の両方を
`{元の名前}_FAILED_{date}_{reason}` にリネームします。`FAILED_REASON.txt`
（`failed_at`/`reason`/`original_experiment`）が追加されます。既に `_FAILED_` 付きの
場合は何もせず終了します。`latest` symlinkがそのマーク対象を指していた場合は自動で
付け替えられます。

### 2-8. 再開

```bash
# 最新の実験を再開（suffixデフォルトは "resumed"）
make resume_exp name=<exp_id_or_name>

# suffixを指定して再開
make resume_exp name=<exp_id_or_name> suffix=retry
```

新しいIDで `experiments/{new_id}_{today}_{base_name}_{suffix}/` を作成し、元の実験の
`run_slurm.sh`・`experiment.py`・`config.yml` 等をコピーします
（`FAILED_REASON.txt`・`uncommitted_changes.diff`・`metadata.yaml`・`__pycache__/` は除外）。
コピー後、`run_slurm.sh`/`config.yml` 内の旧実験名参照は新しい実験名に自動置換されます。
元の実験で `#SBATCH --dependency=` がアクティブだった場合は、古いjob_idを無自覚に
引き継がないよう自動的に無効化（再コメントアウト）されます。必要なら新しいjob_idで
手動で有効化し直してください。
`metadata.yaml` に `resumed_from_exp`（元の実験名）と `resume_checkpoint_dir`（元の出力先）
が記録されるので、チェックポイントの参照パスは `config.yml` 側で明示的に指定してください。

### 2-9. ログの掃除

```bash
make log_clean      # {job_id}_{exp_name}.out 等をjob_idサブディレクトリに整理
make clean_failed    # FAILED/TIMEOUT/OOM等のジョブログ、24h以上ハングしたRUNNINGログ、
                      # 実験ディレクトリが存在しない孤立ログを削除
```

`outputs/`・`logs/`・`data/`・`lib/` の削除を伴う操作は、実行前に必ず影響を確認してください。

---

## 3. 実行の仕方

### 3-1. `runx` — ジョブ投入

`run_slurm.sh` 自体が完全な`#SBATCH`ヘッダーを持つ投入スクリプトなので、`runx`は
それをほぼそのまま`sbatch`するだけです。リソース（GPU/CPU/mem/time/partition）・
実行モード（single/array/seq）・依存ジョブ指定はCLIでは指定せず、
**`run_slurm.sh`を直接編集**します。`runx`のオプションは`--allow-dirty`のみです。

```bash
runx                          # 最新の実験を投入
runx 1                        # 実験ID指定
runx --allow-dirty            # 未コミット状態のまま投入（非推奨、再現性が失われる）
```

**投入の流れ**:

1. Git dirtyチェック（1-7節参照）
2. 実験ディレクトリを解決（IDまたは最新）
3. `logs/{exp_name}/` を作成（`#SBATCH --output`の出力先ディレクトリが無いと`sbatch`が失敗するため）
4. `sbatch experiments/{id}/run_slurm.sh` を直接呼ぶ
5. 投入成功後、`outputs/{exp_name}/latest_job_id.txt` に job_id を保存

### 3-1-1. 他の実験のジョブに依存させたい場合

依存先実験の`outputs/{依存先exp}/latest_job_id.txt`を確認し、依存させたい実験の
`run_slurm.sh`にある以下の行を有効化してjob_idを埋めます。

```bash
#SBATCH --dependency=afterok:<job_id>
```

投入のたびに依存先が変わりうる値なので、都度手動で書き換える前提です
（`runx`側で自動解決はしません）。

partition・GPU/CPU/mem/time・time-limit警告用のsignal margin・array/seqの切り替えは
すべて`run_slurm.sh`自身に書かれています。詳細は2-2節「実験を作成する」を参照してください。

### 3-2. ジョブ実行の流れ（`scripts/slurm_entry.sh`）

`run_slurm.sh`は自分の末尾で`source scripts/slurm_entry.sh`しており、実際の
処理はこのファイルが担います（`source`なので、`run_slurm.sh`側で定義した
`RUN_MODE`/`RUN_COMMAND`/`GRID_ARGS`等の変数はそのまま見える）。投入された
ジョブの内部では以下が自動で行われます。

1. スケジューラ判定（Slurm/PBS/local）と `JOB_ID`/`ARRAY_TASK_ID` 等の環境変数解決
2. `logs/{exp_name}/{job_id}/run_metadata.yaml` を `status: RUNNING` で初期化し、
   `logs/{exp_name}/latest` symlinkを更新
3. `RUN_MODE`（single/array/seq）に応じて `GRID_ARGS`/`GRID_VALUES` から `CONFIGS` を展開
4. 実行コマンドを `logs/{exp_name}/{job_id}/command.sh` に保存
5. （`USE_LOCAL_SSD_INPUT=1`の場合）`data/` をノード付属SSDの `SCRATCH_DIR/data/` へ
   rsyncし、`DATASET_DIR` をそちらに向ける
6. `apptainer exec`（`--nv` でGPU有効化）でコンテナ内、`.venv` を activate した状態で実行
   （`USE_LOCAL_SSD_OUTPUT=1`の場合、`experiment.py`は`OUTPUT_ROOT`
   （`SCRATCH_DIR/outputs`）に出力を書く。completedガードは常に
   `outputs/{exp_name}/`（`/workspace`直下・NFS）側の`completion.json`を見るため、
   scratchが空でも誤って「未実行」扱いにはならない）
7. 終了後、`sacct`（Slurm）/`qstat`（PBS）でジョブの最終状態を判定し、
   `run_metadata.yaml` の `status` を `COMPLETED`/`FAILED`/`TIMEOUT`/`OUT_OF_MEMORY`/
   `CANCELLED`/`NODE_FAIL` に更新、Slackへ通知
8. （`USE_LOCAL_SSD_OUTPUT=1`の場合）`SCRATCH_DIR/outputs/{exp_name}/` を
   `outputs/{exp_name}/`（`/workspace`直下）へrsyncで回収
9. `SCRATCH_DIR`（`/scratch/${USER}/${EXP_NAME}_${JOB_ID}`）は**自動削除しない**。
   `rm -rf`は誤削除リスクがあるため使わず、代わりに`slurm.out`へ
   「削除されていない・手動で消してください」という警告を出すだけに留める

シグナルハンドラ（`TERM`/`INT`/`USR1`）が設定されており、キャンセルやタイムリミット接近も
検知して状態を記録・通知します（`TERM`/`INT`受信時も7〜9のステップは実行されるため、
出力の回収は中断時にも行われます）。scratchの手動クリーンアップが必要な場合は
`slurm.out`の警告を確認してください。

`USE_LOCAL_SSD_INPUT`/`USE_LOCAL_SSD_OUTPUT`は`run_slurm.sh`の「Storage」節で
デフォルト有効（`1`）になっています。`/workspace`側からリアルタイムに出力を監視したい
等の理由でNFSへ直接読み書きしたい場合のみ、個別に`0`にしてください。

### 3-3. ログの見方

```
logs/{exp_name}/{job_id}/
    command.sh           # 実際に実行されたコマンド
    run_metadata.yaml     # ステータス・パーティション・ノード・git_commit・開始時刻
    slurm.out              # 標準出力（Slurm時は標準エラーも含む）
    pbs.err                 # PBS時の標準エラー
```

確認は `command.sh`（何を実行したか）→ `run_metadata.yaml`（結果）→ `config.yml`
（設定）→ `slurm.out`（詳細ログ）の順がおすすめです。全文読み込まず、まず
`run_metadata.yaml` の `status`/`fail_reason` を見てから必要な部分だけ `slurm.out` を
確認すると早いです。

### 3-4. 実験の確認・移動

```bash
groot     # Gitリポジトリのルートへ移動
lsx       # experiments/ 配下の最新10件を "ID 名前" 形式で一覧表示
cdx       # 最新の実験ディレクトリへ移動
cdx 1     # ID指定で移動
```

### 3-5. キャンセル

```bash
cancelx <job_id>
cancelx <job_id> <reason>
```

`run_metadata.yaml` の該当エントリを `status: CANCELLED` に更新し、Slackへ通知した後、
`scancel`/`qdel`（スケジューラ自動判別）でジョブを停止します。

### 3-6. Jupyter

```bash
make jupyter p=<partition> mem=<memory>
# 例: make jupyter p=small-david01 mem=32g
```

### 3-7. `agentrun` capsule内から実行する場合の違い

**capsule内から `sbatch`/`srun`/`salloc` を直接呼ぶことはできません**
（`check_sbatch_forbidden.py` でブロック、ネストしたジョブ投入を防ぐため）。したがって
capsule内のagentは `runx` を直接実行せず、代わりに投入内容を
`artifacts/proposed_train.sbatch` に書き出す（または `artifacts/report.md` に記述する）
ことで、ユーザーに投入を委ねます。`run_slurm.sh`自体が既に完全な`#SBATCH`スクリプト
なので、実験ディレクトリの`run_slurm.sh`をそのまま`artifacts/proposed_train.sbatch`に
コピーすればよく、リソース値を自分で見積もる必要はありません。

---

## 4. 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [`README.md`](./README.md) | セットアップ手順・機能概要 |
| [`TEMPLATE_CONCEPT.md`](./TEMPLATE_CONCEPT.md) | 設計理念・命名規則・実験ごとの注意事項の運用方法 |
| [`FUNCTIONS.md`](./FUNCTIONS.md) | シェル関数・Pythonライブラリ・ツールスクリプトの一覧リファレンス |
| `slurm-agents-capsule/docs/USER_MANUAL.md` | `agentrun`（Apptainer capsule経由でのagent実行）のコマンドリファレンス |
| `slurm-agents-capsule/common/skills/git/SKILL.md` | capsule内でagentに読み込まれるgit運用ルール |
| `slurm-agents-capsule/common/skills/experiments/SKILL.md` | capsule内でagentに読み込まれる実験レイアウトルール |
