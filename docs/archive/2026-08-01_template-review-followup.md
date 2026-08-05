> **2026-08-01 追記:** 本ファイル作成後、以下「未着手 ❌」「部分的に完了 ⚠️」節に
> 記載の項目は全て解消済み（`exp_common.sh`への完全移行、`tools/cancel_job.sh`の
> 残課題、`Makefile`の`clean_failed`/`log_clean`/`test`/`shellcheck`ターゲット、
> `tools/first_setup.sh`の冪等化、`.gitignore`#16、`pyproject.toml`の
> description/optional-dependencies整理、`.github/workflows/ci.yml`、
> ドキュメント乖離6件）。このコミット時点で report.md 指摘16件は全件対応済みとみなす。
> なお本リポジトリは wsi-ad からの移行先(`REFACTOR_PLAN.md` フェーズB)としても
> 使われており、このインフラ整理はその移行の前提作業（wsi-ad版をベースに
> toxpatho独自改修を当て直す作業）を兼ねている。

# 残作業まとめ（report.md 対応状況）

元レビュー: `.agent-runs/20260730_165705_review_claude_reviewed-hpc-experiment-management-template/artifacts/report.md`

現在のブランチ `260727_dev` の作業ツリー（未コミット）を、report.md の指摘16件 + ドキュメント乖離 + 構造面の指摘と突き合わせた結果。
`git diff --stat` で変更が入っているのは `.bashrc` / `scripts/notify_slack.sh` / `scripts/slurm_entry.sh` /
`templates/experiment.py` / `tools/create_exp.sh` の5ファイルと、新規 `scripts/exp_common.sh` / `tests/smoke_slurm_entry.sh` のみ。
それ以外のファイル（`Makefile` / `tools/cancel_job.sh` / `tools/first_setup.sh` / `.bashrc.d/*.sh` / `tools/resume_exp.sh` /
`tools/mark_failed.sh` / `tools/rename_exp.sh` / `pyproject.toml` / `README.md` / `TEMPLATE_CONCEPT.md` / `USAGE.md` /
`FUNCTIONS.md` / `templates/run_slurm.sh` / `tools/start_jupyter.sh` / `.gitignore`）は report.md 時点から無変更。

## 完了 ✅

- **#1 `.bashrc` の `exit 2`**（`.bashrc:11`）— 対話シェルのみ警告する形に修正済み。`_is_interactive_shell` を使用。
- **#2 `create_exp.sh` の partition owner テーブル**（`scripts/exp_common.sh` の `exp_resolve_owner`）—
  `filesrv02` 追加済み。加えて `EXP_PARTITION_OWNER` 環境変数での上書きにも対応。
  バリデーション（partition解決）を `mkdir` より前に移動済み（`tools/create_exp.sh`）。
- **#3(a) PBS の `walltime` grep 失敗**（`scripts/slurm_entry.sh` の `_read_job_time_limit`）—
  `#PBS -l walltime=` へのフォールバックと `|| true` で対応済み。
- **#3(b) `notify_slack.sh` の `SLURM_*` 直参照**（`scripts/notify_slack.sh`）—
  `JOB_ID`/`JOB_PARTITION`/`ARRAY_TASK_ID` 参照に統一済み、全変数に既定値あり。
- **#3(c) `SCHEDULER=local` の `_get_job_state`**（`scripts/slurm_entry.sh:_get_job_state`）—
  local分岐を追加、`UNKNOWN` も `_handle_final_state` で正常系として扱うよう修正済み。
- **#5 `--config` 不整合**（`templates/experiment.py`）—
  `parse_args()` に `--config`（既定 `config.yml`）追加、`load_config()` が引数を実際に使うよう修正済み。
- **#8（slurm_entry.sh 側のみ）Python文字列展開**（`scripts/slurm_entry.sh:_yaml_put`）—
  `run_metadata.yaml` の読み書きから Python/PyYAML を排除し、`awk`/`sed` ベースの `_yaml_put` に置換済み。
- **#7（slurm_entry.sh 側のみ）システムPython依存**— 上記により `slurm_entry.sh` は PyYAML 不要になった。
- **#12 `SIF_PATH` 未定義**（`scripts/slurm_entry.sh` 冒頭）—
  `: "${SIF_PATH:?...}"` で早期に分かるメッセージで落とすよう修正済み（`PROJECT_ROOT`/`EXP_NAME` も同様）。
- **ドキュメント乖離: ⚡INTERRUPTED の誤用**（`scripts/notify_slack.sh:notify_time_limit_warning`）—
  専用の ⏳ `TIME_LIMIT_WARNING` 通知を新設し、`on_time_limit_warning` から呼ぶよう修正済み。
- **テスト土台**: `tests/smoke_slurm_entry.sh` を新規作成。
  SCHEDULER=local／PBS形式ヘッダー／時間制限ヘッダー無し／非ゼロ終了／SIF_PATH未設定、の5ケースを
  apptainerスタブで検証する内容で、report.mdの「#14 テスト皆無」への対応の第一歩。
  **ただしこのセッション内では実行して確認できていない**（capsule の hook が `bash <script>` の直接実行を
  ブロックするため — `sbatch`/`runx` 経由 or ユーザー側での実行が必要）。

## 部分的に完了（要注意）⚠️

- **#11 実験ディレクトリ解決ロジックの共通化** — `scripts/exp_common.sh` は新規作成され、
  `exp_list_names`/`exp_latest_name`/`exp_next_id`/`exp_resolve_dir` 等の共通関数が揃っている。
  `tools/create_exp.sh` はこれを使うよう移行済み。
  **しかし report.md が指摘した本来のバグ（`cdx`/`lsx` が `latest` symlink を実験として拾う）は
  `.bashrc.d/1-experiments.sh:39,71-72` の `sort -r | head -n1` のまま未修正**。
  `tools/resume_exp.sh` / `tools/mark_failed.sh` / `tools/rename_exp.sh` も独自の
  `_resolve_exp`/`_latest_exp` を保持したままで `exp_common.sh` に移行していない
  （4ファイルに旧実装が残存 = 依然として5通りの実装が3通りに減っただけ）。
  → **`exp_common.sh` を作った意味を回収するには、この4ファイルの移行が必須**。
- **`scripts/exp_common.sh:103`** のコメントが `tests/test_exp_common.sh` を参照しているが、
  そのファイルは存在しない（`exp_resolve_owner`/`exp_resolve_partition`/`exp_resolve_signal_margin` の
  純粋関数群に対するユニットテストが書かれる予定だったが未着手）。
- **#7/#8（cancel_job.sh 側）** — `tools/cancel_job.sh:48-50` は
  `data["cancel_reason"] = "${REASON}"` のままで、Python文字列展開の問題は未修正
  （slurm_entry.sh側だけ直っている）。`pyproject.toml` にも `pyyaml` は依然未宣言。

## 未着手 ❌

- **#4 `make clean_failed` が RUNNING ジョブのログを消す**（`Makefile:132-137`）—
  無変更。8日ジョブの mtime 30h超え問題はそのまま。
- **#6 `cancelx` の job_id 前方一致**（`tools/cancel_job.sh:24-28`）—
  無変更。`set -euo pipefail` も未追加（他5本のtoolsと不整合なまま）。通知順序（`scancel`前に通知/更新）も未修正。
- **#9 `make log_clean` の array ジョブログ潰れ**（`Makefile:100-107` の `cut -d_ -f1`）— 無変更。
- **#10 `make setup` の非冪等性**（`tools/first_setup.sh:4-9`）—
  `mkdir` に `-p` 無し、`set -e` 無しのまま。`libraries/`/`notebooks/` の README・.gitignore不整合も未対応。
- **#13 Jupyter token が固定ファイル名**（`tools/start_jupyter.sh:4` の `--output=jupyter-server.out`）— 無変更。
- **#14（残り）** shellcheck の CI 導入、`lib/output_utils.py`/`scripts/preflight_check.py` への pytest — 未着手。
  `Makefile` にもテスト実行ターゲットは無い。
- **#15 `pyproject.toml`** — `description` の不一致（Knowledge graph/PubMed の残骸）、
  重い依存45個の `optional-dependencies` 分割、`pyyaml` 明示、いずれも無変更。
- **#16 `experiments/*/uncommitted_changes.diff` が追跡対象**（`.gitignore`）— 無変更。
- **ドキュメント乖離の一括修正**（report.md の表、全6行）—
  `templates/run_slurm.sh:36-37`（scratch自動削除の矛盾。USAGE.mdと矛盾したまま）、
  `TEMPLATE_CONCEPT.md`（preflightのシード検出/runxの機能過大記載）、
  `FUNCTIONS.md`（`get_run_dir` の第4引数 `output_root` 未記載）、
  `README.md:120-124`（`--cpus-per-task`等の数値不一致）、
  `.bashrc.d/2-git.sh`（テンプレートに含めない旨の記述と実態の齟齬、`core.hooksPath`のグローバル上書き副作用）
  — いずれも未対応。

## 推奨対応順（残りぶんのみ）

report.md の「推奨対応順」10項目のうち 1〜4 は完了、5〜10 が対象。優先度はレビュー時のまま踏襲するのが妥当:

1. **`clean_failed` の RUNNING 削除条件**（#4）— 稼働中ジョブのログが消える実害が最も大きい。
2. **`cancel_job.sh` の前方一致修正 + `set -euo pipefail` + 通知順序**（#6）。
3. **`cancel_job.sh` 側の Python文字列展開除去 + `pyproject.toml` への `pyyaml` 明示**（#7/#8 残り）。
4. **`exp_common.sh` への移行を完遂**（#11 根本対応）— `1-experiments.sh` の `cdx`/`lsx`、
   `resume_exp.sh`/`mark_failed.sh`/`rename_exp.sh` の `_resolve_exp`/`_latest_exp` を置き換え、
   `latest` symlink 誤選択のバグを実際に消す。あわせて `tests/test_exp_common.sh` を書く
   （コメントで予告されたまま存在しない）。
5. **`tests/smoke_slurm_entry.sh` の実行確認** — このセッションでは実行がブロックされたため未検証。
   ユーザー側 or 許可された経路で一度通しておく。
6. **shellcheck を CI に、`output_utils`/`preflight_check` に pytest**（#14 残り）。
7. **ドキュメント乖離の一括修正**（特に `templates/run_slurm.sh:36-37` の scratch 自動削除の矛盾は
   ユーザーが誤って手動削除しないための誤解防止という意味で優先度が高い）。
8. **`make setup` の冪等化**（#10）、**`log_clean` の array ログ潰れ**（#9）、
   **Jupyter token のログファイル移動**（#13）、**`uncommitted_changes.diff` の `.gitignore` 追加**（#16）、
   **`pyproject.toml` の description/optional-dependencies 整理**（#15）— いずれも Medium/Low、後回し可。
