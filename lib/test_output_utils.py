import json

import pytest

from lib.output_utils import (
    complete_run,
    get_run_dir,
    sanitize_variant_key,
    write_run_metadata,
)


class TestSanitizeVariantKey:
    def test_no_separator_is_unchanged(self):
        assert sanitize_variant_key("baseline") == "baseline"

    def test_slash_keeps_last_segment(self, capsys):
        assert sanitize_variant_key("google/gemma-4-31b-it") == "gemma-4-31b-it"
        assert "path separator" in capsys.readouterr().out

    def test_backslash_keeps_last_segment(self):
        assert sanitize_variant_key("org\\model-name") == "model-name"


class TestGetRunDir:
    def test_creates_run_dir_under_project_root_when_no_output_root(self, tmp_path):
        script_path = tmp_path / "experiments" / "0001_exp" / "experiment.py"
        run_dir = get_run_dir(tmp_path, script_path, "baseline")

        assert run_dir == tmp_path / "outputs" / "0001_exp" / "baseline"
        assert run_dir.is_dir()

    def test_creates_run_dir_under_output_root_when_given(self, tmp_path):
        script_path = tmp_path / "experiments" / "0001_exp" / "experiment.py"
        output_root = tmp_path / "scratch"

        run_dir = get_run_dir(
            tmp_path, script_path, "baseline", output_root=output_root
        )

        assert run_dir == output_root / "0001_exp" / "baseline"
        assert run_dir.is_dir()

    def test_guard_checks_project_root_even_with_output_root(self, tmp_path):
        # ある実験が project_root/outputs/ 上で既に completed になっている場合、
        # output_root がまっさらな scratch を指していてもガードは効く必要がある
        # （そうでないと空のscratchが「未実行」と誤認され二重実行される）。
        script_path = tmp_path / "experiments" / "0001_exp" / "experiment.py"
        canonical_dir = tmp_path / "outputs" / "0001_exp" / "baseline"
        canonical_dir.mkdir(parents=True)
        (canonical_dir / "completion.json").write_text(
            json.dumps({"status": "completed"})
        )

        output_root = tmp_path / "scratch"

        with pytest.raises(SystemExit) as exc_info:
            get_run_dir(tmp_path, script_path, "baseline", output_root=output_root)

        assert exc_info.value.code == 0

    def test_running_status_does_not_exit(self, tmp_path, capsys):
        script_path = tmp_path / "experiments" / "0001_exp" / "experiment.py"
        canonical_dir = tmp_path / "outputs" / "0001_exp" / "baseline"
        canonical_dir.mkdir(parents=True)
        (canonical_dir / "completion.json").write_text(
            json.dumps({"status": "running"})
        )

        run_dir = get_run_dir(tmp_path, script_path, "baseline")

        assert run_dir == canonical_dir
        assert "Resuming previously failed/interrupted run" in capsys.readouterr().out

    def test_sanitizes_variant_key_with_separator(self, tmp_path):
        script_path = tmp_path / "experiments" / "0001_exp" / "experiment.py"

        run_dir = get_run_dir(tmp_path, script_path, "google/gemma-4-31b-it")

        assert run_dir == tmp_path / "outputs" / "0001_exp" / "gemma-4-31b-it"
        assert (run_dir / ".variant_key_original.txt").read_text() == "google/gemma-4-31b-it\n"


class TestRunMetadataLifecycle:
    def test_write_then_complete(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        write_run_metadata(run_dir, note="hello")
        meta = json.loads((run_dir / "completion.json").read_text())
        assert meta["status"] == "running"
        assert meta["note"] == "hello"

        complete_run(run_dir)
        meta = json.loads((run_dir / "completion.json").read_text())
        assert meta["status"] == "completed"
        assert "completed_at" in meta
