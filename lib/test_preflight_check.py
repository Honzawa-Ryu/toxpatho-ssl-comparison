import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "preflight_check.py"
_spec = importlib.util.spec_from_file_location("preflight_check", _MODULE_PATH)
preflight_check = importlib.util.module_from_spec(_spec)
sys.modules["preflight_check"] = preflight_check
_spec.loader.exec_module(preflight_check)


def _make_exp(tmp_path, config="", experiment="", run_slurm=None):
    exp_path = tmp_path / "experiments" / "0001_exp"
    exp_path.mkdir(parents=True)
    (exp_path / "config.yml").write_text(config)
    (exp_path / "experiment.py").write_text(experiment)
    if run_slurm is not None:
        (exp_path / "run_slurm.sh").write_text(run_slurm)
    return exp_path


class TestCheckConfigExperimentMatch:
    def test_passes_when_no_config_or_experiment_file(self, tmp_path):
        exp_path = tmp_path / "experiments" / "0001_exp"
        exp_path.mkdir(parents=True)
        assert preflight_check.check_config_experiment_match(exp_path) is True

    def test_passes_when_all_referenced_keys_defined(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            config="lr: 0.001\nbatch_size: 32\n",
            experiment="lr = config['lr']\nbs = config.get('batch_size')\n",
        )
        assert preflight_check.check_config_experiment_match(exp_path) is True

    def test_fails_when_referenced_key_is_undefined(self, tmp_path, capsys):
        exp_path = _make_exp(
            tmp_path,
            config="lr: 0.001\n",
            experiment="epochs = config['epochs']\n",
        )
        assert preflight_check.check_config_experiment_match(exp_path) is False
        assert "epochs" in capsys.readouterr().out


class TestCheckGridSize:
    def test_passes_when_no_run_slurm(self, tmp_path):
        exp_path = tmp_path / "experiments" / "0001_exp"
        exp_path.mkdir(parents=True)
        assert preflight_check.check_grid_size(exp_path) is True

    def test_fails_when_grid_args_present_but_run_mode_single(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            run_slurm=(
                'RUN_MODE="single"\n'
                'GRID_VALUES=("a b")\n'
                'GRID_ARGS=(--lr)\n'
            ),
        )
        assert preflight_check.check_grid_size(exp_path) is False

    def test_fails_when_array_count_mismatches_grid_combinations(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            run_slurm=(
                'RUN_MODE="array"\n'
                'GRID_VALUES=("a b c")\n'
                "#SBATCH --array=0-1\n"
            ),
        )
        assert preflight_check.check_grid_size(exp_path) is False

    def test_passes_when_array_count_matches_grid_combinations(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            run_slurm=(
                'RUN_MODE="array"\n'
                'GRID_VALUES=("a b c")\n'
                "#SBATCH --array=0-2\n"
            ),
        )
        assert preflight_check.check_grid_size(exp_path) is True

    def test_ignores_commented_out_sbatch_directive(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            run_slurm=(
                'RUN_MODE="array"\n'
                'GRID_VALUES=("a b")\n'
                "# #SBATCH --array=0-99\n"
                "#SBATCH --array=0-1\n"
            ),
        )
        assert preflight_check.check_grid_size(exp_path) is True


class TestCheckDirectWrites:
    def test_passes_when_no_experiment_file(self, tmp_path):
        exp_path = tmp_path / "experiments" / "0001_exp"
        exp_path.mkdir(parents=True)
        assert preflight_check.check_direct_writes(exp_path) is True

    def test_fails_on_hardcoded_outputs_write(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            experiment="df.to_csv('outputs/result.csv')\n",
        )
        assert preflight_check.check_direct_writes(exp_path) is False

    def test_passes_when_using_run_dir_helper(self, tmp_path):
        exp_path = _make_exp(
            tmp_path,
            experiment="df.to_csv(run_dir / 'result.csv')\n",
        )
        assert preflight_check.check_direct_writes(exp_path) is True
