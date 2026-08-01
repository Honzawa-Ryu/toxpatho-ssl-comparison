"""paper 版（ViT-B/16 5手法）の埋め込み抽出

`scripts/analysis/run_analysis_paper_slurm.sh` が1本で回していた解析を、テンプレート方式（1実験 = 1フェーズ）に
載せ替えたもの（REFACTOR_PLAN.md §7-6 / Phase 4）。

実処理は lib/analysis/ にあり、ここは config を組み立てて呼ぶだけ。
"""

import os
import sys
from pathlib import Path

import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.analysis.embeddings import run
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)

    config = load_config(Path(__file__).parent)
    params = config.get("analysis", {})

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    run(
        methods_config=str(project_root / params["methods_config"]),
        data_dir=str(dataset_dir / params.get("data_subdir", "shards")),
        n_patches=params.get("n_patches", 2000),
        batch_size=params.get("batch_size", 256),
        output_dir=str(run_dir),
    )

    complete_run(run_dir)


if __name__ == "__main__":
    main()
