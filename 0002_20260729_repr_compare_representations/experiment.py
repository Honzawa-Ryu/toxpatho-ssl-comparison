"""表現比較（CKA / ARI / プロトタイプ / t-SNE）

`scripts/analysis/run_analysis_slurm.sh` が1本で回していた解析を、テンプレート方式（1実験 = 1フェーズ）に
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


def _resolve_input_dir(project_root: Path, params: dict, variant_key: str) -> Path:
    """上流フェーズ（extract_embeddings）の出力ディレクトリを解決する。

    テンプレートの出力レイアウトは outputs/{exp}/{variant_key}/ なので、
    上流の実験名さえ分かれば一意に決まる（REFACTOR_PLAN.md §7-3）。
    上流がまだ回っていなければここで止める（黙って空の結果を出さない）。
    """
    input_dir = project_root / "outputs" / params["input_exp"] / variant_key

    if not input_dir.is_dir():
        print(f"Error: input experiment output not found: {input_dir}", file=sys.stderr)
        print("       先に上流フェーズ（0001_20260729_repr_extract_embeddings）を回してください。", file=sys.stderr)
        sys.exit(1)

    return input_dir


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.analysis.compare import run
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)

    config = load_config(Path(__file__).parent)
    params = config.get("analysis", {})

    input_dir = _resolve_input_dir(project_root, params, variant_key)

    write_run_metadata(
        run_dir, exp_name=exp_name, variant_key=variant_key,
        input_exp=params["input_exp"],
    )

    # 上流の成果物（emb_*.npy / patches_*.{npy,json}）をこの実験の出力先へ複製し、
    # 以降の図表もすべて run_dir 配下に閉じるようにする。
    # lib/analysis の各 run() は「入力を読んだ場所に出力を書く」設計のため。
    _stage_inputs(input_dir, run_dir)

    run(input_dir=str(run_dir), k=params.get("k", 10), rbf=params.get("rbf", True))

    complete_run(run_dir)


def _stage_inputs(input_dir: Path, run_dir: Path) -> None:
    import shutil

    for item in sorted(input_dir.iterdir()):
        if item.is_file() and (
            item.name.startswith("emb_")
            or item.name.startswith("patches_")
            or item.name == "extract_summary.json"
        ):
            dst = run_dir / item.name
            if not dst.exists():
                shutil.copy2(item, dst)


if __name__ == "__main__":
    main()
