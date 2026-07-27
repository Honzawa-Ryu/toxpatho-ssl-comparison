import sys
import os
import re
from pathlib import Path

def get_latest_experiment() -> Path:
    exp_dir = Path("experiments")
    subdirs = [d for d in exp_dir.iterdir() if d.is_dir() and re.match(r"^\d{4}_", d.name)]
    if not subdirs:
        print("❌ Error: No experiments found in experiments/")
        sys.exit(1)
    return sorted(subdirs)[-1]

def check_config_experiment_match(exp_path: Path):
    config_file = exp_path / "config.yml"
    exp_file = exp_path / "experiment.py"
    
    if not config_file.exists() or not exp_file.exists():
        return True
    
    config_content = config_file.read_text(encoding="utf-8")
    exp_content = exp_file.read_text(encoding="utf-8")
    
    # Simple YAML key parser (top level keys)
    config_keys = set(re.findall(r"^([a-zA-Z_][a-zA-Z0-9_]*):", config_content, re.MULTILINE))
    
    # Find config references in Python
    exp_refs = set(re.findall(r"config(?:(?:\.get\(['\"]|\[['\"])([a-zA-Z_][a-zA-Z0-9_]*)['\"]", exp_content))

    # 1. Undefined references
    undefined = exp_refs - config_keys
    if undefined:
        print(f"🔴 Blocker: Keys referenced in experiment.py but missing in config.yml: {undefined}")
        return False

    return True

def _active_lines(content: str) -> str:
    """Drop full-line comments so bash-variable checks ignore commented-out example blocks."""
    return "\n".join(
        line for line in content.split("\n") if not line.strip().startswith("#")
    )

def check_grid_size(exp_path: Path):
    run_slurm = exp_path / "run_slurm.sh"
    if not run_slurm.exists():
        return True

    content = run_slurm.read_text(encoding="utf-8")
    active = _active_lines(content)

    # Check for GRID_VALUES (only in active, uncommented lines)
    grid_matches = re.findall(r"GRID_VALUES=\((.*?)\)", active, re.DOTALL)
    grid_args_present = bool(re.search(r"GRID_ARGS\s*=\s*\(", active))

    total_tasks = None
    if grid_matches:
        values_str = grid_matches[0]
        lines = [line.strip() for line in values_str.split("\n") if line.strip()]
        total_tasks = 1
        for line in lines:
            clean_line = re.sub(r"['\"]", "", line)
            elements = clean_line.split()
            if elements:
                total_tasks *= len(elements)
        if total_tasks > 50:
            print(f"🔴 Blocker: GRID total task count ({total_tasks}) exceeds safety limit of 50.")
            return False
        elif total_tasks > 20:
            print(f"🟡 Warning: GRID task count is {total_tasks}. Ensure resources are optimized.")

    # Use the last assignment (bash "last one wins" if RUN_MODE is set more than once).
    run_mode_matches = re.findall(r'RUN_MODE\s*=\s*["\']?(\w+)["\']?', active)
    run_mode = run_mode_matches[-1] if run_mode_matches else "single"

    # #SBATCH directives are comments to bash, so this must be checked against the
    # raw content (not `active`), but anchored to line-start so a double-commented
    # example ("# #SBATCH --array=...") isn't mistaken for an active directive.
    array_match = re.search(r"^\s*#SBATCH\s+--array=\d+-(\d+)", content, re.MULTILINE)

    if grid_args_present and run_mode not in ("array", "seq"):
        print("🔴 Blocker: GRID_ARGS is defined but RUN_MODE is not \"array\" or \"seq\" "
              "— it would silently run as a single config. Set RUN_MODE and, for array "
              "mode, an active #SBATCH --array=0-N line.")
        return False

    if run_mode == "array":
        if not array_match:
            print("🔴 Blocker: RUN_MODE=\"array\" but no active #SBATCH --array=0-N line found.")
            return False
        if total_tasks is not None:
            declared_max = int(array_match.group(1))
            if declared_max != total_tasks - 1:
                print(
                    f"🔴 Blocker: #SBATCH --array=0-{declared_max} does not match the "
                    f"GRID combination count ({total_tasks}; expected 0-{total_tasks - 1})."
                )
                return False

    return True

def check_direct_writes(exp_path: Path):
    exp_file = exp_path / "experiment.py"
    if not exp_file.exists():
        return True
    
    content = exp_file.read_text(encoding="utf-8")
    matches = re.findall(r"(\b(?:to_parquet|to_csv|open)\(\s*['\"](?:./)?(?:outputs|data)/.*['\"])", content)
    if matches:
        print("🔴 Blocker: Hardcoded direct write to outputs/ or data/ detected in experiment.py:")
        for m in matches:
            print(f"   {m}")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path", nargs="?", default=None,
                         help="対象実験ディレクトリ（省略時は最新実験）")
    args = parser.parse_args()

    if args.exp_path:
        exp_path = Path(args.exp_path)
        if not exp_path.exists():
            print(f"❌ Error: experiment path not found: {exp_path}")
            sys.exit(1)
    else:
        exp_path = get_latest_experiment()

    print(f"🔬 Preflight checking experiment: {exp_path.name}")

    ok = True
    ok &= check_config_experiment_match(exp_path)
    ok &= check_grid_size(exp_path)
    ok &= check_direct_writes(exp_path)

    if ok:
        print("✅ preflight OK — runx 投入可")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
