import os
import sys
import json
from pathlib import Path
from datetime import datetime

def sanitize_variant_key(variant_key: str) -> str:
    """Make a variant_key safe to use as a single directory name.

    If the key contains path separators (e.g. a raw HuggingFace model name
    like "google/gemma-4-31b-it"), only the part after the last separator is
    kept. This keeps outputs/{exp_name}/{variant_key}/ a single directory
    level, which the outputs/*/*/completion.json glob pattern (used by
    review-exp and roadmap) depends on.

    Args:
        variant_key: The raw variant key, possibly containing "/" or "\\".

    Returns:
        The sanitized variant key (unchanged if it had no separators).
    """
    if "/" not in variant_key and "\\" not in variant_key:
        return variant_key
    normalized = variant_key.replace("\\", "/")
    sanitized = normalized.rsplit("/", 1)[-1]
    print(
        f"Warning: variant_key {variant_key!r} contains a path separator; "
        f"using {sanitized!r} instead (last segment after '/')."
    )
    return sanitized


def get_run_dir(
    project_root: str | Path,
    script_path: str | Path,
    variant_key: str,
    output_root: str | Path | None = None,
) -> Path:
    """Resolve (and create) the run directory for a variant.

    The completed-guard always checks completion.json under
    project_root/outputs/, regardless of output_root. This keeps the guard
    meaningful even when output_root points to a node-local scratch
    directory that starts empty on every job (e.g. run results are staged
    on scratch during the run and rsynced back to project_root/outputs/
    only at job end by scripts/slurm_entry.sh) — a stale/empty scratch dir
    must never be mistaken for "not yet run".

    Args:
        project_root: Repository root; always used for the guard check.
        script_path: Path to the calling experiment.py (its parent dirname
            is used as exp_name).
        variant_key: Identifies this run among others in the same experiment.
        output_root: If set, the actual run directory is created under
            output_root/exp_name/variant_key instead of
            project_root/outputs/exp_name/variant_key. Use this to stage
            writes on fast local scratch while the guard still checks the
            canonical project_root location.

    Returns:
        The run directory to write outputs into.
    """
    original_variant_key = variant_key
    variant_key = sanitize_variant_key(variant_key)

    project_root = Path(project_root)
    script_path = Path(script_path)

    # Derive exp_name from the parent directory of script_path
    exp_name = script_path.parent.name

    canonical_dir = project_root / "outputs" / exp_name / variant_key
    run_dir = Path(output_root) / exp_name / variant_key if output_root is not None else canonical_dir

    completion_file = canonical_dir / "completion.json"
    if completion_file.exists():
        try:
            meta = json.loads(completion_file.read_text())
            status = meta.get("status")
            if status == "completed":
                print(f"Error: Run is already completed: {canonical_dir}")
                meta["guard_skipped_at"] = datetime.now().isoformat()
                completion_file.write_text(json.dumps(meta, indent=4))
                sys.exit(0)
            elif status == "running":
                print(f"Warning: Resuming previously failed/interrupted run: {canonical_dir}")
        except Exception:
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    if original_variant_key != variant_key:
        # Keep the original key around for traceability even though the
        # directory itself is named after the sanitized key.
        (run_dir / ".variant_key_original.txt").write_text(original_variant_key + "\n")
    return run_dir

def write_run_metadata(run_dir: str | Path, **kwargs) -> None:
    run_dir = Path(run_dir)
    completion_file = run_dir / "completion.json"

    started_at = datetime.now().isoformat()
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID") or ""
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("PBS_ARRAY_INDEX") or ""

    meta = {
        "job_id": job_id,
        "array_task_id": array_task_id,
        "started_at": started_at,
        "status": "running"
    }
    meta.update(kwargs)

    completion_file.write_text(json.dumps(meta, indent=4))

def complete_run(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    completion_file = run_dir / "completion.json"

    if completion_file.exists():
        try:
            meta = json.loads(completion_file.read_text())
        except Exception:
            meta = {}
    else:
        meta = {}

    meta["status"] = "completed"
    meta["completed_at"] = datetime.now().isoformat()

    completion_file.write_text(json.dumps(meta, indent=4))
