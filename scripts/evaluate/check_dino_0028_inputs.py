"""Read-only inventory for the exp 0028 evaluation; never submits or runs jobs."""

import argparse
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("dino_0028.yaml")


def file_record(path):
    path = path.resolve()
    exists = path.is_file()
    return {"path": str(path), "exists": exists,
            "bytes": path.stat().st_size if exists else None}


def inventory(config, checkpoint_root, evaluation_root):
    checkpoint_dir = Path(config["checkpoint_dir"])
    if checkpoint_dir.is_absolute() or ".." in checkpoint_dir.parts:
        raise ValueError("checkpoint_dir must be relative to checkpoint_root")
    epochs = config["epochs"]
    if not epochs or any(type(epoch) is not int or epoch <= 0 for epoch in epochs):
        raise ValueError("epochs must contain positive integers")
    if len(set(epochs)) != len(epochs):
        raise ValueError("epochs must be unique")
    checkpoints = [
        {"epoch": epoch, **file_record(
            checkpoint_root / checkpoint_dir / f"model_ep{epoch}.pt")}
        for epoch in epochs
    ]
    # These paths are recorded in PROJECT_STATUS.md; their schemas and
    # evaluation behavior must still be inspected once the repo is located.
    required_paths = (
        "data/manifest.csv", "scripts/extract_features.py",
        "scripts/eval_dose_response_patch.py", "scripts/eval_by_finding.py",
    )
    assets = ([file_record(evaluation_root / path) for path in required_paths]
              if evaluation_root is not None else [])
    baselines = ([
        {"name": name, **file_record(
            evaluation_root / "outputs" / "features" / f"{name}.h5")}
        for name in config["baselines"]
    ] if evaluation_root is not None else [])
    missing = [item["path"] for item in checkpoints + assets
               if not item["exists"] or item["bytes"] == 0]
    if evaluation_root is None:
        missing.append("evaluation_root (toxpatho-uni location is not yet known)")
    return {
        "experiment": config["experiment"],
        "checkpoint_root": str(checkpoint_root.resolve()),
        "evaluation_root": (str(evaluation_root.resolve())
                            if evaluation_root is not None else None),
        "checkpoints": checkpoints,
        "evaluation_assets": assets,
        "baseline_feature_candidates": baselines,
        "missing_required_inputs": missing,
        "files_present": not missing,
        "evaluation_run": False,
        "note": ("File presence only: feature/label alignment, splits, normal "
                 "reference fitting, baseline compatibility and environment "
                 "have not been validated. Missing baseline features may be "
                 "extracted after model weights and data are located."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT,
                        help="Repository containing outputs (explicitly use the original repo in a worktree)")
    parser.add_argument("--evaluation-root", type=Path,
                        help="Location of toxpatho-uni; omission reports it as missing")
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        report = inventory(config, args.checkpoint_root, args.evaluation_root)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["files_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
