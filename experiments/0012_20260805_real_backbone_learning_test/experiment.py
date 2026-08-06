import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np

# --- Deep learning ---
import torch
import torch.nn as nn


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(config_path: Path) -> dict:
    """Load a config YAML file.

    The path comes from --config, which run_slurm.sh passes as a path relative
    to the experiment directory (so `--config config.yml` means the config.yml
    sitting next to this script). Resolving it here rather than hardcoding
    exp_dir/config.yml is what makes it possible to feed the same
    experiment.py a different config.
    """
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    # Config file, resolved relative to this experiment directory.
    # run_slurm.sh's RUN_COMMAND passes `--config config.yml` by default.
    parser.add_argument(
        "--config",
        default="config.yml",
        help="config YAML (experiment ディレクトリからの相対パス、または絶対パス)",
    )
    return parser.parse_args()


def build_synthetic_patch_memmap(memmap_path: Path, n_patches: int, patch_size: int, seed: int) -> None:
    """合成ランダムパッチをmemmapに書く(data/ssl_patchesがまだ無いための代用データ)。

    lib/trainer/data.py の MemmapPatchDataset がそのまま読める形式
    (shape=(N, patch_size, patch_size, 3), dtype=uint8)で書く。
    """
    rng = np.random.default_rng(seed)
    mm = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(n_patches, patch_size, patch_size, 3))
    # 1枚ずつ書く(全部を一度にrandintすると N x 224 x 224 x 3 のint配列がメモリに乗る)
    for i in range(n_patches):
        mm[i] = rng.integers(0, 256, size=(patch_size, patch_size, 3), dtype=np.uint8)
    mm.flush()


def build_backbone(model_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """lib/trainer/model.py の prepare_model と同じ分岐でバックボーンを作る。

    ViT/DenseNet121以外(既定のResNet18を含む)は末尾の分類headを外した
    nn.Sequentialを返す。head_sizeはlib/model/zoo.py DICT_MODELの宣言値。
    """
    import lib.model.zoo as zoo

    if model_name not in zoo.DICT_MODEL:
        raise ValueError(f"unknown model_name: {model_name} (have {list(zoo.DICT_MODEL)})")
    model_cls, head_size = zoo.DICT_MODEL[model_name]
    weights = "DEFAULT" if pretrained else None
    encoder = model_cls(weights=weights)

    if "ViT" in model_name:
        encoder.heads = nn.Identity()
        backbone = encoder
    elif model_name == "DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    else:
        backbone = nn.Sequential(*list(encoder.children())[:-1])
    return backbone, head_size


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.sslmodel.sslutils import BarlowTwins as BarlowTwinsSSL
    from lib.sslmodel.utils import fix_seed
    from lib.trainer.data import MemmapPatchDataset
    from torch.utils.data import DataLoader

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = "default"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)
    seed: int = config.get("seed", 42)
    tt: dict = config.get("train_test", {})

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    fix_seed(seed=seed, fix_gpu=True)

    # ── Synthetic patch data (data/ssl_patches doesn't exist yet - see config.yml) ──
    patch_size = int(tt.get("patch_size", 224))
    n_patches = int(tt.get("n_synthetic_patches", 512))
    memmap_path = run_dir / "synthetic_patches.memmap"
    build_synthetic_patch_memmap(memmap_path, n_patches, patch_size, seed)
    logger.info(f"synthetic patches: n={n_patches} patch_size={patch_size} -> {memmap_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; falling back to CPU (run_slurm.sh requests --gres=gpu:1)")

    # ── Real SSL transform + real backbone + real BarlowTwins model/loss ──────
    ssl = BarlowTwinsSSL(DEVICE=device)
    transform = ssl.prepare_transform(
        color_plob=float(tt.get("color_plob", 0.8)),
        blur_plob=float(tt.get("blur_plob", 0.4)),
        solar_plob=float(tt.get("solar_plob", 0.0)),
    )

    model_name = tt.get("model_name", "ResNet18")
    pretrained = bool(tt.get("pretrained", False))
    backbone, head_size = build_backbone(model_name, pretrained)
    logger.info(f"backbone:    {model_name} (pretrained={pretrained}, head_size={head_size})")

    model, criterion = ssl.prepare_model(backbone, head_size=head_size)

    dataset = MemmapPatchDataset(memmap_path, patch_size, rows=np.arange(n_patches), transform=transform)
    batch_size = int(tt.get("batch_size", 32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    lr = float(tt.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = int(tt.get("num_epochs", 20))

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    epoch_losses: list[float] = []
    for epoch in range(num_epochs):
        running_loss, n_batches = 0.0, 0
        for batch in loader:
            loss = ssl.calc_loss(model, batch, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / n_batches
        epoch_losses.append(epoch_loss)
        logger.info(f"epoch {epoch + 1}/{num_epochs}  loss={epoch_loss:.4f}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    results: dict = {
        "data_source": "synthetic_random_patches",  # data/ssl_patches doesn't exist yet
        "n_patches": n_patches,
        "patch_size": patch_size,
        "model_name": model_name,
        "pretrained": pretrained,
        "head_size": head_size,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": lr,
        "epoch_losses": epoch_losses,
        "loss_decreased": epoch_losses[-1] < epoch_losses[0] if len(epoch_losses) > 1 else None,
        "status": "ok",
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
