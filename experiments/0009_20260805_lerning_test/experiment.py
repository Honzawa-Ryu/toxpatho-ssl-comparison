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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


class MemmapFeatureDataset(Dataset):
    """1スライド分のUNI特徴量memmap(features.dat, shape=(N, D) float32)を読むDataset。"""

    def __init__(self, memmap_path: Path, shape: tuple[int, int], dtype: np.dtype):
        self.memmap_path = str(memmap_path)
        self.shape = shape
        self.dtype = dtype
        self._mm: np.memmap | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.memmap_path, dtype=self.dtype, mode="r").reshape(self.shape)
        return self._mm

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = np.array(self._memmap()[idx], dtype=np.float32)
        return torch.from_numpy(row)


def augment_features(x: torch.Tensor, noise_std: float, dropout_p: float) -> torch.Tensor:
    """特徴量版のSSL拡張。

    lib/sslmodel/utils.py の ssl_transform() はPIL画像を前提にしており、この
    memmapが持つのは生パッチではなく抽出済みUNI特徴量なので使えない
    (EXP8由来の data/ssl_patches がまだ生成されていないため)。特徴量dropout +
    加法ノイズで2つの異なる view を作れば、BarlowTwinsLoss が学習ループの
    疎通確認に使うには十分。
    """
    x = F.dropout(x, p=dropout_p, training=True)
    x = x + torch.randn_like(x) * noise_std
    return x


def resolve_memmap_dir(memmap_root: Path, memmap_dir_name: str | None) -> Path:
    """train_test.memmap_dir で指定されたスライド、未指定なら最初に見つかったものを使う。"""
    if memmap_dir_name:
        memmap_dir = memmap_root / str(memmap_dir_name)
        if not memmap_dir.exists():
            raise FileNotFoundError(f"configで指定されたmemmap_dirが見つかりません: {memmap_dir}")
        return memmap_dir

    candidates = sorted(
        d for d in memmap_root.iterdir()
        if d.is_dir() and (d / "features.dat").exists() and (d / "meta_features.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"{memmap_root} に features.dat を含むディレクトリが見つかりません")
    return candidates[0]


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.sslmodel.models.barlowtwins import BarlowTwins, BarlowTwinsLoss
    from lib.sslmodel.utils import fix_seed

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

    # ── Pick a memmap to train on ───────────────────────────────────────────
    memmap_root = dataset_dir / "features_memmap_output"
    memmap_dir = resolve_memmap_dir(memmap_root, tt.get("memmap_dir"))
    logger.info(f"memmap_dir:  {memmap_dir}")

    meta = json.loads((memmap_dir / "meta_features.json").read_text())
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    logger.info(f"features.dat shape={shape} dtype={dtype}")

    # ── Data ─────────────────────────────────────────────────────────────────
    batch_size = int(tt.get("batch_size", 64))
    dataset = MemmapFeatureDataset(memmap_dir / "features.dat", shape, dtype)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    # ── Model (tiny MLP backbone + real BarlowTwins projector/loss) ───────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; falling back to CPU (run_slurm.sh requests --gres=gpu:1)")

    feat_dim = shape[1]
    hidden_dim = int(tt.get("hidden_dim", 512))
    proj_dim = int(tt.get("proj_dim", 128))

    backbone = nn.Sequential(
        nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
    )
    model = BarlowTwins(backbone, head_size=[hidden_dim, hidden_dim, proj_dim]).to(device)
    criterion = BarlowTwinsLoss()
    lr = float(tt.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    noise_std = float(tt.get("noise_std", 0.1))
    dropout_p = float(tt.get("feature_dropout_p", 0.2))
    num_epochs = int(tt.get("num_epochs", 3))

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    epoch_losses: list[float] = []
    for epoch in range(num_epochs):
        running_loss, n_batches = 0.0, 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            z1 = model(augment_features(x, noise_std, dropout_p))
            z2 = model(augment_features(x, noise_std, dropout_p))
            loss = criterion(z1, z2)

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
        "memmap_dir": str(memmap_dir),
        "n_patches": shape[0],
        "feat_dim": feat_dim,
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
