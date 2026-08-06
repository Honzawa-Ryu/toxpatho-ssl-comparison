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
from torch.utils.data.distributed import DistributedSampler


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment", suffix: str = "") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment{suffix}.log.

    Each DDP rank runs this file concurrently; without a per-rank suffix, multiple
    processes would interleave writes into the same experiment.log (mirrors the
    rank-suffixed log filename lib/trainer/context.py uses for non-main ranks).
    """
    logger = logging.getLogger(f"{name}{suffix}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / f"experiment{suffix}.log")
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
    """1スライド分のUNI特徴量memmap(features.dat, shape=(N, D) float32)を読むDataset。

    全rankが同じファイルを読む(lib/trainer/data.pyのMemmapPatchDatasetと同じ、
    行分割はDatasetではなくDistributedSamplerに任せる設計)。
    """

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
    """特徴量版のSSL拡張(EXP9と同じ)。

    lib/sslmodel/utils.py の ssl_transform() はPIL画像前提でこのmemmap(抽出済み
    UNI特徴量)には使えないため、特徴量dropout + 加法ノイズで2つの view を作る。
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

    from lib.trainer import distributed

    # torchrun経由(WORLD_SIZE>1)ならinit_process_group、単一プロセスならno-op
    # (lib/trainer/entry.py の main(ctx) と同じ構造)。
    distributed.setup()
    try:
        _run(project_root, distributed)
    finally:
        distributed.teardown()


def _run(project_root: Path, distributed) -> None:
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.sslmodel.models.barlowtwins import BarlowTwins, BarlowTwinsLoss
    from lib.sslmodel.utils import fix_seed

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = "default"

    world_size = distributed.world_size()
    rank = distributed.rank()
    local_rank = distributed.local_rank()
    is_main = distributed.is_main_process()

    # 全rankがmkdir(exist_ok=True)するだけなので競合しても安全(lib/output_utils.get_run_dir)。
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    log_suffix = "" if is_main else f"_rank{rank}"
    logger = setup_logger(run_dir, exp_name, suffix=log_suffix)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = load_config(config_path)
    seed: int = config.get("seed", 42)
    tt: dict = config.get("train_test", {})

    if is_main:
        write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, world_size=world_size)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")
    logger.info(f"rank/world_size/local_rank: {rank}/{world_size}/{local_rank}")

    fix_seed(seed=seed, fix_gpu=True)

    # ── Pick a memmap to train on (shared across ranks) ─────────────────────
    memmap_root = dataset_dir / "features_memmap_output"
    memmap_dir = resolve_memmap_dir(memmap_root, tt.get("memmap_dir"))
    logger.info(f"memmap_dir:  {memmap_dir}")

    meta = json.loads((memmap_dir / "meta_features.json").read_text())
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    logger.info(f"features.dat shape={shape} dtype={dtype}")

    if world_size > 1 and shape[0] < world_size:
        raise ValueError(
            f"patch数 ({shape[0]}) が world_size ({world_size}) 未満のため空プロセスが出ます"
        )

    # ── Data ─────────────────────────────────────────────────────────────────
    batch_size = int(tt.get("batch_size", 64))
    dataset = MemmapFeatureDataset(memmap_dir / "features.dat", shape, dtype)

    # rank分割: lib/trainer/data.py の本番パターンと同じ(world_size==1では従来のRandomSampler)。
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, drop_last=True, num_workers=2,
    )

    # ── Model (tiny MLP backbone + real BarlowTwins projector/loss) ───────────
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available; falling back to CPU (run_slurm.sh requests --gres=gpu:N)")

    feat_dim = shape[1]
    hidden_dim = int(tt.get("hidden_dim", 512))
    proj_dim = int(tt.get("proj_dim", 128))

    backbone = nn.Sequential(
        nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
    )
    model = BarlowTwins(backbone, head_size=[hidden_dim, hidden_dim, proj_dim]).to(device)
    model = distributed.wrap(model)  # world_size>1でSyncBN化+DDPラップ、単一プロセスではno-op

    # gather_distributed=True でrank間のcross-correlation行列をall_reduceする
    # (これをFalseのままだとGPUを増やしても各rankのローカルバッチだけで損失を
    # 計算してしまい、実効バッチサイズが数学的に増えない。Goal.yamlの既知の落とし穴)。
    criterion = BarlowTwinsLoss(gather_distributed=world_size > 1)
    lr = float(tt.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    noise_std = float(tt.get("noise_std", 0.1))
    dropout_p = float(tt.get("feature_dropout_p", 0.2))
    num_epochs = int(tt.get("num_epochs", 3))

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    epoch_losses: list[float] = []
    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)  # 未呼び出しだと全epochで同じrank分割・同じ順序になる

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

        epoch_loss_local = running_loss / n_batches
        # rank間の平均をとってから記録する(1rankだけの値だと他rankの進み具合が分からない)
        epoch_loss = distributed.all_reduce_mean(
            torch.tensor(epoch_loss_local, device=device)
        ).item()
        epoch_losses.append(epoch_loss)
        logger.info(f"epoch {epoch + 1}/{num_epochs}  loss={epoch_loss:.4f} (local={epoch_loss_local:.4f})")

    distributed.barrier()  # 全rankが学習を終えてからrank0の書き出しへ進む

    if is_main:
        results: dict = {
            "memmap_dir": str(memmap_dir),
            "n_patches": shape[0],
            "feat_dim": feat_dim,
            "world_size": world_size,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size_per_rank": batch_size,
            "effective_batch_size": batch_size * world_size,
            "num_epochs": num_epochs,
            "lr": lr,
            "epoch_losses": epoch_losses,
            "loss_decreased": epoch_losses[-1] < epoch_losses[0] if len(epoch_losses) > 1 else None,
            "status": "ok",
        }

        # ── Save results ─────────────────────────────────────────────────────
        (run_dir / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False)
        )
        complete_run(run_dir)
        logger.info("Done.")

    distributed.barrier()  # rank0の書き出し完了を他rankが待ってからteardown()へ進む


if __name__ == "__main__":
    main()
