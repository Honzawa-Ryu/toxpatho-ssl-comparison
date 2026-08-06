# -*- coding: utf-8 -*-
"""
# 学習データのロード

`lib/analysis/blur_qc.py`(`sample_patches_to_shared_memmap`)が書き出す
全WSI共有 memmap（`{data_dir}/patches.memmap` + `{data_dir}/index.csv`）を
読む Dataset と、fold 分割済み DataLoader の構築。

旧 WebDataset(.tar shard) 版からの移行（`data/shards/` は撤去済み。
EXP8: experiments/0008_20260805_sample_ssl_patches_memmap が新形式の
生成元）。map-style Dataset + memmap によりランダムアクセスでも高速に読める
ため、WebDataset 特有の epoch 長の手動指定（`with_epoch`）は不要になった。

分散学習時は `DistributedSampler` で train 行を rank ごとに分割する
（REFACTOR_PLAN.md §6-4 の "分散学習に移行する際は split_by_node 等を足す"
に対応）。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

from lib.trainer import distributed
from lib.trainer.context import RunContext


def load_index_table(data_dir: str) -> pd.DataFrame:
    """`{data_dir}/index.csv`(row, wsi_id, x, y, blur_score)を読む"""
    index_path = Path(data_dir) / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv が見つかりません: {index_path}")
    return pd.read_csv(index_path, dtype={"wsi_id": str})


class MemmapPatchDataset(Dataset):
    """共有 memmap（shape=(total_n_patches, patch_size, patch_size, 3), dtype=uint8）
    のうち、`rows` で指定した行だけを読む map-style Dataset。

    `np.memmap` は DataLoader worker プロセス（fork）をまたいで使い回すと
    問題が起きやすいため、`__init__` では開かず worker 内で最初の
    `__getitem__` 呼び出し時に遅延オープンする。
    """

    def __init__(self, memmap_path: str, patch_size: int, rows: np.ndarray, transform=None):
        self.memmap_path = memmap_path
        self.patch_size = patch_size
        self.rows = np.asarray(rows)
        self.transform = transform
        self._mm = None

    def __len__(self) -> int:
        return len(self.rows)

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.memmap_path, dtype=np.uint8, mode="r").reshape(
                -1, self.patch_size, self.patch_size, 3
            )
        return self._mm

    def __getitem__(self, idx: int):
        row = int(self.rows[idx])
        # np.array(...) でmmapページ由来のバッファをコピーする
        # (PIL Image / 後続のtransformがmmap参照を持ち越さないようにするため)
        patch = np.array(self._memmap()[row])
        img = Image.fromarray(patch, mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img


def split_wsi_ids_by_fold(wsi_ids: np.ndarray, fold_idx: int, num_folds: int) -> tuple[set, set]:
    """ソート済みwsi_idをnum_folds個に分割し、fold_idx番目をval、残りをtrainとする。"""
    if not 0 <= fold_idx < num_folds:
        raise ValueError(f"fold_idx must be in [0, {num_folds}), got {fold_idx}")
    if len(wsi_ids) < num_folds:
        raise ValueError(
            f"WSI 数 ({len(wsi_ids)}) が num_folds ({num_folds}) 未満のため "
            f"空の fold ができる。--num_folds を減らすか WSI を増やすこと。"
        )
    folds = np.array_split(np.sort(wsi_ids), num_folds)
    val_wsi_ids = set(folds[fold_idx].tolist())
    train_wsi_ids = {w for i, f in enumerate(folds) if i != fold_idx for w in f.tolist()}
    return train_wsi_ids, val_wsi_ids


def prepare_data(
    ctx: RunContext,
    data_dir: str | None = None,
    patch_size: int = 224,
    fold_idx: int = 0,
    num_folds: int = 5,
    batch_size: int = 32,
):
    """`data_dir` の共有 memmap を WSI 単位で fold 分割し、train/eval ローダーを返す。

    WSI 単位で分割するのは、同じ WSI 由来のパッチが train/val 両方に混ざって
    リークするのを避けるため（旧 shard 版の「shard 単位で fold 分割」と同じ考え方）。

    `data_dir` 省略時は `${DATASET_DIR}/ssl_patches` を使う。DATASET_DIR は
    scripts/slurm_entry.sh が USE_LOCAL_SSD_INPUT の有無に関わらず常に export する
    (0なら${PROJECT_ROOT}/data、1ならノードローカルSSDへrsync済みのコピー)。
    これまでこの関数がDATASET_DIRを見ずdata_dirを常にリポジトリ直下の相対パス
    "data/ssl_patches"に固定していたため、USE_LOCAL_SSD_INPUT=1にしても実際には
    NFS(${PROJECT_ROOT}/data/ssl_patches)を読み続けてしまうバグがあった。
    """
    if data_dir is None:
        data_dir = str(Path(os.environ.get("DATASET_DIR", "data")) / "ssl_patches")
    args = ctx.args
    index_df = load_index_table(data_dir)
    memmap_path = str(Path(data_dir) / "patches.memmap")
    if not Path(memmap_path).exists():
        raise FileNotFoundError(f"patches.memmap が見つかりません: {memmap_path}")

    wsi_ids = index_df["wsi_id"].unique()
    if len(wsi_ids) == 0:
        raise FileNotFoundError(f"{data_dir}/index.csv にパッチがありません")

    train_wsi_ids, val_wsi_ids = split_wsi_ids_by_fold(wsi_ids, fold_idx, num_folds)
    train_rows = index_df.loc[index_df["wsi_id"].isin(train_wsi_ids), "row"].to_numpy()
    val_rows = index_df.loc[index_df["wsi_id"].isin(val_wsi_ids), "row"].to_numpy()

    world_size = distributed.world_size()
    if world_size > 1 and len(train_rows) < world_size:
        raise ValueError(
            f"train patch 数 ({len(train_rows)}) が world_size ({world_size}) 未満のため "
            f"空プロセスが出る。WSI を増やすか GPU 数を減らすこと "
            f"（docs/multi_gpu_migration.md §5）。"
        )

    print(
        f"Fold {fold_idx}: Train WSIs={len(train_wsi_ids)} ({len(train_rows)} patches), "
        f"Val WSIs={len(val_wsi_ids)} ({len(val_rows)} patches)"
        + (f", world_size={world_size}" if world_size > 1 else "")
    )

    # --- Transforms ---
    train_transform = ctx.ssl_class.prepare_transform(
        color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob,
    )
    # 決定的リサイズのみの val_transform / val_dataset / val_loader は削除した(旧実装踏襲)。
    # SSL の pretext 損失には拡張ペアが必要なので eval_loader も train_transform を使う。

    # --- Datasets ---
    train_dataset = MemmapPatchDataset(memmap_path, patch_size, train_rows, transform=train_transform)
    eval_dataset = MemmapPatchDataset(memmap_path, patch_size, val_rows, transform=train_transform)

    # rank分割: WebDataset版のsplit_by_node相当。DistributedSamplerがtrain行を
    # rankごとに非重複に割り振る(world_size==1では従来のRandomSamplerと同じ)。
    # eval側は意図的に分割しない(全rankが同一val集合を読み、early stopping /
    # effective-rank監視の判定をrank間で一致させるため。旧実装のsplit_by_rank=Falseと同じ)。
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=distributed.rank(),
            shuffle=True, drop_last=True,
        )

    # --- DataLoaders ---
    num_workers_train = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers_train,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, eval_loader
