# -*- coding: utf-8 -*-
"""
# 学習データのロード

WebDataset shard の fold 分割と DataLoader 構築。
分散学習に移行する際は、ここに split_by_node / split_by_worker を足す
（REFACTOR_PLAN.md §6-4）。

"""
import glob

import numpy as np
import torch
from torch.utils.data import default_collate
import webdataset as wds

from lib.trainer import distributed
from lib.trainer.context import RunContext


def get_shard_file_list(data_dir: str):
    """ディレクトリ内の .tar ファイルをソートして取得"""
    shards = sorted(glob.glob(f"{data_dir}/*.tar"))
    return np.array(shards)


def create_sharded_dataset(urls, batch_size, transform=None, is_train=True, split_by_rank=False):
    """WebDatasetのパイプラインを作成（高速化＆無限ループ修正版）

    split_by_rank: Trueなら `wds.split_by_node` でrank単位にshardを分割する
    （マルチGPU時、各GPUプロセスが同じshardを重複して読むのを防ぐ。
    Goal.yaml 2026-08-03 / docs/multi_gpu_migration.md §1）。
    world_size==1（単一GPU/未分散）では`wds.split_by_node`は素通りするだけなので、
    既存の単一GPU実行の挙動は変わらない。
    """
    # 修正1: resampled=is_train を削除（無限ループ回避）
    dataset = wds.WebDataset(
        urls,
        shardshuffle=is_train,  # 学習時はTrue、検証時はFalse
        empty_check=False,  # 空のシャードをスキップ
        nodesplitter=wds.split_by_node if split_by_rank else wds.shardlists.single_node_only,
    )
    
    if is_train:
        # メモリを節約するためバッファを少し小さくする
        dataset = dataset.shuffle(1000)

    dataset = (
        dataset
        .decode("pil")
        .to_tuple("jpg;png;jpeg", "json") 
    )

    def apply_transform(sample):
        img, meta = sample
        if transform:
            # transformがリストでない（BarlowTwins用のCompose等）前提
            img = transform(img)
        return img  # モデルの入力に合わせて調整（通常はAugmentされた画像ペア）

    dataset = dataset.map(apply_transform)
    
    # 修正2: WebDataset側でバッチ化を行う（超重要）
    # drop_last=True に相当する partial=False を設定
    dataset = dataset.batched(batch_size, partial=False, collation_fn=default_collate)
    
    return dataset
    
def prepare_data(
    ctx: RunContext,
    data_dir: str="data/shards",
    fold_idx: int = 0,
    num_folds: int = 5,
    batch_size: int = 32
    ):
    """shardsを分割して特定のfoldのtrain/evalローダーを返す"""
    args = ctx.args
    all_shards = get_shard_file_list(data_dir)
    if len(all_shards) == 0:
        raise FileNotFoundError(f"No .tar files found in {data_dir}")
    if not 0 <= fold_idx < num_folds:
        raise ValueError(f"fold_idx must be in [0, {num_folds}), got {fold_idx}")
    if len(all_shards) < num_folds:
        raise ValueError(
            f"shard 数 ({len(all_shards)}) が num_folds ({num_folds}) 未満のため "
            f"空の fold ができる。--num_folds を減らすか shard を増やすこと。"
        )

    folds = np.array_split(all_shards, num_folds)
    val_shards = folds[fold_idx].tolist()
    # fold のインデックスで除外する。以前は `if f[0] not in val_shards` と
    # 「fold の先頭 shard が val に含まれるか」で判定していたため、空 fold があると
    # f[0] が IndexError になった（REFACTOR_PLAN.md §3-4）。
    # fold 同士は素なので、通常ケースでの選択結果は従来と同一。
    train_shards = [s for i, f in enumerate(folds) if i != fold_idx for s in f]

    world_size = distributed.world_size()
    if world_size > 1 and len(train_shards) < world_size:
        raise ValueError(
            f"train shard 数 ({len(train_shards)}) が world_size ({world_size}) 未満のため "
            f"空プロセスが出る。shard を増やすか GPU 数を減らすこと "
            f"（docs/multi_gpu_migration.md §5）。"
        )

    print(f"Fold {fold_idx}: Train shards={len(train_shards)}, Val shards={len(val_shards)}"
          + (f", world_size={world_size}" if world_size > 1 else ""))

    # --- Transforms ---
    train_transform = ctx.ssl_class.prepare_transform(
        color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob,
    )
    # 決定的リサイズのみの val_transform / val_dataset / val_loader は削除した。
    # 検証損失（compute_val_loss）も SSL 指標（evaluator.evaluate）も eval_loader を
    # 使っており、どこからも参照されていなかった（REFACTOR_PLAN.md §3-1）。
    # SSL の pretext 損失には拡張ペアが必要なので eval_loader を使うのが正しい。

    # --- Datasets ---
    # batch_size を WebDataset に渡す
    num_workers_train = 16
    # epoch長はworld_size台のプロセス全体で ~1,000,000 枚/epoch になるよう、
    # rank単体の目標枚数を world_size で割る（docs/multi_gpu_migration.md §1
    # 「epoch長のworld_size考慮」）。world_size==1では従来と同じ値になる。
    samples_per_rank_per_epoch = 1_000_000 // world_size
    train_dataset = create_sharded_dataset(
        train_shards,
        batch_size=batch_size,
        transform=train_transform,
        is_train=True,
        split_by_rank=True,  # shardをrank単位に分割（重複読み込み防止）
    ).with_epoch(samples_per_rank_per_epoch // (args.batch_size * num_workers_train))
    # eval_dataset: val shards with augmented pairs for alignment/uniformity.
    # 意図的に split_by_rank=False（全rankが同一のval shard集合を読む）:
    # early stopping / effective-rank監視の判定を全rankで一致させ、rank0限定の
    # チェックポイント保存・ログと矛盾なく動かすため（docs/multi_gpu_migration.md §1,2）。
    eval_dataset = create_sharded_dataset(val_shards, batch_size=batch_size, transform=train_transform, is_train=False)

    # --- DataLoaders ---
    # 修正3: DataLoaderの batch_size を None にし、num_workers を増やす
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers_train,             # 8コアなら6が上限
        pin_memory=True,
        persistent_workers=True,   # ← 追加(worker再生成を回避)
        prefetch_factor=4,         # ← 復活(GPUを待たせない)
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=None,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, eval_loader
