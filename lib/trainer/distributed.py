# -*- coding: utf-8 -*-
"""
# 分散学習の抽象化

`RANK`/`WORLD_SIZE`/`LOCAL_RANK` が設定されていない単一プロセス実行（従来の
単一GPU sbatch 投入）では、すべて no-op のまま動作する（後方互換必須、
Goal.yaml 2026-08-03 追記）。`torchrun` 等でこれらの環境変数が設定された
場合のみ `torch.distributed` を初期化し、DDP を有効化する。

    setup()             -> 環境変数があれば init_process_group、無ければ何もしない
    teardown()          -> 初期化していれば destroy_process_group
    rank() / world_size() / local_rank() -> 環境変数 RANK / WORLD_SIZE / LOCAL_RANK
    is_main_process()   -> rank() == 0
    wrap(model)          -> world_size>1 なら SyncBatchNorm化 + DistributedDataParallel
    unwrap(model)        -> DDPラップ後も生の(state_dict互換の)モジュールを取り出す
"""
import os

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def is_distributed_env() -> bool:
    """torchrun 等が分散実行用の環境変数を設定しているか。"""
    return "WORLD_SIZE" in os.environ and _env_int("WORLD_SIZE", 1) > 1


def setup():
    """プロセスグループを初期化する。単一プロセス実行では何もしない。"""
    if not is_distributed_env():
        return None
    if dist.is_available() and dist.is_initialized():
        return None
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return None


def teardown():
    """プロセスグループを破棄する。初期化していなければ何もしない。"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return None


def rank() -> int:
    """このプロセスのグローバルランク。分散未初期化なら常に0。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def world_size() -> int:
    """参加プロセス数。分散未初期化なら常に1。"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def local_rank() -> int:
    """ノード内でのローカルランク（GPUインデックスに対応）。"""
    return _env_int("LOCAL_RANK", 0)


def is_main_process() -> bool:
    """ログ・チェックポイント書き出しを担当するプロセスかどうか。"""
    return rank() == 0


def is_initialized() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def wrap(model):
    """モデルを分散用にラップする。単一プロセスではそのまま返す。

    world_size > 1 のときのみ、BatchNorm を SyncBatchNorm に変換してから
    DistributedDataParallel でラップする（Goal.yaml 2026-08-03: ResNet50/
    DenseNet121バックボーンや各種ProjectionHeadのBatchNormがローカルバッチ
    統計のままだと有効バッチサイズが小さいままになるため）。
    """
    if not is_initialized() or world_size() <= 1:
        return model
    if torch.cuda.is_available():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        device_id = local_rank()
        model = model.to(device_id)
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device_id], output_device=device_id,
            find_unused_parameters=False,
        )
    # CPU/gloo経路（GPUなし環境でのDDP配線のスモークテスト用）
    return torch.nn.parallel.DistributedDataParallel(model)


def unwrap(model):
    """DDPラップされていれば `.module` を返し、そうでなければそのまま返す。

    state_dict の保存や、DINO の `update_moving_average` のようなカスタム
    メソッド呼び出しは、DDPありなし双方で同じキー・同じ呼び出し方にする
    必要があるため、これを経由させる。
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def barrier():
    """全rankが揃うまで待つ（未初期化なら何もしない)。

    rank0のみがチェックポイント/サマリを書く箇所の前後で使い、rank0の書き込み完了前に
    他rankが teardown()(=destroy_process_group)へ進んでしまわないようにする。
    """
    if is_initialized() and world_size() > 1:
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """全rankでtensorを合計し world_size で割る（未初期化なら何もしない）。"""
    if not is_initialized() or world_size() <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()
    return tensor
