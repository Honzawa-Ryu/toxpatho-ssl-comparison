# -*- coding: utf-8 -*-
"""
# 分散学習の抽象化

現時点ではシングルGPU実行しかしないため全て no-op。
複数GPUへ移行する際の変更をこのファイルに閉じ込めるために用意している
（REFACTOR_PLAN.md §6-4 / §5-3）。

将来の実装:
    setup()            -> torch.distributed.init_process_group(...)
    teardown()         -> torch.distributed.destroy_process_group()
    rank() / world_size() -> 環境変数 RANK / WORLD_SIZE
    wrap(model)        -> DistributedDataParallel(model, device_ids=[local_rank])
"""


def setup():
    """プロセスグループを初期化する。シングルGPUでは何もしない。"""
    return None


def teardown():
    """プロセスグループを破棄する。シングルGPUでは何もしない。"""
    return None


def rank() -> int:
    """このプロセスのグローバルランク。シングルGPUでは常に0。"""
    return 0


def world_size() -> int:
    """参加プロセス数。シングルGPUでは常に1。"""
    return 1


def is_main_process() -> bool:
    """ログ・チェックポイント書き出しを担当するプロセスかどうか。"""
    return rank() == 0


def wrap(model):
    """モデルを分散用にラップする。シングルGPUではそのまま返す。"""
    return model
