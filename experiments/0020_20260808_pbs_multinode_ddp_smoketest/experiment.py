# -*- coding: utf-8 -*-
"""2ノード疎通確認用の最小スクリプト。data/lib一切不使用、依存はtorchのみ。

torchrunがRANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORTを設定した状態で
起動される前提(scripts/slurm_entry.sh の _run_multi_node 経由)。
各rankがNCCLでprocess groupに参加できるか、all_reduceが全ノードにまたがって
正しい値を返すかだけを確認する。
"""
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    x = torch.tensor([float(rank)], device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = float(sum(range(world_size)))
    ok = "OK" if x.item() == expected else "MISMATCH"

    line = (
        f"[rank {rank}/{world_size}] host={hostname} local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)} "
        f"all_reduce_result={x.item()} expected={expected} {ok}"
    )
    print(line, flush=True)

    # pbsdsh経由だと標準出力がジョブスクリプト側に届かないことがあるため
    # (2026-08-08、job 2507316で exit_status=0 なのにログにprint出力が
    # 一切見当たらないことを実機確認)、共有ストレージ(/work)上の
    # JOB_LOG_DIR にも直接書く。これなら計算ノードの/local(=ノード終了で
    # 消える)ではなく、ジョブ終了後もログインノードから読める。
    job_log_dir = os.environ.get("JOB_LOG_DIR")
    if job_log_dir:
        Path(job_log_dir, f"rank{rank}.log").write_text(line + "\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
