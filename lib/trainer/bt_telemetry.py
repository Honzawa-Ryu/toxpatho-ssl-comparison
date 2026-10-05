"""Barlow Twins の step 単位テレメトリ。読み取り専用で、学習の挙動には影響しない。

動機: exp 0029 は ep64 まで損失・grad_norm とも単調に下がっていたのに、ep65 の epoch 平均が
突然 NaN になった。ログが epoch 平均だけだったので、いつ・どの統計から壊れたのかが分からず、
原因(bf16 の標準化の桁落ち / LARS の高原)を切り分けられなかった。step ごとの損失・grad_norm と、
`BarlowTwinsLoss.last_stats`(z の std 最小値・std==0 の次元数・オフセット/ばらつき比など)を残す。
clip_logging.py と同じく、epoch 末に1回だけ device->host 転送し、rank0 が書く。
"""
import json
import math
import time
from pathlib import Path

# BarlowTwinsLoss._record_stats の並びと一致させること
STAT_NAMES = ("z_std_min", "n_zero_std", "z_mean_abs_max", "z_ratio_max",
              "c_diag_mean", "on_diag", "off_diag")


def _num(x):
    """JSON は NaN/inf を表せない(allow_nan=False で書く)ので、非有限は文字列で残す。"""
    x = float(x)
    return x if math.isfinite(x) else str(x)


def step_rows(epoch, losses, grad_norms, stats, first_nonfinite=None):
    """1 epoch 分の行(dict)を作る。stats は step ごとの値のリスト(無ければ空)。"""
    rows = []
    for step, (loss, gn) in enumerate(zip(losses, grad_norms)):
        row = dict(epoch=epoch, step=step, loss=_num(loss), grad_norm=_num(gn))
        if step < len(stats):
            row.update({name: _num(v) for name, v in zip(STAT_NAMES, stats[step])})
        if first_nonfinite == step:
            row["first_nonfinite"] = True
        rows.append(row)
    return rows


def write_epoch(directory, epoch, losses, grad_norms, stats_by_step, first_nonfinite=None):
    """rank0 だけが呼ぶ。atomic に公開し、再開した試行はタイムスタンプで区別する。"""
    import torch

    stats = torch.stack(stats_by_step).cpu().tolist() if stats_by_step else []
    path = Path(directory) / f"bt_steps_ep{epoch:04d}_{time.time_ns()}.jsonl"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in step_rows(epoch, losses, grad_norms, stats, first_nonfinite):
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)
