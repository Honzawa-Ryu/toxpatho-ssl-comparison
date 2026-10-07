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

import torch

# BarlowTwinsLoss._record_stats の並びと一致させること
STAT_NAMES = ("z_std_min", "n_zero_std", "z_mean_abs_max", "z_ratio_max",
              "c_diag_mean", "on_diag", "off_diag")


def _num(x):
    """JSON は NaN/inf を表せない(allow_nan=False で書く)ので、非有限は文字列で残す。"""
    x = float(x)
    return x if math.isfinite(x) else str(x)


def step_rows(epoch, losses, grad_norms, stats, first_nonfinite=None, groups=()):
    """1 epoch 分の行(dict)を作る。stats / groups は step ごとの値のリスト(無ければ空)。"""
    rows = []
    for step, (loss, gn) in enumerate(zip(losses, grad_norms)):
        row = dict(epoch=epoch, step=step, loss=_num(loss), grad_norm=_num(gn))
        if step < len(stats):
            row.update({name: _num(v) for name, v in zip(STAT_NAMES, stats[step])})
        if step < len(groups):
            row.update({name: _num(v) for name, v in zip(GROUP_NAMES, groups[step])})
        if first_nonfinite == step:
            row["first_nonfinite"] = True
        rows.append(row)
    return rows


def write_epoch(directory, epoch, losses, grad_norms, stats_by_step, first_nonfinite=None, groups=()):
    """rank0 だけが呼ぶ。atomic に公開し、再開した試行はタイムスタンプで区別する。
    groups は GradGroupProbe.rows() (無ければ空)。"""
    stats = torch.stack(stats_by_step).cpu().tolist() if stats_by_step else []
    path = Path(directory) / f"bt_steps_ep{epoch:04d}_{time.time_ns()}.jsonl"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in step_rows(epoch, losses, grad_norms, stats, first_nonfinite, groups):
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# グループ別 grad_norm（2026-10-07, 0030 の結果を受けて追加）
#
# 0030 で起点は「grad_norm が 17→29→99→639→2219 と数 step で指数的に増えること」と分かったが、
# 全パラメータ合計の L2 ノルムしか無く、**どのグループが先に跳ねたか**が分からなかった。
# LARS の更新は重み(ndim>1)では eta·||p||·g/||g|| に正規化されて勾配の大きさに依らない一方、
# bias / LayerNorm・BatchNorm(ndim<=1, --lars_exclude_bias_bn)は生の勾配に lr_bias を掛けて
# 更新する(lib/trainer/optim.py)。仮説 H2' はこの生勾配グループの正のフィードバック。
# 下の4分割 + 上位テンソルで H2' を検証する:
#   gn_lars     ndim>1 (LARS 適応あり)         gn_raw   ndim<=1 (LARS 適応なし・生勾配)
#   gn_backbone 名前が backbone. で始まる       gn_head  それ以外(projection_head)
#   top{1,2,3}_idx / _norm  パラメータ毎ノルムの上位3つ(名前は bt_param_names.json の idx 番目)
# すべて device 上の演算で、.item() の同期は増やさない。
# ---------------------------------------------------------------------------
GROUP_NAMES = ("gn_lars", "gn_raw", "gn_backbone", "gn_head",
               "top1_idx", "top1_norm", "top2_idx", "top2_norm", "top3_idx", "top3_norm")


class GradGroupProbe:
    """epoch の最初に作る。`record(params, per_param_norms)` を step ごとに呼び、
    `rows()` を epoch 末に取り出す。params は loop.py が grads を作ったのと同じ順序の
    パラメータ列(p.grad is not None のもの)。"""

    def __init__(self, model, backbone_prefix="backbone."):
        self.names, params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.names.append(name[len("module."):] if name.startswith("module.") else name)
            params.append(p)
        self._ids = [id(p) for p in params]
        dev = params[0].device if params else "cpu"
        is_raw = torch.tensor([p.ndim <= 1 for p in params], device=dev)
        is_bb = torch.tensor([n.startswith(backbone_prefix) for n in self.names], device=dev)
        self.masks = (~is_raw, is_raw, is_bb, ~is_bb)
        self.records = []

    @torch.no_grad()
    def record(self, params, per_param_norms):
        if [id(p) for p in params] != self._ids:
            # 勾配を持つパラメータの集合が変わった(通常は起きない)。列を揃えられないので NaN を残す。
            self.records.append(torch.full((len(GROUP_NAMES),), float("nan"),
                                           device=per_param_norms[0].device))
            return
        norms = torch.stack(per_param_norms).float()
        sq = norms * norms
        groups = [(sq * m).sum().sqrt() for m in self.masks]
        k = min(3, norms.numel())
        top_v, top_i = torch.topk(norms, k)
        top = []
        for j in range(3):
            if j < k:
                top += [top_i[j].float(), top_v[j]]
            else:
                top += [torch.tensor(float("nan"), device=norms.device)] * 2
        self.records.append(torch.stack(groups + top))

    def rows(self):
        return torch.stack(self.records).cpu().tolist() if self.records else []


def write_param_names(directory, names):
    """bt_param_names.json(idx -> パラメータ名)。rank0 が 1 度だけ書く。既にあれば触らない。"""
    path = Path(directory) / "bt_param_names.json"
    if path.exists():
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(names, ensure_ascii=False, indent=0))
    temporary.replace(path)
