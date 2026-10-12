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
# SimSiam(lib/sslmodel/sslutils.py SimSiam.calc_loss)の並びと一致させること。
#   z_nstd_sqrtd: L2 正規化した z の次元ごとの std の平均 × sqrt(d)。SimSiam 論文 Fig.2 の崩壊指標
#                 (健全なら ≈1、完全崩壊で 0)。
#   z_std_min:    z(投影出力, BN affine なし)の次元ごとの std の最小値(定数に潰れた次元の検出)
#   cos_pz:       予測 p と相手ビューの z の cos(= -損失)。cos_zz: 2 ビューの z 同士の cos
#   p_nstd_sqrtd: p について同じ指標(predictor 側の崩壊)
#   z_erank / p_erank: z / p の実効ランク(特異値のエントロピーの exp、rank0 のローカルバッチ)。
#                 ⚠️ 投影の出力 BN(affine なし)が次元ごとの std を 1 に揃えるので、z_std_min や z_nstd_sqrtd は
#                 ランク崩壊(全次元が同じ方向)を見逃しうる。崩壊はこの実効ランクと cos_zz(→1)で見る。
SIMSIAM_STAT_NAMES = ("z_nstd_sqrtd", "z_std_min", "cos_pz", "cos_zz", "p_nstd_sqrtd", "z_erank", "p_erank")


def _num(x):
    """JSON は NaN/inf を表せない(allow_nan=False で書く)ので、非有限は文字列で残す。"""
    x = float(x)
    return x if math.isfinite(x) else str(x)


def step_rows(epoch, losses, grad_norms, stats, first_nonfinite=None, groups=(), stat_names=STAT_NAMES,
              group_names=None):
    """1 epoch 分の行(dict)を作る。stats / groups は step ごとの値のリスト(無ければ空)。
    stat_names は criterion ごとの統計名(BT は STAT_NAMES、SimSiam は SIMSIAM_STAT_NAMES)。"""
    rows = []
    for step, (loss, gn) in enumerate(zip(losses, grad_norms)):
        row = dict(epoch=epoch, step=step, loss=_num(loss), grad_norm=_num(gn))
        if step < len(stats):
            row.update({name: _num(v) for name, v in zip(stat_names, stats[step])})
        if step < len(groups):
            row.update({name: _num(v) for name, v in zip(group_names or GROUP_NAMES, groups[step])})
        if first_nonfinite == step:
            row["first_nonfinite"] = True
        rows.append(row)
    return rows


def write_epoch(directory, epoch, losses, grad_norms, stats_by_step, first_nonfinite=None, groups=(),
                stat_names=STAT_NAMES, prefix="bt_steps", group_names=None):
    """rank0 だけが呼ぶ。atomic に公開し、再開した試行はタイムスタンプで区別する。
    groups は GradGroupProbe.rows() (無ければ空)。prefix は BT が "bt_steps"(既存の解析と互換)、
    他手法は "steps"。どちらも `*steps_ep{epoch:04d}_*.jsonl` で拾える。"""
    stats = torch.stack(stats_by_step).cpu().tolist() if stats_by_step else []
    path = Path(directory) / f"{prefix}_ep{epoch:04d}_{time.time_ns()}.jsonl"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in step_rows(epoch, losses, grad_norms, stats, first_nonfinite, groups, stat_names, group_names):
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
#   gn_backbone 名前が backbone. で始まる       gn_head  それ以外(projection_head / predictor)
#   (MAE は backbone. 接頭辞が無いので、エンコーダ = gn_backbone、decoder_* / mask_token = gn_head)
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
        if any(n.startswith(backbone_prefix) for n in self.names):
            bb = [n.startswith(backbone_prefix) for n in self.names]
        else:
            # MAE はエンコーダが backbone. 接頭辞を持たない(patch_embed / blocks / norm)。
            # デコーダ(decoder_* / mask_token)を head 側として分ける。
            bb = [not n.startswith(("decoder", "mask_token")) for n in self.names]
        is_bb = torch.tensor(bb, device=dev)
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

    @property
    def column_names(self):
        return GROUP_NAMES


# ---------------------------------------------------------------------------
# 更新量・重み・内部活性の step テレメトリ(2026-10-12, 「崩壊時に原因を究明できるように」)
#
# 勾配(GradGroupProbe)だけでは「その勾配で重みがどれだけ動いたか」が分からない。LARS は重みの更新を
# 勾配の大きさから切り離し、clip や Adam も更新と勾配の関係を変えるので、発散の機構(H2' のような
# 正のフィードバック)を検証するには**実際の更新量**が要る。さらに崩壊は表現の内部(どの block の出力が
# 先に潰れる / 膨らむか)から始まることがあるので、backbone の各 block の出力の大きさも残す。
#   upd_w / upd_raw     optimizer.step() 前後の重みの差の L2 ノルム(ndim>1 / ndim<=1)
#   pn_w / pn_raw       step 後の重みの L2 ノルム
#   upd_ratio_max/_idx  テンソルごとの ||Δp|| / ||p|| の最大値とそのテンソル(名前は bt_param_names.json)
#   ln_gain_mean / _min LayerNorm/BN のゲイン(ndim==1 の *.weight。backbone のみ)の平均と最小
#   act_patch, act_b0..  backbone の patch_embed と各 block の出力の RMS(この step の forward の平均。
#                        2 ビューの手法は両ビューの平均)。学習中の forward だけを測る(評価・val は測らない)
# どれも device 上の演算で .item() の同期は増やさない。更新量のために step 前の重みを 1 回複製する
# (BT ViT-B + 8192 ヘッドで約 0.9GB。GPU メモリに余裕がある前提)。
# ---------------------------------------------------------------------------
DYN_NAMES = ("upd_w", "upd_raw", "pn_w", "pn_raw", "upd_ratio_max", "upd_ratio_idx", "ln_gain_mean", "ln_gain_min")

import re as _re
_BLOCK_RE = _re.compile(r"^(module\.)?(backbone\.|student_backbone\.)?blocks\.(\d+)$")
_PATCH_RE = _re.compile(r"^(module\.)?(backbone\.|student_backbone\.)?patch_embed$")


class StepProbe(GradGroupProbe):
    """GradGroupProbe に更新量・重み・内部活性を足したもの。loop.py は
    record(勾配) → before_step(params) → optimizer.step() → after_step(params) の順に呼び、epoch 末に close()。"""

    def __init__(self, model, backbone_prefix="backbone."):
        super().__init__(model, backbone_prefix)
        dev = self.masks[0].device
        named = self._named_trainable(model)
        has_bb = any(n.startswith(backbone_prefix) for n, _ in named)
        # backbone の LN ゲインだけ(ヘッドの BN は含めない)。MAE は接頭辞が無いのでデコーダを除いたもの
        ln = [n.endswith(".weight") and p.ndim == 1
              and (n.startswith(backbone_prefix) if has_bb else not n.startswith(("decoder", "mask_token")))
              for n, p in named]
        self.ln_mask = torch.tensor(ln, device=dev)
        self._prev = None
        self._dyn = None
        # 活性のフック(学習中の forward だけ数える: active のときだけ貯める)
        self.active = False
        mods = []
        for name, m in model.named_modules():
            if _PATCH_RE.match(name):
                mods.append(("act_patch", m))
        blocks = sorted(((int(_BLOCK_RE.match(n).group(3)), m) for n, m in model.named_modules() if _BLOCK_RE.match(n)),
                        key=lambda t: t[0])
        mods += [(f"act_b{i}", m) for i, m in blocks]
        self.act_names = tuple(n for n, _ in mods)
        self._act_sum = torch.zeros(len(mods), device=dev)
        self._act_cnt = torch.zeros(len(mods), device=dev)
        self._handles = []
        for k, (_, m) in enumerate(mods):
            self._handles.append(m.register_forward_hook(self._make_hook(k)))

    @staticmethod
    def _named_trainable(model):
        return [(n[len("module."):] if n.startswith("module.") else n, p)
                for n, p in model.named_parameters() if p.requires_grad]

    def _make_hook(self, k):
        def hook(_module, _inp, out):
            if not self.active:
                return
            o = out[0] if isinstance(out, (tuple, list)) else out
            with torch.no_grad():
                self._act_sum[k] += o.detach().float().pow(2).mean().sqrt()
                self._act_cnt[k] += 1
        return hook

    @torch.no_grad()
    def before_step(self, params):
        self._prev = [p.detach().clone() for p in params]

    @torch.no_grad()
    def after_step(self, params):
        nan = torch.tensor(float("nan"), device=self._act_sum.device)
        if self._prev is None or [id(p) for p in params] != self._ids:
            dyn = [nan] * len(DYN_NAMES)
        else:
            cur = [p.detach() for p in params]
            dn = torch.stack(torch._foreach_norm(torch._foreach_sub(cur, self._prev))).float()
            pn = torch.stack(torch._foreach_norm(cur)).float()
            w, raw = self.masks[0], self.masks[1]
            ratio = dn / pn.clamp_min(1e-12)
            rmax, ridx = ratio.max(0)
            gains = torch.stack([p.detach().float().mean() for p, m in zip(cur, self.ln_mask.tolist()) if m]) \
                if bool(self.ln_mask.any()) else nan.view(1)
            dyn = [((dn * dn) * w).sum().sqrt(), ((dn * dn) * raw).sum().sqrt(),
                   ((pn * pn) * w).sum().sqrt(), ((pn * pn) * raw).sum().sqrt(),
                   rmax, ridx.float(), gains.mean(), gains.min()]
        act = self._act_sum / self._act_cnt.clamp_min(1)
        act = torch.where(self._act_cnt > 0, act, torch.full_like(act, float("nan")))
        self._act_sum.zero_()
        self._act_cnt.zero_()
        self._prev = None
        if self.records:
            self.records[-1] = torch.cat([self.records[-1], torch.stack(dyn), act])

    def rows(self):
        # record() だけ呼ばれて after_step() が無かった step(非有限で抜けた等)は長さが揃わないので NaN で埋める
        width = len(self.column_names)
        fixed = [r if r.numel() == width else torch.cat([r, torch.full((width - r.numel(),), float("nan"), device=r.device)])
                 for r in self.records]
        return torch.stack(fixed).cpu().tolist() if fixed else []

    @property
    def column_names(self):
        return GROUP_NAMES + DYN_NAMES + self.act_names

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def write_param_names(directory, names):
    """bt_param_names.json(idx -> パラメータ名)。rank0 が 1 度だけ書く。既にあれば触らない。"""
    path = Path(directory) / "bt_param_names.json"
    if path.exists():
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(names, ensure_ascii=False, indent=0))
    temporary.replace(path)
