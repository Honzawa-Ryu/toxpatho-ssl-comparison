# -*- coding: utf-8 -*-
"""
# barlow twins module

reference: 
https://arxiv.org/abs/2103.03230
https://docs.lightly.ai/self-supervised-learning/examples/barlowtwins.html

@author: Katsuhisa MORITA
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.
    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer).
    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self, 
        blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]]
    ):
        super(ProjectionHead, self).__init__()

        layers = []
        for input_dim, output_dim, batch_norm, non_linearity in blocks:
            use_bias = not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """Computes one forward pass through the projection head.
        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        return self.layers(x)

class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.
    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]
    [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230
    """

    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 8192,
                 output_dim: int = 8192):
        super(BarlowTwinsProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])

class _GradScale(torch.autograd.Function):
    """順伝播は恒等、逆伝播で勾配を scale 倍する(ログに出る損失の値は変えない)。

    なぜ要るか(2026-10-13): 公式実装(と本実装)は c を各 rank の局所寄与の和として all_reduce する
    (autograd 非対応の in-place all_reduce)。そのため各 rank の勾配は「真の勾配のうち自分のサンプルの寄与」で、
    DDP がそれを rank 間で**平均**するので、パラメータの勾配は **真の勾配 / world_size** になる。
    LARS の重みグループは更新が勾配の大きさに依らないので影響しないが、bias/LN グループ
    (--lars_exclude_bias_bn、生の勾配 × lr_bias)と、LARS の trust ratio 内の wd 項との比は world_size で変わる。
    公式は 16 GPU で学習している(README: "approximately 7 days on 16 v100 GPUs")ので、
    公式の lr_biases は「真の勾配 / 16」に対する値。--bt_grad_ref_world 16 で勾配を world_size / 16 倍すると、
    ノード数に依らず公式と同じ実効更新になる(ノード数を変えて高速化しても結果が変わらない)。
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.scale, None


class BarlowTwinsLoss(torch.nn.Module):
    """Implementation of the Barlow Twins Loss from Barlow Twins[0] paper.
    This code specifically implements the Figure Algorithm 1 from [0].
    
    [0] Zbontar,J. et.al, 2021, Barlow Twins... https://arxiv.org/abs/2103.03230
        Examples:
        >>> # initialize loss function
        >>> loss_fn = BarlowTwinsLoss()
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> out0, out1 = model(t0, t1)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(out0, out1)
    """

    # state.pt は criterion をオブジェクトごと pickle する(loop.py)。属性を後から足したので、
    # 古い state.pt(例: 0029 の ep64)から復元した criterion は __init__ を通らず、これらを
    # 持たない。クラス属性で既定値を持たせて AttributeError を避ける。再開時に CLI の値を
    # 反映し直すのは lib/trainer/entry.py の resume 経路。
    fp32 = False
    collect_stats = False
    last_stats = None
    grad_scale = 1.0   # --bt_grad_ref_world(下の _GradScale)。旧 pickle 互換のためクラス属性で既定値を持つ
    # 損失の内訳(2026-10-14)。毎 forward で [on_diag, off_diag](λ を掛ける前、detach 済み)を入れる。
    # loop.py が epoch 平均を取って WandB / ログに出す(train と val)。on/off は全 rank 集約後の c から計算するので全 rank 同一。
    PART_NAMES = ("on_diag", "off_diag")
    last_parts = None
    # 損失がバッチ統計(次元ごとの標準化・相関行列)に依存する。val もこの条件で測らないと比較にならない
    # (lib/trainer/loop.py:compute_val_loss)。
    batch_statistics_loss = True

    def __init__(
        self, 
        lambda_param: float = 5e-3, 
        gather_distributed : bool = False,
        fp32: bool = False,
        collect_stats: bool = False,
    ):
        """Lambda param configuration with default value like in [0]
        Args:
            lambda_param: 
                Parameter for importance of redundancy reduction term. 
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are 
                gathered and summed before the loss calculation.
            fp32:
                True なら損失計算だけ autocast を切って fp32 で行う(`BarlowTwins.fp32_head`
                と対で使う)。経緯は forward のコメント参照。
            collect_stats:
                True なら毎 forward で z の統計(`last_stats`)を記録する。学習には影響しない
                (no_grad で計算するだけ)。lib/trainer/bt_telemetry.py が読む。
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed
        self.fp32 = fp32
        self.collect_stats = collect_stats
        self.last_stats = None

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # fp32=True のときは autocast を切ったスコープで計算する(DINOLoss.fp32 と同じ流儀)。
        #
        # 動機(2026-10-05, exp 0029 が ep65 で NaN になった件。**仮説であり未検証**):
        # この損失は各次元をバッチ方向に標準化する `(z - mean) / std`。BT の損失は各次元の
        # 平均(オフセット)に依存しないので、投影出力 z の各次元は大きな共通オフセットを持ちうる。
        # bf16 は仮数が 8bit しかなく、オフセット 300 の近傍では刻みが 2 になる。バッチ内の
        # 値が全て同じ bf16 値に丸められると std が厳密に 0 になり、`0/0` で損失が NaN になる。
        # この実装は std に eps を足さない(公式コードが BN の eps で有限のままかは未照合)。
        # autocast 下では Linear の出力 z がそもそも bf16 で返るので、損失側で .float() するだけ
        # では足りない。ヘッドの Linear も fp32 にする必要がある(BarlowTwins.fp32_head)。
        if not self.fp32:
            loss = self._compute(z_a, z_b)
        else:
            with torch.amp.autocast(device_type=z_a.device.type, enabled=False):
                loss = self._compute(z_a.float(), z_b.float())
        if self.grad_scale != 1.0:
            loss = _GradScale.apply(loss, self.grad_scale)
        return loss

    # 公式(facebookresearch/barlowtwins main.py)は損失の直前を `nn.BatchNorm1d(D, affine=False)` で標準化し、
    # モデル全体を SyncBN 化する = **global batch の平均・分散(biased)+ eps 1e-5**。
    BN_EPS = 1e-5

    def _standardize(self, z: torch.Tensor):
        """バッチ方向の標準化。分散時は全 rank の統計(global batch)で行い、勾配も通す(= SyncBN と同じ)。

        旧実装(〜2026-10-12)は各 GPU のローカルバッチ(256)で `(z - mean) / std` してから c を集約していた。
        これは global batch(2048)での標準化と一致せず(rank ごとに別のオフセット・スケールで割る)、
        std に eps も無かったので 1 次元でも値が一定になると 0/0 で NaN になった(レビュー指摘 2026-10-12)。
        統計は autocast に関係なく fp32 で取る(公式も AMP 下の BN は fp32)。
        """
        z = z.float()
        n_local = z.new_tensor(float(z.size(0)))
        s1, s2 = z.sum(0), (z * z).sum(0)
        if self.gather_distributed and dist.is_initialized() and dist.get_world_size() > 1:
            import torch.distributed.nn.functional as dist_fn   # autograd 対応の all_reduce(backward も集約)
            stats = dist_fn.all_reduce(torch.cat([s1, s2, n_local.view(1)]))
            s1, s2, n = stats[: z.size(1)], stats[z.size(1): 2 * z.size(1)], stats[-1]
        else:
            n = n_local
        mean = s1 / n
        var = (s2 / n - mean * mean).clamp_min(0)   # biased(BN と同じ)
        # 割るのは eps 込み。統計(last_stats)には eps 無しの std を返す(std==0 の次元を数えられるように)
        return (z - mean) / (var + self.BN_EPS).sqrt(), mean, var.sqrt()

    def _compute(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension (global batch, BN 相当)
        z_a_norm, mean_a, std_a = self._standardize(z_a)
        z_b_norm, mean_b, std_b = self._standardize(z_b)
        if not self.fp32:
            # 相関行列の matmul は従来どおり autocast(bf16)に任せる(公式も AMP 下の matmul は半精度)
            z_a_norm, z_b_norm = z_a_norm.to(z_a.dtype), z_b_norm.to(z_b.dtype)
        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD

        # sum cross-correlation matrix between multiple gpus
        # (Goal.yaml 2026-08-03 / docs/multi_gpu_migration.md §2: マルチGPU時に
        # 有効化しないと各GPUのローカルバッチだけでcross-correlationを計算してしまい
        # 実効バッチサイズが増えない。gather_distributed=Falseの単一GPU実行では
        # 従来通り no-op。)
        if self.gather_distributed and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                c = c / world_size
                dist.all_reduce(c)

        # loss — computed without materializing a DxD identity/mask (the old code
        # built torch.eye(D) on CPU every step; at the paper's D=8192 that is a
        # 67M-element CPU op that saturates all cores). (c - I)^2 decomposes into
        # diagonal (c_ii - 1)^2 and off-diagonal c_ij^2, so use pure GPU reductions.
        diag = torch.diagonal(c)
        on_diag = (diag - 1).pow(2).sum()
        off_diag = c.pow(2).sum() - diag.pow(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        self.last_parts = torch.stack([on_diag.detach().float(), off_diag.detach().float()])

        if self.collect_stats:
            self._record_stats(std_a, std_b, mean_a, mean_b, diag, on_diag, off_diag)

        return loss

    @torch.no_grad()
    def _record_stats(self, std_a, std_b, mean_a, mean_b, diag, on_diag, off_diag):
        """z のバッチ統計を `last_stats`(float32, 長さ7)に残す。NaN の原因切り分け用。

        順に: z_std_min / n_zero_std / z_mean_abs_max / z_ratio_max / c_diag_mean / on_diag / off_diag。
        std・mean は標準化に実際に使った値(2026-10-12 以降は global batch・eps 込み。それ以前のログはローカルバッチ・eps なし)。
        z_ratio_max = max |mean| / std。bf16 が刻みを落とす目安(オフセット/ばらつき比)で、
        std==0 の次元があると極端に大きな値になる。std はこのクラスが実際に使った精度で
        計算した値をそのまま float 化して見る(bf16 経路なら bf16 の std)。
        """
        std = torch.cat([std_a, std_b]).float()
        mean_abs = torch.cat([mean_a, mean_b]).float().abs()
        tiny = torch.finfo(torch.float32).tiny
        self.last_stats = torch.stack([
            std.min(),
            (std == 0).sum().float(),
            mean_abs.max(),
            (mean_abs / std.clamp_min(tiny)).max(),
            diag.float().mean(),
            on_diag.float(),
            off_diag.float(),
        ])

class BarlowTwins(nn.Module):
    def __init__(self, backbone, head_size=[2048, 512, 128], fp32_head=False):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(*head_size)
        # fp32_head=True なら、backbone は autocast(bf16)のまま、投影ヘッド(Linear×3 と BN)
        # だけ fp32 で回す。BarlowTwinsLoss(fp32=True) と対で使う。動機は
        # BarlowTwinsLoss.forward のコメント参照(exp 0029 の NaN の仮説)。
        # 演算量の大半は backbone で、ヘッドは約0.8%(fwd+bwd の演算量比: 6x140M params 対
        # 6x86M params x 197 tokens)。ただし fp32 は bf16 より遅い(GH200 で10倍強)ので、
        # 速度低下は数%〜10%程度の見込み(**未測定**。投入後の epoch 時間で確かめること)。
        # state_dict の形は変わらないので、fp32_head の有無でチェックポイントは互換。
        self.fp32_head = fp32_head

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        if self.fp32_head:
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                return self.projection_head(x.float())
        z = self.projection_head(x)
        return z

