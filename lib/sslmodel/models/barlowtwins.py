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
            return self._compute(z_a, z_b)
        with torch.amp.autocast(device_type=z_a.device.type, enabled=False):
            return self._compute(z_a.float(), z_b.float())

    def _compute(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension
        mean_a, std_a = z_a.mean(0), z_a.std(0)
        mean_b, std_b = z_b.mean(0), z_b.std(0)
        z_a_norm = (z_a - mean_a) / std_a # NxD
        z_b_norm = (z_b - mean_b) / std_b # NxD
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

        if self.collect_stats:
            self._record_stats(std_a, std_b, mean_a, mean_b, diag, on_diag, off_diag)

        return loss

    @torch.no_grad()
    def _record_stats(self, std_a, std_b, mean_a, mean_b, diag, on_diag, off_diag):
        """z のバッチ統計を `last_stats`(float32, 長さ7)に残す。NaN の原因切り分け用。

        順に: z_std_min / n_zero_std / z_mean_abs_max / z_ratio_max / c_diag_mean / on_diag / off_diag。
        std・mean は**各 rank のローカルバッチ**の値(c は rank 間で集約済みなので後半3つは全体)。
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

