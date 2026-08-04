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

    def __init__(
        self, 
        lambda_param: float = 5e-3, 
        gather_distributed : bool = False
    ):
        """Lambda param configuration with default value like in [0]
        Args:
            lambda_param: 
                Parameter for importance of redundancy reduction term. 
                Defaults to 5e-3 [0].
            gather_distributed:
                If True then the cross-correlation matrices from all gpus are 
                gathered and summed before the loss calculation.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.gather_distributed = gather_distributed

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:

        device = z_a.device

        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
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

        return loss

class BarlowTwins(nn.Module):
    def __init__(self, backbone, head_size=[2048, 512, 128]):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(*head_size)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

