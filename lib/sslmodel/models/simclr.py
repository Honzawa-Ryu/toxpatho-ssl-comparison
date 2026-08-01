# -*- coding: utf-8 -*-
"""
SimCLR module

Reference:
    Chen et al. 2020, A Simple Framework for Contrastive Learning
    https://arxiv.org/abs/2002.05709

@author: Katsuhisa MORITA
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimCLR(nn.Module):
    def __init__(self, backbone: nn.Module, head_size: List[int] = [2048, 2048, 128]):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(*head_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def project_single(self, x: torch.Tensor) -> torch.Tensor:
        """Single-view projection for evaluation."""
        return self.forward(x)


class NTXentLoss(nn.Module):
    """NT-Xent (InfoNCE) loss from SimCLR.

    Temperature default 0.5 matches the SimCLR paper.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)          # (2N, D)
        z = F.normalize(z, dim=1)

        # Pairwise cosine similarity scaled by temperature
        sim = torch.mm(z, z.T) / self.temperature  # (2N, 2N)

        # Exclude self-similarities from the denominator
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])  # (2N,)

        return F.cross_entropy(sim, labels)
