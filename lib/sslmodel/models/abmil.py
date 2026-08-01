# -*- coding: utf-8 -*-
"""
Gated Attention-Based MIL for WSI-level classification.
Reference: Ilse et al., 2018 (https://arxiv.org/abs/1802.04712)
"""

import torch
import torch.nn as nn


class ABMIL(nn.Module):
    """
    Gated attention MIL.
    Input:  x [N_patches, in_dim]
    Output: logit (scalar), attn [N_patches]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))

    def forward(self, x: torch.Tensor):
        v = self.attention_V(x)         # [N, hidden_dim]
        u = self.attention_U(x)         # [N, hidden_dim]
        a = self.attention_w(v * u)     # [N, 1]
        a = torch.softmax(a, dim=0)     # [N, 1]
        z = (a * x).sum(dim=0)          # [in_dim]
        logit = self.classifier(z)      # [1]
        return logit, a.squeeze(1)      # scalar, [N]
