# -*- coding: utf-8 -*-
"""
# byol module

reference: https://github.com/facebookresearch/simsiam

@author: Katsuhisa MORITA
"""
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

class SimSiam(nn.Module):
    """
    SimSiam model.
    """
    def __init__(self, backbone, head_size=512, dim=2048, pred_dim=512):
        """
        head_size: last layer dimension
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # set the backbone encoder
        self.backbone = backbone

        # build a 3-layer projector
        self.fc = nn.Sequential(nn.Linear(head_size, head_size, bias=False),
                                nn.BatchNorm1d(head_size),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(head_size, head_size, bias=False),
                                nn.BatchNorm1d(head_size),
                                nn.ReLU(inplace=True), # second layer
                                nn.Linear(head_size, dim),
                                nn.BatchNorm1d(dim, affine=False)) # output layer
        self.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        z1 = torch.flatten(z1, start_dim=1) #flatten
        z2 = torch.flatten(z2, start_dim=1) #flatten
        z1 = self.fc(z1) # NxC
        z2 = self.fc(z2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def project_single(self, x: torch.Tensor) -> torch.Tensor:
        """Single-view projection (backbone + projector) for evaluation."""
        z = self.backbone(x)
        z = torch.flatten(z, start_dim=1)
        return self.fc(z)

class NegativeCosineSimilarity(torch.nn.Module):
    """Implementation of the Negative Cosine Simililarity"""
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Same parameters as in torch.nn.CosineSimilarity
        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return -cosine_similarity(x0, x1, self.dim, self.eps).mean()