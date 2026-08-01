# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

architectures

reference:
https://docs.lightly.ai/self-supervised-learning/examples/swav.html

@author: Katsuhisa
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional

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

class SwaVProjectionHead(ProjectionHead):
    """Projection head used for SwaV.
    [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    """
    def __init__(self,
                 input_dim: int = 2048,
                 hidden_dim: int = 2048,
                 output_dim: int = 128):
        super(SwaVProjectionHead, self).__init__([
            (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
            (hidden_dim, output_dim, None, None),
        ])


class SwaVPrototypes(nn.Module):
    """Multihead Prototypes used for SwaV.
    Each output feature is assigned to a prototype, SwaV solves the swapped
    predicition problem where the features of one augmentation are used to
    predict the assigned prototypes of the other augmentation.
    Examples:
        >>> # use features with 128 dimensions and 512 prototypes
        >>> prototypes = SwaVPrototypes(128, 512)
        >>>
        >>> # pass batch through backbone and projection head.
        >>> features = model(x)
        >>> features = nn.functional.normalize(features, dim=1, p=2)
        >>>
        >>> # logits has shape bsz x 512
        >>> logits = prototypes(features)
    """
    def __init__(self,
                input_dim: int = 128,
                n_prototypes: Union[List[int], int] = 3000):
        super(SwaVPrototypes, self).__init__()
        #Default to a list of 1 if n_prototypes is an int.
        self.n_prototypes = n_prototypes if isinstance(n_prototypes, list) else [n_prototypes]
        self._is_single_prototype = True if isinstance(n_prototypes, int) else False
        self.heads = nn.ModuleList([nn.Linear(input_dim, prototypes) for prototypes in self.n_prototypes])

    def forward(self, x) -> Union[torch.Tensor, List[torch.Tensor]]:
        out = []
        for layer in self.heads:
            out.append(layer(x))
        return out[0] if self._is_single_prototype else out
    
    def normalize(self):
        """Normalizes the prototypes so that they are on the unit sphere."""
        with torch.no_grad():
            for layer in self.heads:
                w = layer.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                layer.weight.copy_(w)

class SwaV(nn.Module):
    def __init__(self, backbone, head_size=[2048, 512, 128], n_prototypes=512):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(
            input_dim = head_size[0],
            hidden_dim = head_size[1],
            output_dim = head_size[2]
        )
        self.prototypes = SwaVPrototypes(head_size[-1], n_prototypes=n_prototypes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

# loss fn
@torch.no_grad()
def sinkhorn(
    out: torch.Tensor, 
    iterations: int = 3, 
    epsilon: float = 0.05,
    gather_distributed: bool = False,
) -> torch.Tensor:
    """Distributed sinkhorn algorithm.
    As outlined in [0] and implemented in [1].
    
    [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882
    [1]: https://github.com/facebookresearch/swav/ 
    Args:
        out:
            Similarity of the features and the SwaV prototypes.
        iterations:
            Number of sinkhorn iterations.
        epsilon:
            Temperature parameter.
        gather_distributed:
            If True then features from all gpus are gathered to calculate the
            soft codes Q. 
    Returns:
        Soft codes Q assigning each feature to a prototype.
    
    """
    world_size = 1
    #if gather_distributed and dist.is_initialized():
    #    world_size = dist.get_world_size()

    # get the exponential matrix and make it sum to 1
    Q = torch.exp(out / epsilon).t()
    sum_Q = torch.sum(Q)
    #if world_size > 1:
    #    dist.all_reduce(sum_Q)
    Q /= sum_Q

    B = Q.shape[1] * world_size

    for _ in range(iterations):
        # normalize rows
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #if world_size > 1:
        #    dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        # normalize columns
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


class SwaVLoss(nn.Module):
    """Implementation of the SwaV loss.
    Attributes:
        temperature:
            Temperature parameter used for cross entropy calculations.
        sinkhorn_iterations:
            Number of iterations of the sinkhorn algorithm.
        sinkhorn_epsilon:
            Temperature parameter used in the sinkhorn algorithm.
        sinkhorn_gather_distributed:
            If True then features from all gpus are gathered to calculate the
            soft codes in the sinkhorn algorithm. 
    
    """

    def __init__(self,
                 temperature: float = 0.1,
                 sinkhorn_iterations: int = 3,
                 sinkhorn_epsilon: float = 0.05,
                 sinkhorn_gather_distributed: bool = False):
        super(SwaVLoss, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_gather_distributed = sinkhorn_gather_distributed


    def subloss(self, z: torch.Tensor, q: torch.Tensor):
        """Calculates the cross entropy for the SwaV prediction problem.
        Args:
            z:
                Similarity of the features and the SwaV prototypes.
            q:
                Codes obtained from Sinkhorn iterations.
        Returns:
            Cross entropy between predictions z and codes q.
        """
        return - torch.mean(
            torch.sum(q * nn.functional.log_softmax(z / self.temperature, dim=1), dim=1)
        )


    def forward(self,
                high_resolution_outputs: List[torch.Tensor],
                low_resolution_outputs: List[torch.Tensor]):
        """Computes the SwaV loss for a set of high and low resolution outputs.
        Args:
            high_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                high resolution crops.
            low_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                low resolution crops.
        Returns:
            Swapping assignments between views loss (SwaV) as described in [0].
        [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882
        """
        n_crops = len(high_resolution_outputs) + len(low_resolution_outputs)

        # multi-crop iterations
        loss = 0.
        for i in range(len(high_resolution_outputs)):

            # compute codes of i-th high resolution crop
            with torch.no_grad():
                q = sinkhorn(
                    high_resolution_outputs[i].detach(),
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon,
                    gather_distributed=self.sinkhorn_gather_distributed,
                )

            # compute subloss for each pair of crops
            subloss = 0.
            for v in range(len(high_resolution_outputs)):
                if v != i:
                    subloss += self.subloss(high_resolution_outputs[v], q)

            for v in range(len(low_resolution_outputs)):
                subloss += self.subloss(low_resolution_outputs[v], q)

            loss += subloss / (n_crops - 1)

        return loss / len(high_resolution_outputs)