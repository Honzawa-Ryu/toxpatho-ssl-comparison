# -*- coding: utf-8 -*-
"""
SSL evaluation metrics: effective rank, feature std, alignment, uniformity.
"""

import numpy as np
import torch
import torch.nn.functional as F


class SSLEvaluator:
    """
    Computes SSL evaluation metrics from backbone and projection embeddings.

    Metrics:
        effective_rank / normalized_effective_rank: representation diversity (Roy & Vetterli, 2007)
        feature_dim_std: mean per-dimension std; near zero indicates dimensional collapse
        alignment: mean squared distance between l2-normalized positive pairs (Wang & Isola, 2020)
        uniformity: log mean pairwise Gaussian kernel on unit sphere (Wang & Isola, 2020)
    """

    def __init__(self, samples_n: int = 2048, uniformity_samples: int = 512):
        self.samples_n = samples_n
        # pdist is O(N^2); cap samples to keep memory reasonable
        self.uniformity_samples = uniformity_samples

    @torch.no_grad()
    def evaluate(self, model, dataloader, device) -> dict:
        """
        Args:
            model: SSL model with .backbone and forward() returning projection outputs
            dataloader: yields [x1, x2] augmented pairs
            device: torch device
        Returns:
            dict of metric name -> float value
        """
        model.eval()
        backbone_embeds = []
        proj1_list = []
        proj2_list = []
        curr_samples = 0

        for data in dataloader:
            # Single-view SSL (e.g. MAE) yields one tensor per sample; contrastive
            # methods yield a [x1, x2] pair. Support both: eff_rank / feat_std need
            # only x1, and alignment / uniformity are computed only when a pair exists.
            if isinstance(data, (list, tuple)):
                x1 = data[0].to(device)
                x2 = data[1].to(device) if len(data) >= 2 else None
            else:
                x1, x2 = data.to(device), None

            z_b = self._backbone_embed(model, x1)
            if z_b.dim() > 2:
                z_b = z_b.flatten(start_dim=1)
            backbone_embeds.append(z_b.cpu())

            if x2 is not None:
                # SimSiam/BYOL forward needs two views; fall back to project_single if available
                if hasattr(model, 'project_single'):
                    proj1_list.append(model.project_single(x1).cpu())
                    proj2_list.append(model.project_single(x2).cpu())
                else:
                    proj1_list.append(model(x1).cpu())
                    proj2_list.append(model(x2).cpu())

            curr_samples += x1.size(0)
            if curr_samples >= self.samples_n:
                break

        if not backbone_embeds:
            return {}

        Z = torch.cat(backbone_embeds)[: self.samples_n]
        metrics = {}

        eff_rank, norm_eff_rank, _ = self._effective_rank(Z)
        metrics["effective_rank"] = eff_rank
        metrics["normalized_effective_rank"] = norm_eff_rank
        metrics["feature_dim_std"] = Z.std(dim=0).mean().item()

        if proj1_list:
            Z_p1 = torch.cat(proj1_list)[: self.samples_n]
            Z_p2 = torch.cat(proj2_list)[: self.samples_n]
            metrics["alignment"] = self._alignment(Z_p1, Z_p2)
            Z_uniform = torch.cat([Z_p1, Z_p2])[: self.uniformity_samples]
            metrics["uniformity"] = self._uniformity(Z_uniform)

        return metrics

    def _backbone_embed(self, model, x: torch.Tensor) -> torch.Tensor:
        """Backbone embedding, tolerant of models without a `.backbone`.

        Contrastive/distillation wrappers expose `.backbone`; MAE has no such
        attribute and instead provides `forward_features` (full unmasked encoder,
        mean-pooled patch tokens).
        """
        if hasattr(model, 'backbone'):
            return model.backbone(x)
        if hasattr(model, 'forward_features'):
            return model.forward_features(x)
        return model(x)

    def _effective_rank(self, Z: torch.Tensor):
        d = Z.size(1)
        Z_c = Z - Z.mean(dim=0)
        _, S, _ = torch.linalg.svd(Z_c, full_matrices=False)
        p = (S + 1e-10) / (S + 1e-10).sum()
        entropy = -(p * p.log()).sum()
        eff_rank = entropy.exp()
        return eff_rank.item(), (eff_rank / d).item(), S.numpy()

    def _alignment(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        z1_n = F.normalize(z1, dim=1)
        z2_n = F.normalize(z2, dim=1)
        return (z1_n - z2_n).norm(dim=1).pow(2).mean().item()

    def _uniformity(self, z: torch.Tensor) -> float:
        z_n = F.normalize(z, dim=1)
        return torch.pdist(z_n).pow(2).mul(-2).exp().mean().log().item()
