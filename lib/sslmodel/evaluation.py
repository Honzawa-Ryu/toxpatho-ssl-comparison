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

        eff_rank, norm_eff_rank, S = self._effective_rank(Z)
        metrics["effective_rank"] = eff_rank
        # 部分的な次元崩壊の指標(2026-10-15、SimSiam 0038 の診断で使ったものを学習中にも出す)。
        #   n_comp_90: 分散の 90% を説明する主成分の数(0038 は 4〜7、DINO 0028 は 68〜371)
        #   pc1_ratio: 第 1 主成分の分散の割合(0038 は 0.39〜0.72、DINO は 0.04〜0.17)
        #   mean_cos:  別々の画像の特徴の cos の平均(0038 は 0.87〜0.98、DINO は 0.11〜0.83)
        metrics.update(self._dimensional_collapse(Z, S))
        metrics["normalized_effective_rank"] = norm_eff_rank
        metrics["feature_dim_std"] = Z.std(dim=0).mean().item()
        # ばらつき / 大きさ。完全崩壊で 0 に近づく(effective_rank が 0 になる判定と同じ量)
        Zf = Z.float()
        metrics["relative_spread"] = ((Zf - Zf.mean(0)).pow(2).mean().sqrt()
                                      / Zf.pow(2).mean().sqrt().clamp_min(1e-30)).item()

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

    # サンプル間のばらつき(中心化後の RMS)が特徴の大きさ(RMS)のこの比を下回ったら、特徴は定数に潰れている
    # とみなし実効ランク 0 を返す。健全な特徴の比は 0.1〜1 程度、fp32 の評価で入力に依らず同じ出力なら 1e-6 以下。
    COLLAPSE_SPREAD_RATIO = 1e-3

    @classmethod
    def _effective_rank(cls, Z: torch.Tensor):
        """実効ランク(Roy & Vetterli 2007: 正規化特異値のエントロピーの exp)。

        ⚠️ 完全崩壊(全サンプルが同じ特徴)では中心化後の特異値がすべて 0(か丸め誤差)になる。旧実装は
        各特異値に 1e-10 を足して正規化していたため、一様分布になって**実効ランクが最大の d(768)を返し**、
        崩壊を見逃していた(レビュー指摘 2026-10-12)。丸め誤差だけの特異値も同じく一様に近く大きな値になるので、
        ばらつきが特徴の大きさに比べて無視できるときは 0 を返す。
        """
        d = Z.size(1)
        Z = Z.float()
        Z_c = Z - Z.mean(dim=0)
        _, S, _ = torch.linalg.svd(Z_c, full_matrices=False)
        spread = Z_c.pow(2).mean().sqrt()
        scale = Z.pow(2).mean().sqrt()
        if not torch.isfinite(S).all():
            return float('nan'), float('nan'), S.numpy()
        if S.sum() <= 0 or spread <= cls.COLLAPSE_SPREAD_RATIO * scale:
            return 0.0, 0.0, S.numpy()
        p = S / S.sum()
        entropy = -(p * torch.log(p.clamp_min(1e-30))).sum()
        eff_rank = entropy.exp()
        return eff_rank.item(), (eff_rank / d).item(), S.numpy()

    @staticmethod
    def _dimensional_collapse(Z: torch.Tensor, S) -> dict:
        import numpy as np
        var = np.asarray(S, dtype=np.float64) ** 2
        if var.sum() <= 0 or not np.isfinite(var).all():
            return {"n_comp_90": float("nan"), "pc1_ratio": float("nan"), "mean_cos": float("nan")}
        frac = np.cumsum(var) / var.sum()
        zn = F.normalize(Z.float(), dim=1)
        n = zn.size(0)
        g = zn @ zn.T
        mean_cos = ((g.sum() - g.diagonal().sum()) / (n * (n - 1))).item() if n > 1 else float("nan")
        return {"n_comp_90": float(np.searchsorted(frac, 0.9) + 1), "pc1_ratio": float(var[0] / var.sum()),
                "mean_cos": mean_cos}

    def _alignment(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        z1_n = F.normalize(z1, dim=1)
        z2_n = F.normalize(z2, dim=1)
        return (z1_n - z2_n).norm(dim=1).pow(2).mean().item()

    def _uniformity(self, z: torch.Tensor) -> float:
        z_n = F.normalize(z, dim=1)
        return torch.pdist(z_n).pow(2).mul(-2).exp().mean().log().item()
