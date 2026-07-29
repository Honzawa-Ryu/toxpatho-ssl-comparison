# -*- coding: utf-8 -*-
"""
# DINO module

Self-contained implementation following DINO
(Caron et al., 2021, https://arxiv.org/abs/2104.14294,
 https://github.com/facebookresearch/dino) built on a timm ViT backbone.

Student/teacher self-distillation with a momentum (EMA) teacher, centering +
sharpening of teacher outputs, and a weight-normed projection head.

For fair comparison with SimSiam / Barlow Twins in this project, the default
setup uses 2 global crops at 224px (no local crops). Local multi-crop can be
enabled via `n_local_crops` if desired.

@author: wsi-ad
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


class DINOHead(nn.Module):
    """3-layer MLP projection head with a weight-normalized last layer."""

    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048,
                 bottleneck_dim=256, nlayers=3, use_bn=False, norm_last_layer=True):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """Cross-entropy between centered+sharpened teacher and student outputs."""

    def __init__(self, out_dim=65536, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_outputs, teacher_outputs):
        """student_outputs / teacher_outputs: list of tensors (one per view)."""
        student = [s / self.student_temp for s in student_outputs]
        teacher = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                   for t in teacher_outputs]

        total_loss = 0.0
        n_terms = 0
        for iq, tq in enumerate(teacher):
            for iv, sv in enumerate(student):
                if iv == iq:
                    # skip same view (teacher & student on identical crop)
                    continue
                loss = torch.sum(-tq * F.log_softmax(sv, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_terms += 1
        total_loss /= max(n_terms, 1)
        self.update_center(torch.cat(teacher_outputs, dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(nn.Module):
    """DINO student/teacher wrapper around a timm ViT backbone."""

    def __init__(self, backbone_name="vit_base_patch16_224", out_dim=8192,
                 momentum=0.9995, norm_last_layer=True, n_global_crops=2,
                 n_local_crops=0, freeze_last_layer_epochs=1):
        super().__init__()
        self.momentum = momentum
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.freeze_last_layer_epochs = freeze_last_layer_epochs

        student_backbone = timm.create_model(
            backbone_name, num_classes=0, dynamic_img_size=True)
        embed_dim = student_backbone.num_features

        self.student_backbone = student_backbone
        self.student_head = DINOHead(embed_dim, out_dim, norm_last_layer=norm_last_layer)
        # weight_norm on the DINO head breaks copy.deepcopy, so build the teacher
        # fresh and sync it from the student via state_dict.
        self.teacher_backbone = timm.create_model(
            backbone_name, num_classes=0, dynamic_img_size=True)
        self.teacher_head = DINOHead(embed_dim, out_dim, norm_last_layer=norm_last_layer)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # expose backbone for the featurize / eval path
        self.backbone = self.student_backbone

    def cancel_last_layer_gradients(self, epoch):
        """DINO anti-collapse: freeze the projection head's last layer for the
        first `freeze_last_layer_epochs` epochs so the backbone stabilizes before
        the classifier can push outputs to the trivial (uniform) solution."""
        if epoch >= self.freeze_last_layer_epochs:
            return
        for p in self.student_head.last_layer.parameters():
            p.grad = None

    @torch.no_grad()
    def update_moving_average(self):
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data = pt.data * self.momentum + ps.data * (1 - self.momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data = pt.data * self.momentum + ps.data * (1 - self.momentum)

    def _forward_views(self, backbone, head, views):
        """Group same-resolution views into one batch, forward, split back."""
        outs = []
        start = 0
        while start < len(views):
            end = start + 1
            while end < len(views) and views[end].shape[-1] == views[start].shape[-1]:
                end += 1
            batch = torch.cat(views[start:end], dim=0)
            feat = backbone(batch)
            out = head(feat)
            outs.extend(torch.chunk(out, end - start, dim=0))
            start = end
        return outs

    def forward(self, x):
        """Single-view student projection (used by SSLEvaluator / featurize)."""
        return self.student_head(self.student_backbone(x))

    def forward_student(self, views):
        return self._forward_views(self.student_backbone, self.student_head, views)

    @torch.no_grad()
    def forward_teacher(self, global_views):
        return self._forward_views(self.teacher_backbone, self.teacher_head, global_views)
