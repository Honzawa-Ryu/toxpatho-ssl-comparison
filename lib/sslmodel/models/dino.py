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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
                 center_momentum=0.9, teacher_temp_end=None,
                 teacher_temp_warmup_epochs=30):
        """teacher_temp_end=None なら teacher_temp 固定(0017の運用、既定)。

        teacher_temp_end を指定すると論文(Caron et al. 2021)の線形warmup
        (teacher_temp -> teacher_temp_end を warmup_epochs かけて、以降は
        teacher_temp_end 固定)が有効になる。低いteacher温度=強いsharpening
        は崩壊回避方向に働くため、0.04固定は意図的なanti-collapse設定
        (0021でこれを論文値に寄せた際にepoch15で恒久崩壊した経緯があるため、
        既定はあくまで固定運用のままにしてopt-inにしてある)。
        """
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_start = teacher_temp
        self.teacher_temp = teacher_temp
        self.teacher_temp_end = teacher_temp_end
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.teacher_temp_warmup_steps = 0
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def set_schedule(self, niter_per_ep: int):
        """niter_per_ep(1epochあたりのstep数)が判明してから呼ぶ。

        モデル構築時点ではデータローダー未生成でniter_per_epが不明なため、
        epoch単位で受け取ったwarmup長をここでstep数へ変換する。
        """
        self.teacher_temp_warmup_steps = self.teacher_temp_warmup_epochs * niter_per_ep

    def update_teacher_temp(self, step: int):
        """global step(resumeを跨いだ通算step)から現在のteacher_tempを設定する。

        stepの純関数として毎回再計算するため、resume時にself.teacher_temp自体を
        永続化・復元する必要がない(warmup途中で再開しても正しい値になる)。
        teacher_temp_end=None(既定)なら何もしない。
        """
        if self.teacher_temp_end is None:
            return
        if self.teacher_temp_warmup_steps <= 0:
            self.teacher_temp = self.teacher_temp_end
            return
        progress = min(1.0, step / self.teacher_temp_warmup_steps)
        self.teacher_temp = self.teacher_temp_start + \
            (self.teacher_temp_end - self.teacher_temp_start) * progress

    def forward(self, student_outputs, teacher_outputs):
        """student_outputs / teacher_outputs: list of tensors (one per view)."""
        # log_softmax は student ビューごとに1回だけ計算して使い回す。
        # 以前は teacher ビューごとにループ内で計算し直しており、同じテンソルを
        # 2回(= n_global_crops 回)作っていた。log_softmax は autocast下でも fp32 で
        # 保持され backward 用に保存されるため、out_dim=65536 では
        # 10ビュー x batch x 65536 x 4B = 約0.7GB/回 とメモリを直撃する。
        student_logsm = [F.log_softmax(s / self.student_temp, dim=-1)
                         for s in student_outputs]
        teacher = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                   for t in teacher_outputs]

        total_loss = 0.0
        n_terms = 0
        for iq, tq in enumerate(teacher):
            for iv, sv in enumerate(student_logsm):
                if iv == iq:
                    # skip same view (teacher & student on identical crop)
                    continue
                loss = torch.sum(-tq * sv, dim=-1)
                total_loss += loss.mean()
                n_terms += 1
        total_loss /= max(n_terms, 1)
        self.update_center(torch.cat(teacher_outputs, dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """teacher出力のcenterをバッチ平均で更新する。

        マルチGPU時、centerがローカルバッチだけで更新されるとGPU間でcenterが
        揃わず(=各GPUが異なるcollapse防止基準で動く)、崩壊防止の効果が薄れる
        （Goal.yaml 2026-08-03 / docs/multi_gpu_migration.md §2）。全GPUの
        teacher出力の合計をall_reduceしてから真のglobalバッチ平均を取る。
        単一プロセス実行（dist未初期化）では従来通りローカル平均のみ。
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size > 1:
                dist.all_reduce(batch_center)
                batch_center = batch_center / (teacher_output.shape[0] * world_size)
            else:
                batch_center = batch_center / teacher_output.shape[0]
        else:
            batch_center = batch_center / teacher_output.shape[0]
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(nn.Module):
    """DINO student/teacher wrapper around a timm ViT backbone."""

    def __init__(self, backbone_name="vit_base_patch16_224", out_dim=8192,
                 momentum=0.9995, norm_last_layer=True, n_global_crops=2,
                 n_local_crops=0, freeze_last_layer_epochs=1, momentum_end=None,
                 drop_path_rate=0.0):
        """momentum_end=None なら momentum 固定(0017の運用、既定)。

        momentum_end を指定すると論文のcosineスケジュール
        (momentum -> momentum_end)が有効になる。momentumが小さいほど
        teacherがstudentを速く追従する(0.996は0.9995の約12.5倍速い)ため、
        0.9995固定は意図的なanti-collapse設定。DINOLossのteacher温度と同様、
        既定は固定運用のままでopt-inにしてある。
        """
        super().__init__()
        self.momentum_start = momentum
        self.momentum = momentum
        self.momentum_end = momentum_end
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.freeze_last_layer_epochs = freeze_last_layer_epochs

        # stochastic depth は student だけに掛ける(公式 main_dino.py と同じ:
        # student は drop_path_rate=args.drop_path_rate、teacher は既定の0で構築する)。
        # teacherは勾配を持たずEMAで更新されるだけなので、確率的にブロックを落とす
        # 意味がない。drop_pathはパラメータを持たないため state_dict は一致する。
        student_backbone = timm.create_model(
            backbone_name, num_classes=0, dynamic_img_size=True,
            drop_path_rate=drop_path_rate)
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

    def update_teacher_momentum(self, step: int, total_steps: int):
        """global step(resumeを跨いだ通算step)から現在のmomentumをcosineで設定する。

        DINOLoss.update_teacher_tempと同様、stepの純関数として毎回再計算する
        のでresume安全(self.momentumはstate_dictに含まれない通常のfloat属性な
        ため、永続化に頼らずここで再計算する)。momentum_end=None(既定)なら
        何もしない。
        """
        if self.momentum_end is None:
            return
        progress = min(1.0, step / max(1, total_steps))
        self.momentum = self.momentum_end - 0.5 * (self.momentum_end - self.momentum_start) * \
            (1 + math.cos(math.pi * progress))

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
        """Single-view student projection (used by SSLEvaluator / featurize).

        Also accepts a list/tuple of views (multicrop) and dispatches to
        forward_student, so that DDP-wrapped calls (`ddp_model(views)`) go
        through DistributedDataParallel.forward() and get gradient-sync hooks
        armed for the backward pass (calling forward_student directly on the
        unwrapped .module bypasses those hooks -> silently un-synced grads).
        """
        if isinstance(x, (list, tuple)):
            return self.forward_student(x)
        return self.student_head(self.student_backbone(x))

    def forward_student(self, views):
        return self._forward_views(self.student_backbone, self.student_head, views)

    @torch.no_grad()
    def forward_teacher(self, global_views):
        return self._forward_views(self.teacher_backbone, self.teacher_head, global_views)
