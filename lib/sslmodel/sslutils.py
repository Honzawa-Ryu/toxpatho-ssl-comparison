# -*- coding: utf-8 -*-
"""
# SSL Method

@author: Katsuhisa MORITA
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

import lib.sslmodel as sslmodel
from lib.sslmodel.models import barlowtwins, simsiam, byol, swav, linearhead, simclr, mae, dino
from lib.trainer import distributed

class BarlowTwins:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512, pred_dim=128, projection_dim=512):
        model = barlowtwins.BarlowTwins(backbone, head_size=[head_size, projection_dim, pred_dim])
        # マルチGPU時はcross-correlation行列をGPU間でall_reduceする(有効バッチサイズを
        # 正しく増やすために必須。Goal.yaml 2026-08-03)。単一GPUではFalseのまま=従来通り。
        criterion = barlowtwins.BarlowTwinsLoss(gather_distributed=distributed.world_size() > 1)
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512, pred_dim=128, projection_dim=512):
        model = barlowtwins.BarlowTwins(backbone, head_size=[head_size, projection_dim, pred_dim])
        model.load_state_dict(torch.load(model_path))
        model = model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # put data on GPU
        z1, z2 = model(x1), model(x2)
        loss = criterion(z1, z2)
        return loss

class Byol:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512, projection_hidden_size:int=2048):
        """return ssl model"""
        model = byol.BYOL(
            backbone, 
            image_size=224, 
            hidden_layer=-1, 
            projection_size = 256, 
            projection_hidden_size = projection_hidden_size, 
            moving_average_decay = 0.99,
            DEVICE=self.DEVICE,
            )
        criterion = byol.loss_fn
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512, projection_hidden_size:int=2048):
        """return backbone model"""
        model = byol.BYOL(
            backbone, image_size=224, 
            hidden_layer=-1, 
            projection_size = 256, projection_hidden_size = projection_hidden_size,
            moving_average_decay = 0.99,
            DEVICE=self.DEVICE,
            )
        model.load_state_dict(torch.load(model_path))
        model = model.online_encoder.net
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # put data on GPU
        online_pred_one, online_pred_two, target_proj_one, target_proj_two = model(x1=x1, x2=x2) # forward
        loss = (criterion(online_pred_one, target_proj_two) + criterion(online_pred_two, target_proj_one)).mean() * 0.5 # loss
        return loss

class SwaV:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512, n_prototypes:int=512):
        """return ssl model"""
        model = swav.SwaV(
            backbone, head_size=[head_size, 512, 128],
            n_prototypes=n_prototypes,
        )
        # マルチGPU時はSinkhorn-KnoppをGPU間でall_reduce/all_gatherする
        # (プロトタイプ割当がバッチ全体の分布に依存するため。Goal.yaml 2026-08-03)。
        # 単一GPUではFalseのまま=従来通り。
        criterion = swav.SwaVLoss(sinkhorn_gather_distributed=distributed.world_size() > 1)
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        if getattr(self, "n_local_crops", 0) > 0:
            # multi-crop: n_global x 224 + n_local x local_crop_size (paper: 2x224 + 6x96)
            return sslmodel.utils.multicrop_transform(
                n_global=self.n_global_crops, n_local=self.n_local_crops,
                local_size=getattr(self, "local_crop_size", 96),
                color_plob=color_plob, blur_plob=blur_plob, solar_plob=solar_plob,
            )
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob,
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512, n_prototypes:int=512):
        """return backbone model"""
        model = swav.SwaV(
            backbone, head_size=[head_size, 512, 128],
            n_prototypes=n_prototypes,
        )
        model.load_state_dict(torch.load(model_path))
        model = model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        model.prototypes.normalize()
        high_res_batch = torch.cat([x.to(self.DEVICE) for x in data[:2]], dim=0)
        high_res_out = model(high_res_batch)
        high_resolution_crops = list(high_res_out.chunk(2, dim=0))

        if len(data) > 2:
            low_res_batch = torch.cat([x.to(self.DEVICE) for x in data[2:]], dim=0)
            low_res_out = model(low_res_batch)
            low_resolution_crops = list(low_res_out.chunk(len(data) - 2, dim=0))
        else:
            low_resolution_crops = []

        loss = criterion(high_resolution_crops, low_resolution_crops)
        return loss

class SimSiam:        
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512, dim=2048, pred_dim=512,):
        """return ssl model"""
        model= simsiam.SimSiam(
            backbone,
            head_size=head_size,
            dim=dim,
            pred_dim=pred_dim,)
        criterion = simsiam.NegativeCosineSimilarity()
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512, dim=2048, pred_dim=512,):
        """return backbone model"""
        model= simsiam.SimSiam(
            backbone,
            head_size=head_size,
            dim=dim,
            pred_dim=pred_dim,)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        model=model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # put data on GPU
        p1, p2, z1, z2 = model(x1=x1, x2=x2) # forward
        loss = 0.5 * (criterion(z1, p2) + criterion(z2, p1))
        return loss

class WSL:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512, num_classes=8):
        """return num_classifier model"""
        model= linearhead.LinearHead(backbone, dim=head_size, num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=False, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512, num_classes=8):
        """return backbone model"""
        model= linearhead.LinearHead(backbone, dim=head_size, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model = model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, label, criterion):
        data, label = data.to(self.DEVICE), label.to(self.DEVICE) # put data on GPU
        output = model(data)
        loss = criterion(output, label)
        return loss

# Test Class
class BarlowTwinsWS(BarlowTwins):
    def __init__(self, DEVICE="cpu"):
        super().__init__(DEVICE=DEVICE)
        self.DEVICE=DEVICE

    def prepare_transform(
        self,
        color_plob=None,
        blur_plob=None,
        solar_plob=None,
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.weak_strong_transform()
        return train_transform


class SimCLR:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE = DEVICE

    def prepare_model(self, backbone, head_size: int = 512, projection_dim: int = 128):
        model = simclr.SimCLR(backbone, head_size=[head_size, head_size, projection_dim])
        criterion = simclr.NTXentLoss(temperature=0.5)
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(self, color_plob=0.8, blur_plob=0.2, solar_plob=0.):
        return sslmodel.utils.ssl_transform(
            color_plob=color_plob, blur_plob=blur_plob, solar_plob=solar_plob,
            split=True, multi=False,
        )

    def prepare_featurize_model(self, backbone, model_path: str = "", head_size: int = 512, projection_dim: int = 128):
        model = simclr.SimCLR(backbone, head_size=[head_size, head_size, projection_dim])
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
        z1, z2 = model(x1), model(x2)
        return criterion(z1, z2)


class MAE:
    """Masked Autoencoder (non-contrastive, generative SSL).

    ViT-B/16 encoder + lightweight ViT decoder built internally on timm blocks.
    The `backbone` argument (torchvision) is ignored; kept for interface parity.
    Loss is computed inside the model, so `criterion` is None.
    """

    def __init__(self, DEVICE="cpu"):
        self.DEVICE = DEVICE

    def prepare_model(self, backbone=None, head_size: int = 768,
                      mask_ratio: float = 0.75, norm_pix_loss: bool = True):
        # head_size carries the encoder width from DICT_MODEL: 768=ViT-B, 1024=ViT-L.
        factory = mae.mae_vit_large_patch16 if head_size >= 1024 else mae.mae_vit_base_patch16
        model = factory(mask_ratio=mask_ratio, norm_pix_loss=norm_pix_loss)
        criterion = None  # reconstruction loss is computed inside the model
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(self, color_plob=0.8, blur_plob=0.2, solar_plob=0.):
        # MAE learns from a single (masked) view. Use minimal photometric aug;
        # RandomResizedCrop + flip are the standard MAE augmentations.
        return sslmodel.utils.ssl_transform(
            color_plob=color_plob, blur_plob=blur_plob, solar_plob=solar_plob,
            split=False, multi=False,
        )

    def prepare_featurize_model(self, backbone=None, model_path: str = "",
                                head_size: int = 768, mask_ratio: float = 0.75,
                                norm_pix_loss: bool = True):
        factory = mae.mae_vit_large_patch16 if head_size >= 1024 else mae.mae_vit_base_patch16
        model = factory(mask_ratio=mask_ratio, norm_pix_loss=norm_pix_loss)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        featurizer = mae.MAEFeaturizer(model)
        featurizer.to(self.DEVICE)
        return featurizer

    def calc_loss(self, model, data, criterion=None):
        x = data[0] if isinstance(data, (list, tuple)) else data
        x = x.to(self.DEVICE)
        loss, _, _ = model(x)
        return loss


class DINO:
    """DINO self-distillation (non-contrastive).

    Student/teacher timm ViT-B/16 with an EMA teacher. Default 2 global crops at
    224px (no local crops) for a fair comparison with SimSiam / Barlow Twins.
    The `backbone` argument (torchvision) is ignored; kept for interface parity.
    The training loop calls `model.update_moving_average()` after each step.
    """

    def __init__(self, DEVICE="cpu"):
        self.DEVICE = DEVICE
        self.out_dim = 8192
        self.n_global_crops = 2
        self.n_local_crops = 0

    def prepare_model(self, backbone=None, head_size: int = 768,
                      out_dim: int = 8192, momentum: float = 0.9995,
                      teacher_temp: float = 0.04, student_temp: float = 0.1,
                      freeze_last_layer_epochs: int = 1,
                      momentum_end=None, teacher_temp_end=None,
                      teacher_temp_warmup_epochs: int = 30,
                      drop_path_rate: float = 0.0):
        # 既定値は0017と同じ「据え置き運用」で、公式(main_dino.py の vit_base 既定)とは
        # out_dim(公式65536) と momentum(公式0.996 cosine->1.0) が異なる。
        # 論文準拠で走らせる場合は --dino_out_dim 65536 / --dino_momentum_start 0.996
        # --dino_momentum_end 1.0 を明示すること(0025以降はそうしている)。
        # 注: 「out_dim が小さい方が崩壊しにくい」は誤りで、公式は逆に大きい値を使う
        # (プロトタイプ数が多いほどサンプルを散らせるので一様解に落ちにくい)。
        # momentum 0.9995 も公式ヘルプでは batch 256 向けの推奨値であり、
        # global batch 1024 の本プロジェクトでは 0.996 が論文設定にあたる。
        # momentum_end / teacher_temp_end は既定 None = 固定運用(0017と同一)。
        # 指定すると論文(Caron et al. 2021)のcosine/linearスケジュールが有効になる
        # (どちらか片方だけ有効にできるので、崩壊要因の切り分けに使える)。
        self.out_dim = out_dim
        model = dino.DINO(
            backbone_name="vit_base_patch16_224", out_dim=out_dim,
            momentum=momentum, n_global_crops=self.n_global_crops,
            n_local_crops=self.n_local_crops,
            freeze_last_layer_epochs=freeze_last_layer_epochs,
            momentum_end=momentum_end, drop_path_rate=drop_path_rate)
        criterion = dino.DINOLoss(
            out_dim=out_dim, teacher_temp=teacher_temp, student_temp=student_temp,
            teacher_temp_end=teacher_temp_end,
            teacher_temp_warmup_epochs=teacher_temp_warmup_epochs)
        criterion.to(self.DEVICE)
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(self, color_plob=0.8, blur_plob=0.2, solar_plob=0.):
        if getattr(self, "n_local_crops", 0) > 0:
            # multi-crop: n_global x 224 + n_local x local_crop_size (paper: 2x224 + 8x96)
            return sslmodel.utils.multicrop_transform(
                n_global=self.n_global_crops, n_local=self.n_local_crops,
                local_size=getattr(self, "local_crop_size", 96),
                color_plob=color_plob, blur_plob=blur_plob, solar_plob=solar_plob,
            )
        # 2 global crops (224px) -> [x1, x2], same view setup as SimSiam / BT.
        return sslmodel.utils.ssl_transform(
            color_plob=color_plob, blur_plob=blur_plob, solar_plob=solar_plob,
            split=True, multi=False,
        )

    def prepare_featurize_model(self, backbone=None, model_path: str = "",
                                head_size: int = 768, out_dim: int = 8192):
        state = None
        if model_path:
            state = torch.load(model_path, map_location=self.DEVICE)
            # out_dim はランごとに異なりうる(0024以前は8192、0025以降は公式の65536)。
            # 呼び出し側(lib/model/zoo.py)は out_dim を渡さないので、チェックポイントの
            # 形状から復元する。ここを固定値にすると 65536 のランを読めない。
            key = 'student_head.last_layer.weight_v'
            if key in state:
                out_dim = state[key].shape[0]
        model = dino.DINO(
            backbone_name="vit_base_patch16_224", out_dim=out_dim,
            n_global_crops=self.n_global_crops, n_local_crops=self.n_local_crops)
        if state is not None:
            model.load_state_dict(state)
        featurizer = model.student_backbone  # timm ViT, num_classes=0 -> pooled features
        featurizer.to(self.DEVICE)
        return featurizer

    def calc_loss(self, model, data, criterion):
        views = [v.to(self.DEVICE) for v in data]
        global_views = views[:self.n_global_crops]
        # student側は勾配が必要なので model(views) で呼ぶ(DDPラップ時は
        # DistributedDataParallel.forward()経由になり、backward時の勾配同期
        # フックが正しく起動する。DINO.forward()はlist/tupleを渡すと
        # forward_studentへdispatchする実装にしてある)。
        # DDPは任意属性を .module へ転送しないため、ラップ後のmodelに対して
        # model.forward_student(...) と直接呼ぶと AttributeError で即死する
        # (2026-08-10、0017の4ノードジョブ(2513399)で実機確認:
        #  "AttributeError: 'DistributedDataParallel' object has no attribute
        #  'forward_student'" が全rankで発生しepoch 1に到達せず終了)。
        # また仮に unwrap() 経由で forward_student を直接呼ぶと、今度はDDPの
        # 勾配同期フックをバイパスして各rankが同期されないまま独立に学習する
        # 静かなバグになるため、student側は必ず model(views) で呼ぶこと。
        # teacher側は@torch.no_grad()で勾配を持たずDDP同期が不要なため、
        # unwrap()した生モジュールを呼ぶ(cancel_last_layer_gradients/
        # update_moving_averageと同じパターン。単一プロセス実行では
        # unwrap()はno-opでmodelをそのまま返す)。
        student_out = model(views)
        teacher_out = distributed.unwrap(model).forward_teacher(global_views)
        loss = criterion(student_out, teacher_out)
        return loss
