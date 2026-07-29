# -*- coding: utf-8 -*-
"""
# モデル・損失・スケジューラの構築

分散学習に移行する際は、ここで DDP ラップを行う（REFACTOR_PLAN.md §6-4）。

"""
import torch.nn as nn
from timm.scheduler import CosineLRScheduler

import lib.sslmodel as sslmodel
import lib.model.zoo as zoo
from lib.trainer.context import RunContext
from lib.trainer.optim import build_optimizer


# prepare model
def prepare_model(
        ctx: RunContext,
        model_name:str='ResNet18', patience:int=7, delta:float=0, lr:float=0.003, weight_decay:float=0., num_epoch:int=150, freeze_backbone:bool=False,
        layer_wise_lr:bool=False, backbone_lr_ratio:float=0.1
    ):
    """
    preparation of models
    Parameters
    ----------
        ctx (RunContext)
            args / ssl_class / dir_name を保持する実行コンテキスト

        model_name (str)
            model architecture name

        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    args, ssl_class = ctx.args, ctx.ssl_class
    # model building with indicated name
    # mae/dino build their own timm ViT internally and ignore this backbone,
    # so avoid an (offline-unsafe) ImageNet weight download for them.
    _self_backbone = args.ssl_name in ("mae", "dino")
    _weights = None if (_self_backbone or model_name == "DenseNet121") else "DEFAULT"
    try:
        encoder = zoo.DICT_MODEL[model_name][0](weights=_weights)
        size=zoo.DICT_MODEL[model_name][1]
    except:
        print("indicated model name is not implemented")
        ValueError
    if "ViT" in model_name:
        # UNIFIED backbone: BT/SimSiam/SwAV use the SAME timm ViT-B/16 as DINO
        # (dynamic_img_size for pos-embed interpolation on SwAV's 96px local crops).
        # This makes the SSL-method comparison backbone-invariant, and gains
        # SDPA/flash-attention (torchvision ViT uses plain MHA -> no flash, slower).
        # MAE keeps its own (architecturally identical) masking encoder.
        if args.ssl_name in ("barlowtwins", "simsiam", "swav"):
            import timm
            backbone = timm.create_model("vit_base_patch16_224", num_classes=0, dynamic_img_size=True)
            print(f"{args.ssl_name}: unified timm ViT-B/16 (dynamic_img_size) backbone [same as DINO]")
        else:
            encoder.heads = nn.Identity() # ViTはheadがあるため、Identityに置き換える
            backbone = encoder
    elif model_name=="DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            )
    else:
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            )
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    # method-specific paper-faithful model knobs
    model_kwargs = {}
    if args.ssl_name == "swav":
        model_kwargs["n_prototypes"] = args.n_prototypes
    if args.ssl_name == "barlowtwins" and args.proj_dim > 0:
        model_kwargs["projection_dim"] = args.proj_dim
        model_kwargs["pred_dim"] = args.proj_dim
    model, criterion = ssl_class.prepare_model(backbone, head_size=size, **model_kwargs)
    # model.load_state_dict(torch.load(args.model_path))
    if args.optimizer == "lars" and args.lars_exclude_bias_bn:
        # Barlow Twins / SwAV: weights (LARS-adapted, weight-decayed) vs bias & BN
        # (ndim<=1: excluded from LARS adaptation and weight decay, own LR).
        weights, biases = [], []
        for p in model.parameters():
            if p.requires_grad:
                (biases if p.ndim <= 1 else weights).append(p)
        lr_bias = args.lr_bias if args.lr_bias > 0 else lr
        optimizer = build_optimizer(ctx, [
            {'params': weights},
            {'params': biases, 'lr': lr_bias, 'weight_decay': 0.0,
             'weight_decay_filter': True, 'lars_adaptation_filter': True},
        ], lr=lr, weight_decay=weight_decay)
        print(f"LARS param-groups: weights(lr={lr:.3g},wd={weight_decay:.1e}) | "
              f"bias/BN(lr={lr_bias:.3g},no-LARS,no-wd)")
    elif args.fix_pred_lr and hasattr(model, "predictor"):
        # SimSiam: predictor kept at a constant LR (not decayed by the scheduler).
        pred_ids = set(id(p) for p in model.predictor.parameters())
        pred_params, base_params = [], []
        for p in model.parameters():
            if p.requires_grad:
                (pred_params if id(p) in pred_ids else base_params).append(p)
        optimizer = build_optimizer(ctx, [
            {'params': base_params},
            {'params': pred_params, 'fix_lr': True},
        ], lr=lr, weight_decay=weight_decay)
        print(f"SimSiam fix-pred-lr: predictor held at constant lr={lr:.3g}")
    elif layer_wise_lr:
        backbone_param_ids = set(id(p) for p in model.backbone.parameters())
        backbone_params = []
        head_params = []
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in backbone_param_ids:
                    backbone_params.append(p)
                else:
                    head_params.append(p)
        optimizer = build_optimizer(ctx, [
            {'params': backbone_params, 'lr': lr * backbone_lr_ratio},
            {'params': head_params, 'lr': lr}
        ], lr=lr, weight_decay=weight_decay)
        print(f"Using layer-wise learning rate. Backbone LR: {lr * backbone_lr_ratio:.2e}, Head LR: {lr:.2e}")
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = build_optimizer(ctx, parameters, lr=lr, weight_decay=weight_decay)
    
    scheduler = CosineLRScheduler(
        optimizer, t_initial=num_epoch, lr_min=args.lr_min,
        warmup_t=args.warmup_t, warmup_lr_init=args.warmup_lr_init, warmup_prefix=True)
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{ctx.dir_name}/checkpoint.pt')
    return model, criterion, optimizer, scheduler, early_stopping
