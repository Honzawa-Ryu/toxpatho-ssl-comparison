# -*- coding: utf-8 -*-
"""
# model zoo

SSL手法比較パイプラインが共有するモデル定義とビルダー。
旧 `src/utils.py` のモデル部分を切り出したもの
（REFACTOR_PLAN.md §2-4 / §5-3 / Phase 1-b）。
WSI マスク処理は lib/wsi.py に分けた。

@author: Katsuhisa MORITA
"""
import re

import torch
import torch.nn as nn
import torchvision

import lib.sslmodel.sslutils as sslutils

## Model architecture
# name: [Model_Class, last_layer_size]
DICT_MODEL = {
    "EfficientNetB3": [torchvision.models.efficientnet_b3, 1536],
    "ViTB16": [torchvision.models.vit_b_16, 768],
    "ViTL16": [torchvision.models.vit_l_16, 1024],  # MAE-large: head_size=1024 selects the ViT-L encoder
    "ConvNextTiny": [torchvision.models.convnext_tiny, 768],
    "ResNet18": [torchvision.models.resnet18, 512],
    "RegNetY16gf": [torchvision.models.regnet_y_1_6gf, 888],
    "DenseNet121": [torchvision.models.densenet121, 1024],
    "ResNet50": [torchvision.models.resnet50, 2048],
}

## SSL method
DICT_SSL={
    "barlowtwins":sslutils.BarlowTwins,
    "swav":sslutils.SwaV,
    "byol":sslutils.Byol,
    "simsiam":sslutils.SimSiam,
    "simclr":sslutils.SimCLR,
    "mae":sslutils.MAE,
    "dino":sslutils.DINO,
    "wsl":sslutils.WSL,
}

## Pretrained pathology foundation models (external weights, no in-project SSL head).
# Each builder returns a timm model with the classification head removed
# (num_classes=0), so a plain forward yields the pooled patch embedding.
# Inputs use the same 224x224 / ImageNet mean-std transform as the SSL methods.
def _build_uni():
    import timm
    # UNI (MahmoodLab), ViT-L/16, 1024-d CLS embedding. Weights are pulled from the
    # local HF cache (models--MahmoodLab--uni); set HF_HUB_OFFLINE=1 to force offline.
    return timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True, init_values=1e-5, dynamic_img_size=True,
    )

DICT_FOUNDATION = {
    "UNI": _build_uni,
}

def prepare_foundation_eval(model_name:str="UNI", DEVICE="cpu"):
    """Build an eval-ready pretrained foundation encoder (forward -> pooled embedding)."""
    if model_name not in DICT_FOUNDATION:
        raise ValueError(f"unknown foundation model: {model_name} (have {list(DICT_FOUNDATION)})")
    model = DICT_FOUNDATION[model_name]()
    model.to(DEVICE)
    model.eval()
    return model

def _load_state_dict_dense(model, weights):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls (pretrained-models). This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(weights.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            weights[new_key] = weights[key]
            del weights[key]
    model.load_state_dict(weights)
    return model

def prepare_model_eval(
        model_name:str='ResNet18', ssl_name="barlowtwins",
        model_path="",
        pretrained=False,
        DEVICE="cpu"):
    """
    preparation of models
    Parameters
    ----------
        modelname (str)
            model architecture name

    """
    # model building with indicated name
    if pretrained:
        if model_name=="DenseNet121":
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder = _load_state_dict_dense(encoder, torch.load(model_path))
            model = nn.Sequential(
                *list(encoder.children())[:-1],
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
                )
        else:
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder.load_state_dict(torch.load(model_path))
            model=nn.Sequential(*list(encoder.children())[:-1])
    elif "ViT" in model_name and ssl_name in ("barlowtwins", "simsiam", "swav"):
        # Barlow Twins / SimSiam / SwAV were trained with a UNIFIED timm ViT-B/16
        # backbone (dynamic_img_size), and their projector heads differ from the eval
        # defaults (BT 8192, SwAV 3000 prototypes). For feature extraction we only need
        # the backbone: rebuild the exact timm ViT and load ONLY the backbone.* weights
        # from the snapshot (model_ep*.pt or state.pt), avoiding head/torchvision mismatch.
        import timm
        backbone = timm.create_model("vit_base_patch16_224", num_classes=0, dynamic_img_size=True)
        sd = torch.load(model_path, map_location=DEVICE, weights_only=False)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        bsd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        if not bsd:
            raise RuntimeError(f"no 'backbone.*' keys found in {model_path}")
        missing, unexpected = backbone.load_state_dict(bsd, strict=False)
        if missing or unexpected:
            print(f"[prepare_model_eval] {ssl_name} backbone load: "
                  f"{len(missing)} missing, {len(unexpected)} unexpected keys")
        model = backbone
    else:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size = DICT_MODEL[model_name][1]
        if "ViT" in model_name:
            # ViT forward is not a plain nn.Sequential over children; match the
            # training-time backbone (drop the classification head only).
            encoder.heads = nn.Identity()
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
        model = DICT_SSL[ssl_name](DEVICE=DEVICE).prepare_featurize_model(
            backbone, model_path=model_path,
            head_size=size,
        )
    model.to(DEVICE)
    model.eval()
    return model
