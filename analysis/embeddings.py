# -*- coding: utf-8 -*-
"""
# 固定パッチ集合の埋め込み抽出

`scripts/analysis/extract_embeddings.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。CLI と methods.yaml の解釈は呼び出し側に残してある。
"""
import io
import os
import glob
import json

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import webdataset as wds

THUMB = 96  # thumbnail size (px) stored for prototype montages


def eval_transform(grayscale=False):
    steps = [transforms.Resize((224, 224))]
    if grayscale:
        # decolorize but keep 3 channels so ImageNet-pretrained stems still apply
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


def load_fixed_patches(data_dir, shard_idx, n_patches, grayscale=False, filter_wsi=""):
    """Deterministically load n_patches patches (tensors + thumbnails + meta).

    filter_wsi: if set, keep only patches from that WSI (single-slide analysis).
    grayscale: decolorize inputs (color-ablation control).
    """
    all_shards = sorted(glob.glob(f"{data_dir}/*.tar"))
    if not all_shards:
        raise FileNotFoundError(f"No .tar shards in {data_dir}")
    if shard_idx is None:
        shard_idx = list(range(max(0, len(all_shards) - 4), len(all_shards)))
    shards = [all_shards[i] for i in shard_idx]
    print(f"[patches] using shards: {[os.path.basename(s) for s in shards]}"
          f"{' | grayscale' if grayscale else ''}"
          f"{f' | wsi={filter_wsi}' if filter_wsi else ''}")

    tf = eval_transform(grayscale)
    ds = wds.WebDataset(shards, shardshuffle=False).decode()
    tensors, thumbs, meta = [], [], []
    for sample in ds:
        raw = sample.get("jpg")
        if raw is None:
            continue
        j = sample.get("json", {})
        if isinstance(j, (bytes, str)):
            j = json.loads(j)
        if filter_wsi and str(j.get("wsi", "")) != filter_wsi:
            continue
        img = raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        tensors.append(tf(img))
        thumbs.append(np.asarray(img.resize((THUMB, THUMB)), dtype=np.uint8))
        meta.append({"key": sample.get("__key__", ""), **j})
        if len(tensors) >= n_patches:
            break
    print(f"[patches] loaded {len(tensors)} patches")
    if not tensors:
        raise RuntimeError("No patches matched (check --filter_wsi / shards).")
    return torch.stack(tensors), np.stack(thumbs), meta


@torch.no_grad()
def embed(model, x, device, batch_size=256):
    outs = []
    model.eval()
    for i in range(0, x.size(0), batch_size):
        xb = x[i:i + batch_size].to(device)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = model(xb)
        f = torch.flatten(f, start_dim=1).float().cpu()
        outs.append(f)
    return torch.cat(outs).numpy()
