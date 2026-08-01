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
import yaml
from PIL import Image
import webdataset as wds

import lib.model.zoo as zoo

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


def run(methods_config, data_dir="data/shards", n_patches=2000, batch_size=256,
        output_dir="outputs/representation_analysis", grayscale=False, filter_wsi=""):
    """全手法で共通のパッチ集合を埋め込み、emb_{name}.npy として保存する。

    旧 `scripts/analysis/extract_embeddings.py` の main() 本体をそのまま関数化した
    もの（REFACTOR_PLAN.md §5-0「研究の実処理は lib/」/ Phase 4）。
    scripts 側は CLI シムとして残してある。

    checkpoint が見つからない手法はスキップするので、学習が全部終わる前でも回せる。
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(methods_config) as f:
        cfg = yaml.safe_load(f)
    shard_idx = (cfg.get("patches") or {}).get("shards")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    x, thumbs, meta = load_fixed_patches(
        data_dir, shard_idx, n_patches,
        grayscale=grayscale, filter_wsi=filter_wsi)
    np.save(os.path.join(output_dir, "patches_thumbs.npy"), thumbs)
    with open(os.path.join(output_dir, "patches_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    done = {}
    for m in cfg["methods"]:
        name = m["name"]
        ssl_name = m["ssl_name"]
        is_foundation = ssl_name == "foundation"
        if is_foundation:
            # Pretrained external foundation model (e.g. UNI): no in-project
            # checkpoint; weights come from timm/HF cache, so no model_path to check.
            print(f"[embed] {name}  ({m['model_name']} / foundation)")
            model = zoo.prepare_foundation_eval(
                model_name=m["model_name"], DEVICE=device)
        else:
            mp = m["model_path"]
            if not os.path.exists(mp):
                print(f"[skip] {name}: checkpoint not found ({mp})")
                continue
            print(f"[embed] {name}  ({m['model_name']} / {ssl_name})")
            model = zoo.prepare_model_eval(
                model_name=m["model_name"], ssl_name=ssl_name,
                model_path=mp, pretrained=False, DEVICE=device)
        emb = embed(model, x, device, batch_size=batch_size)
        np.save(os.path.join(output_dir, f"emb_{name}.npy"), emb)
        print(f"        -> emb_{name}.npy  shape={emb.shape}")
        done[name] = list(emb.shape)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with open(os.path.join(output_dir, "extract_summary.json"), "w") as f:
        json.dump({"n_patches": int(x.size(0)), "embeddings": done}, f, indent=2)
    print(f"[done] embeddings for {list(done)} in {output_dir}")
    return done
