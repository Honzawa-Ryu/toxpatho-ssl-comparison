# -*- coding: utf-8 -*-
"""
# ABMIL による WSI レベル評価

`scripts/evaluate/abmil_eval.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。CLI と fold ループは呼び出し側に残してある。

`train_abmil` の既定は「固定エポック・val はモデル選択に使わない」。
経緯は同関数の docstring と REFACTOR_PLAN.md §3-2 を参照。
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import lib.model.zoo as zoo
import lib.wsi as wsi_lib


WSI_EXTENSIONS = {".svs", ".ndpi", ".tiff", ".tif", ".vms", ".vmu", ".scn", ".mrxs", ".bif"}

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_wsis(directory: str):
    paths = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in WSI_EXTENSIONS:
            paths.append(os.path.join(directory, f))
    return paths


class _PatchDataset(Dataset):
    """Extracts patches from a single WSI for feature extraction."""

    def __init__(self, wsi_path: str, patch_size: int, num_patch: int, seed: int):
        from openslide import OpenSlide
        wsi = OpenSlide(wsi_path)
        mask = wsi_lib.get_mask_inside(wsi, patch_size=patch_size)
        h, w = mask.shape
        indices = np.where(mask.flatten())[0]
        if len(indices) == 0:
            raise ValueError(f"No valid patches found in {wsi_path}")
        if len(indices) > num_patch:
            rng = np.random.RandomState(seed)
            indices = rng.choice(indices, num_patch, replace=False)
        self.wsi = wsi
        self.patch_size = patch_size
        self.locations = [(int(patch_size * (i // w)), int(patch_size * (i % w))) for i in indices]

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        y, x = self.locations[idx]
        patch = self.wsi.read_region((x, y), 0, (self.patch_size, self.patch_size))
        patch = Image.fromarray(np.array(patch, np.uint8)[:, :, :3])
        return PATCH_TRANSFORM(patch)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def build_backbone(model_name: str, ssl_name: str, checkpoint_path: str, device: torch.device):
    """
    Reconstruct SSL backbone from checkpoint.
    Supports both state.pt (model_state_dict key) and model_ssl.pt (direct state dict).
    Returns (backbone, feature_dim).
    """
    encoder = zoo.DICT_MODEL[model_name][0](weights=None)
    feat_dim = zoo.DICT_MODEL[model_name][1]

    if "ViT" in model_name:
        encoder.heads = nn.Identity()
        backbone = encoder
    elif model_name == "DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
    else:
        backbone = nn.Sequential(*list(encoder.children())[:-1])

    ssl_cls = zoo.DICT_SSL[ssl_name]
    full_model, _ = ssl_cls(DEVICE=device).prepare_model(backbone, head_size=feat_dim)

    raw = torch.load(checkpoint_path, map_location=device)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    full_model.load_state_dict(state_dict)

    backbone = full_model.backbone
    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    return backbone, feat_dim


# ---------------------------------------------------------------------------
# Feature extraction (with disk cache)
# ---------------------------------------------------------------------------

def _extract_features(backbone, wsi_path, device, patch_size, num_patch, batch_size, seed):
    ds = _PatchDataset(wsi_path, patch_size=patch_size, num_patch=num_patch, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in loader:
            out = backbone(batch.to(device))
            feats.append(out.flatten(1).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


def get_features(backbone, wsi_path, label_str, cache_dir, device,
                 patch_size, num_patch, batch_size, seed):
    name = os.path.splitext(os.path.basename(wsi_path))[0]
    cache_path = os.path.join(cache_dir, label_str, f"{name}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    feats = _extract_features(backbone, wsi_path, device, patch_size, num_patch, batch_size, seed)
    np.save(cache_path, feats)
    return feats


# ---------------------------------------------------------------------------
# ABMIL training / evaluation
# ---------------------------------------------------------------------------

def train_abmil(model, train_feats, train_labels, val_feats, val_labels,
                device, lr, num_epoch, patience, early_stop=False):
    """ABMIL を固定エポックで学習する。

    既定（`early_stop=False`）では val をモデル選択に使わない。これは
    REFACTOR_PLAN.md §3-2 / TODO.md B章 A案 の修正:

    以前は毎エポック val AUC を測り、その最大値のエポックの重みを復元して返していた。
    呼び出し側 (`main`) はその復元済みモデルを**同じ val** で評価して AUC を報告するため、
    報告値が実質「エポック中の val AUC 最大値」となり楽観バイアスを含んでいた。
    val 枚数が少ないほど影響が大きい。

    さらに、比較対象の `mean_pool_baseline` は train で fit -> val で予測という
    正しい手順を踏んでいるため、**ABMIL と mean-pool の比較自体が不公平**だった。

    `early_stop=True` で旧挙動（val による best-epoch 選択）に戻せるが、
    その場合に報告される AUC は楽観バイアスを含む。
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None
    no_improve = 0
    last_auc = float("nan")

    for _ in range(num_epoch):
        model.train()
        order = list(range(len(train_feats)))
        random.shuffle(order)
        for i in order:
            x = torch.from_numpy(train_feats[i]).to(device)
            y = torch.tensor([float(train_labels[i])], device=device)
            optimizer.zero_grad()
            logit, _ = model(x)
            criterion(logit.unsqueeze(0), y).backward()
            optimizer.step()

        # val AUC はモニタリング用に毎エポック計算するが、既定ではモデル選択に使わない
        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for x, y in zip(val_feats, val_labels):
                logit, _ = model(torch.from_numpy(x).to(device))
                probs.append(torch.sigmoid(logit).item())
                trues.append(y)

        try:
            auc = roc_auc_score(trues, probs)
        except ValueError:
            auc = 0.5
        last_auc = auc

        if not early_stop:
            continue

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if early_stop and best_state:
        model.load_state_dict(best_state)
        return model, best_auc
    # 固定エポック: 最終エポックのモデルをそのまま返す
    return model, last_auc


def eval_abmil(model, feats_list, labels, device):
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for x in feats_list:
            logit, _ = model(torch.from_numpy(x).to(device))
            p = torch.sigmoid(logit).item()
            probs.append(p)
            preds.append(int(p >= 0.5))
    return probs, preds


# ---------------------------------------------------------------------------
# Mean-pooling baseline
# ---------------------------------------------------------------------------

def mean_pool_baseline(train_feats, train_labels, val_feats, val_labels):
    X_train = np.stack([f.mean(0) for f in train_feats])
    X_val   = np.stack([f.mean(0) for f in val_feats])
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, train_labels)
    probs = clf.predict_proba(X_val)[:, 1]
    preds = clf.predict(X_val)
    return probs, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
