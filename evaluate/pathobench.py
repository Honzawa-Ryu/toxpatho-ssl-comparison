# -*- coding: utf-8 -*-
"""
# Patho-Bench 向けの WSI パッチ特徴抽出

`scripts/evaluate/extract_features_pathobench.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。CLI と入出力の組み立ては呼び出し側に残してある。
"""
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from openslide import OpenSlide
from tqdm import tqdm

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 注意: abmil 側の PATCH_TRANSFORM と違い Resize を含まない。
# WSIPatchDataset は既に patch_size で切り出したパッチを渡すため。
PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    NORMALIZE,
])


def _get_tissue_mask(wsi: OpenSlide, patch_size: int, level: int = 0) -> np.ndarray:
    """Return boolean mask (rows, cols) where True = tissue patch location."""
    import cv2

    thumb_level = wsi.get_best_level_for_downsample(32)
    downsample = wsi.level_downsamples[thumb_level]
    thumb = wsi.read_region((0, 0), thumb_level, wsi.level_dimensions[thumb_level])
    thumb = np.array(thumb.convert("RGB"))

    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Convert mask back to level-0 grid
    w0, h0 = wsi.dimensions
    step = int(patch_size)
    cols = w0 // step
    rows = h0 // step
    mask = np.zeros((rows, cols), dtype=bool)

    scale = patch_size / downsample
    for r in range(rows):
        for c in range(cols):
            mx = int(c * scale)
            my = int(r * scale)
            mx = min(mx, binary.shape[1] - 1)
            my = min(my, binary.shape[0] - 1)
            if binary[my, mx] > 0:
                mask[r, c] = True
    return mask


def get_patch_coords(wsi: OpenSlide, patch_size: int) -> list:
    """Return list of (x, y) level-0 pixel coordinates for tissue patches."""
    mask = _get_tissue_mask(wsi, patch_size)
    coords = []
    for r, c in zip(*np.where(mask)):
        coords.append((int(c * patch_size), int(r * patch_size)))
    return coords

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WSIPatchDataset(torch.utils.data.Dataset):
    def __init__(self, wsi: OpenSlide, coords: list, patch_size: int, transform=None):
        self.wsi = wsi
        self.coords = coords
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        patch = self.wsi.read_region((x, y), 0, (self.patch_size, self.patch_size))
        patch = patch.convert("RGB")
        if self.transform:
            patch = self.transform(patch)
        return patch, torch.tensor([x, y], dtype=torch.int32)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_wsi_features(
    model: nn.Module,
    wsi_path: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple:
    """Return (features, coords) arrays for a single WSI."""
    wsi = OpenSlide(wsi_path)
    coords = get_patch_coords(wsi, patch_size)
    if not coords:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    dataset = WSIPatchDataset(wsi, coords, patch_size, transform=PATCH_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    all_features, all_coords = [], []
    for patches, patch_coords in tqdm(loader, desc=f"  {Path(wsi_path).name}", leave=False):
        patches = patches.to(device)
        feats = model(patches)
        if feats.dim() > 2:
            feats = feats.flatten(start_dim=1)
        all_features.append(feats.cpu().numpy().astype(np.float32))
        all_coords.append(patch_coords.numpy())

    wsi.close()
    return np.concatenate(all_features, axis=0), np.concatenate(all_coords, axis=0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
