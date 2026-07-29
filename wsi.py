# -*- coding: utf-8 -*-
"""
# WSI tissue masking

旧 `src/utils.py` の WSI パッチ関連部分を切り出したもの
（REFACTOR_PLAN.md §2-5 / §5-3 / Phase 1-b）。

@author: Katsuhisa MORITA
"""
import numpy as np
import cv2


def get_mask_inside(
    image,
    patch_size:int=224,
    slice_min_patch:int=1000,
    adjacent_cells_slice=[(-1,0),(1,0),(0,-1),(0,1),],
    adjacent_cells_inside=[(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)],
    ):
    """Return a bool mask of patch positions fully inside a tissue slice.

    Parameters
    ----------
    image: openslide.OpenSlide
    patch_size: int
    """
    # load
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    # get mask
    saturation = whole[:,:,1] #saturation
    threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    # get slices > slice_min_patch
    left_mask = np.full((mask.shape[0]+1, mask.shape[1]+1), fill_value=False, dtype=bool)
    left_mask[:-1, :-1] = mask
    n_left_mask = np.sum(left_mask)
    slice_idx = np.full_like(mask, fill_value=-1, dtype=int)
    i_slice = 0
    while n_left_mask > 0:
        mask_i = np.where(left_mask)
        slice_left_indices = [(mask_i[0][0], mask_i[1][0])]
        while len(slice_left_indices) > 0:
            i, j = slice_left_indices.pop()
            if left_mask[i, j]:
                slice_idx[i, j] = i_slice
                for adj_i, adj_j in adjacent_cells_slice:
                    if left_mask[i+adj_i, j+adj_j]:
                        slice_left_indices.append((i+adj_i, j+adj_j))
                left_mask[i, j] = False
                n_left_mask -= 1
        slice_mask = slice_idx == i_slice
        if np.sum(slice_mask) < slice_min_patch:
            slice_idx[slice_mask] = -1
        else:
            i_slice += 1
    # re-difinition mask
    mask=slice_idx>-1
    # get inside mask
    mask_inside=np.zeros(mask.shape, dtype=bool)
    for i in range(1,mask.shape[0]-1):
        for v in range(1,mask.shape[1]-1):
            tf = mask[i][v]
            if tf:
                for adj in adjacent_cells_inside:
                    tf&=mask[i+adj[0]][v+adj[1]]
            mask_inside[i][v]=tf
    return mask_inside
