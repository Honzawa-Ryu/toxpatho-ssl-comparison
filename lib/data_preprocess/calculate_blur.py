# -*- coding: utf-8 -*-
"""
入力済みパッチ画像に対するぼやけ(blur)スコアリング。

`lib/analysis/blur_qc.py` の `compute_blur_scores` はWSI本体 + TRIDENT座標h5
から自前でパッチを切り出す設計だが、こちらは呼び出し側が既に切り出し済みの
パッチ配列をそのまま受け取り、ラプラシアン分散でスコア化してh5に追記する。
実データ(TRIDENT座標h5 + WSI本体)から直接算出したい場合は
`append_blur_scores_from_wsi` を使う。
"""
from pathlib import Path

import cv2
import h5py
import numpy as np
from openslide import OpenSlide


def laplacian_blur_score(patch_rgb: np.ndarray) -> float:
    """RGBパッチ1枚のラプラシアン分散を返す(値が小さいほどぼやけている)。"""
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_blur_scores(patches: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) のRGBパッチ配列から各パッチのラプラシアン分散を計算する。

    Args:
        patches: (N, H, W, 3) のRGBパッチ配列(uint8を想定)。

    Returns:
        scores: (N,) float32 — 各パッチのラプラシアン分散(ぼやけスコア)。
    """
    if patches.ndim != 4 or patches.shape[-1] != 3:
        raise ValueError(
            f"patches は (N, H, W, 3) の形状を想定していますが、{patches.shape} でした。"
        )

    scores = np.empty(len(patches), dtype=np.float32)
    for i, patch in enumerate(patches):
        scores[i] = laplacian_blur_score(patch)
    return scores


def append_blur_scores_to_h5(
    patches: np.ndarray,
    h5_path: str | Path,
    dataset_name: str = "blur_score",
) -> np.ndarray:
    """パッチ配列のぼやけスコアを計算し、h5に追記して返す。

    `h5_path` は既存のTRIDENT座標h5(`coords` データセットを持つもの)を想定し、
    同じ件数のパッチが渡されたことを検証した上で `dataset_name` のデータセットを
    追記する。ファイルが存在しない場合は新規作成する。同名データセットが既に
    存在する場合は上書きする(再実行時に冪等)。

    Args:
        patches: (N, H, W, 3) のRGBパッチ配列(uint8を想定)。
        h5_path: 追記先のh5ファイルパス。
        dataset_name: 追記するデータセット名。

    Returns:
        scores: (N,) float32 — 各パッチのラプラシアン分散(ぼやけスコア)。
    """
    scores = compute_blur_scores(patches)

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "a") as f:
        if "coords" in f and len(f["coords"]) != len(scores):
            raise ValueError(
                f"{h5_path} の coords 件数({len(f['coords'])})と"
                f" patches 件数({len(scores)})が一致しません。"
            )
        if dataset_name in f:
            del f[dataset_name]
        f.create_dataset(dataset_name, data=scores, compression="gzip")

    return scores


def read_patches_from_wsi(
    wsi_path: str | Path,
    coords: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """WSI本体(svs等)から coords の座標(level 0基準)でパッチを切り出す。

    Args:
        wsi_path: WSI本体(.svs等)のパス。
        coords: (N, 2) の (x, y) 座標配列(level 0基準)。
        patch_size: パッチ一辺のピクセル数(level 0基準)。

    Returns:
        patches: (N, patch_size, patch_size, 3) のRGBパッチ配列(uint8)。
    """
    patches = np.empty((len(coords), patch_size, patch_size, 3), dtype=np.uint8)
    wsi = OpenSlide(str(wsi_path))
    try:
        for i, (x, y) in enumerate(coords):
            patch = wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size))
            patches[i] = np.array(patch.convert("RGB"))
    finally:
        wsi.close()
    return patches


def append_blur_scores_from_wsi(
    wsi_path: str | Path,
    h5_path: str | Path,
    patch_size: int = 224,
    dataset_name: str = "blur_score",
) -> np.ndarray:
    """TRIDENT座標h5とWSI本体からパッチを切り出し、ぼやけスコアを計算してh5に追記する。

    `h5_path` の `coords` データセットから座標を読み、`wsi_path` からその座標の
    パッチを切り出した上で `append_blur_scores_to_h5` に委譲する。

    Args:
        wsi_path: WSI本体(.svs等)のパス。
        h5_path: 座標h5(`coords` データセットを持つもの)のパス。
        patch_size: パッチ一辺のピクセル数(level 0基準)。
        dataset_name: 追記するデータセット名。

    Returns:
        scores: (N,) float32 — 各パッチのラプラシアン分散(ぼやけスコア)。
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(
                f"{h5_path} に 'coords' データセットが見つかりません。"
                f" 実際のキー: {list(f.keys())}"
            )
        coords = f["coords"][:]

    patches = read_patches_from_wsi(wsi_path, coords, patch_size)
    return append_blur_scores_to_h5(patches, h5_path, dataset_name=dataset_name)
