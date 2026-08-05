# -*- coding: utf-8 -*-
"""
WSIパッチのぼやけ(blur)QC(Goal.md 参照)。

Step 1: `compute_blur_scores` — TRIDENT座標h5とWSI本体からパッチを取り出し、
        ラプラシアン分散でぼやけ度を計算する。
Step 2: `sample_patches_to_memmap` — ぼやけスコアで足切りした上でランダムに
        サンプリングし、`numpy.memmap` 形式で保存する。

CLIの組み立ては呼び出し側（scripts/analysis/score_blur.py /
scripts/analysis/sample_patches_memmap.py）が担う（REFACTOR_PLAN.md §5-0）。
"""
import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from openslide import OpenSlide


def laplacian_variance(patch_rgb: np.ndarray) -> float:
    """RGBパッチ1枚のラプラシアン分散を返す(値が小さいほどぼやけている)。"""
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_blur_scores(
    wsi_path: str,
    coords_h5_path: str,
    patch_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
    """WSI1枚分の座標を読み、各パッチのぼやけスコアを計算する。

    Args:
        wsi_path: WSI本体(.svs等)のパス。
        coords_h5_path: TRIDENT座標h5のパス(`coords` データセットを持つこと)。
        patch_size: パッチ一辺のピクセル数(level 0基準)。

    Returns:
        coords: (n_patches, 2) int32 — coords_h5_path からそのままコピー。
        scores: (n_patches,) float32 — ラプラシアン分散。
    """
    with h5py.File(coords_h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(
                f"{coords_h5_path} に 'coords' データセットが見つかりません。"
                f" 実際のキー: {list(f.keys())}"
            )
        coords = f["coords"][:].astype(np.int32)

    scores = np.empty(len(coords), dtype=np.float32)
    wsi = OpenSlide(wsi_path)
    try:
        for i, (x, y) in enumerate(coords):
            patch = wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size))
            patch_rgb = np.array(patch.convert("RGB"))
            scores[i] = laplacian_variance(patch_rgb)
    finally:
        wsi.close()

    return coords, scores


def _derive_rng(seed: int, wsi_id: str) -> np.random.Generator:
    """WSIごとに独立したサンプリング結果になるよう、seedとwsi_idから派生させたRNGを作る。"""
    wsi_hash = int(hashlib.sha256(wsi_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    return np.random.default_rng(np.random.SeedSequence([seed, wsi_hash]))


def build_offset_table(wsi_ids: list[str], n_patches_per_wsi: int) -> dict[str, int]:
    """wsi_idのソート順に、各WSIへ固定サイズのmemmap区間を割り当てる。

    offsetがwsi_idのみから決定的に求まるため、途中で一部WSIをスキップして
    resumeしても他WSIの区間とはずれない。
    """
    return {wsi_id: i * n_patches_per_wsi for i, wsi_id in enumerate(sorted(wsi_ids))}


def init_shared_memmap(memmap_path: str, total_n_patches: int, patch_size: int) -> None:
    """全WSI共有の巨大memmapファイルを確保する。既に存在する場合は何もしない(resume対応)。"""
    path = Path(memmap_path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(total_n_patches, patch_size, patch_size, 3))
    mm.flush()
    del mm


def sample_patches_to_shared_memmap(
    wsi_path: str,
    coords: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    n_patches: int,
    patch_size: int,
    memmap_path: str,
    total_n_patches: int,
    offset: int,
    seed: int,
) -> list[dict]:
    """1WSI分: 全体統一のぼやけ閾値でフィルタ + ランダムサンプリングし、
    全WSI共有の単一memmapのうち自分に割り当てられた
    `[offset, offset + n_patches)` 区間だけへ書き込む。

    他WSIの区間には触れないため、WSIごとに独立して(並列に)呼んでもよい。
    閾値通過パッチが n_patches 未満の場合は、水増しせず ValueError にする
    (呼び出し前に全WSIが n_patches 以上確保できることを確認しておく想定)。

    Args:
        offset: `build_offset_table` が返す、このWSI専用の書き込み開始行。
        total_n_patches: 共有memmap全体の行数(全WSI分の合計)。

    Returns:
        index行のlist(1WSI分、n_patches件)。各要素は
        `{wsi_id, row, x, y, blur_score}`。
    """
    wsi_id = Path(wsi_path).stem

    passed = np.where(scores >= threshold)[0]
    if len(passed) < n_patches:
        raise ValueError(
            f"{wsi_id}: 閾値通過パッチ数 {len(passed)} が n_patches={n_patches} 未満です。"
            f" 水増しはしないため、呼び出し前に閾値または対象WSIを見直してください。"
        )

    rng = _derive_rng(seed, wsi_id)
    selected = rng.choice(passed, size=n_patches, replace=False)
    sel_coords = coords[selected]
    sel_scores = scores[selected]

    mm = np.memmap(
        memmap_path, dtype=np.uint8, mode="r+",
        shape=(total_n_patches, patch_size, patch_size, 3),
    )
    wsi = OpenSlide(wsi_path)
    try:
        for i, (x, y) in enumerate(sel_coords):
            patch = wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size)).convert("RGB")
            mm[offset + i] = np.array(patch, dtype=np.uint8)
    finally:
        wsi.close()
    mm.flush()
    del mm

    return [
        {
            "row": offset + i,
            "wsi_id": wsi_id,
            "x": int(x),
            "y": int(y),
            "blur_score": float(score),
        }
        for i, ((x, y), score) in enumerate(zip(sel_coords, sel_scores))
    ]


def sample_patches_to_memmap(
    wsi_path: str,
    coords: np.ndarray,
    scores: np.ndarray,
    threshold_scope: str,
    percentile: float,
    n_patches: int,
    patch_size: int,
    out_dir: str,
    seed: int,
    global_threshold: float | None = None,
) -> dict:
    """閾値フィルタ + ランダムサンプリング + memmap保存をWSI1枚分行う。

    Args:
        threshold_scope: "none" | "per_slide" | "global"。
        percentile: 下位何%を除外するか(per_slide/globalで使用)。
        global_threshold: threshold_scope="global" のとき使う、事前算出済みの
            スコア閾値(全WSIプールでのパーセンタイル値)。呼び出し側が
            scores_summary.csv 相当のプール全体から1回だけ算出して渡す。
        seed: 再現性のための基準シード。WSIごとの派生シードは内部で計算する。

    Returns:
        meta: `{wsi_id}_patches.meta.json` に書く内容と同じ辞書。
    """
    wsi_id = Path(wsi_path).stem

    if threshold_scope == "none":
        threshold = -np.inf
    elif threshold_scope == "per_slide":
        threshold = float(np.percentile(scores, percentile))
    elif threshold_scope == "global":
        if global_threshold is None:
            raise ValueError("threshold_scope='global' には global_threshold が必須です。")
        threshold = float(global_threshold)
    else:
        raise ValueError(f"未知の threshold_scope: {threshold_scope!r}")

    passed = np.where(scores >= threshold)[0]

    rng = _derive_rng(seed, wsi_id)
    if len(passed) <= n_patches:
        if len(passed) < n_patches:
            print(
                f"[warn] {wsi_id}: 通過パッチ数 {len(passed)} が "
                f"n_patches={n_patches} 未満のため、あるだけ全て採用します(水増しはしません)。"
            )
        selected = passed
    else:
        selected = rng.choice(passed, size=n_patches, replace=False)

    sel_coords = coords[selected]
    sel_scores = scores[selected]

    patches = np.empty((len(selected), patch_size, patch_size, 3), dtype=np.uint8)
    wsi = OpenSlide(wsi_path)
    try:
        for i, (x, y) in enumerate(sel_coords):
            patch = wsi.read_region((int(x), int(y)), 0, (patch_size, patch_size))
            patches[i] = np.array(patch.convert("RGB"))
    finally:
        wsi.close()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = out_dir / f"{wsi_id}_patches.memmap"
    meta_path = out_dir / f"{wsi_id}_patches.meta.json"

    mm = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=patches.shape)
    mm[:] = patches
    mm.flush()
    del mm

    meta = {
        "wsi_id": wsi_id,
        "shape": list(patches.shape),
        "dtype": "uint8",
        "patch_size": patch_size,
        "coords": sel_coords.tolist(),
        "blur_score": sel_scores.tolist(),
        "threshold_scope": threshold_scope,
        "percentile": percentile,
        "seed": seed,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return meta
