# -*- coding: utf-8 -*-
"""lib/analysis/blur_qc.py の単体テスト(ダミーWSI/パッチによる検証、Goal.md §9)。

OpenSlideは実スライドを開かず、`OpenSlide` をFakeクラスにモンキーパッチして
決定的なパッチ画像を返すことで、実データ・GPUなしで検証する。

Run:
    python -m unittest lib.analysis.tests.test_blur_qc
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from lib.analysis.blur_qc import (
    compute_blur_scores,
    laplacian_variance,
    sample_patches_to_memmap,
)


class _FakeOpenSlide:
    """WSIごとにread_regionを (x, y) から決定的に生成するダミー実装。"""

    def __init__(self, path):
        self.path = path
        self.closed = False

    def read_region(self, location, level, size):
        x, y = location
        w, h = size
        rng = np.random.default_rng(abs(hash((x, y))) % (2**32))
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def close(self):
        self.closed = True


class TestLaplacianVariance(unittest.TestCase):
    def test_sharp_patch_scores_higher_than_flat_patch(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        flat = np.full((224, 224, 3), 128, dtype=np.uint8)
        self.assertGreater(laplacian_variance(sharp), laplacian_variance(flat))

    def test_flat_patch_has_zero_variance(self):
        flat = np.full((64, 64, 3), 10, dtype=np.uint8)
        self.assertEqual(laplacian_variance(flat), 0.0)


class TestComputeBlurScores(unittest.TestCase):
    def test_reads_coords_and_computes_scores(self):
        coords = np.array([[0, 0], [224, 0], [448, 224]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            coords_h5_path = Path(tmp) / "wsi001_patches.h5"
            with h5py.File(coords_h5_path, "w") as f:
                f.create_dataset("coords", data=coords)

            with mock.patch("lib.analysis.blur_qc.OpenSlide", _FakeOpenSlide):
                out_coords, scores = compute_blur_scores(
                    "wsi001.svs", str(coords_h5_path), patch_size=224
                )

        np.testing.assert_array_equal(out_coords, coords)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(scores.dtype, np.float32)
        self.assertTrue(np.all(scores >= 0))

    def test_missing_coords_key_raises_keyerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            coords_h5_path = Path(tmp) / "bad.h5"
            with h5py.File(coords_h5_path, "w") as f:
                f.create_dataset("not_coords", data=np.zeros((1, 2), dtype=np.int32))

            with self.assertRaises(KeyError):
                compute_blur_scores("wsi001.svs", str(coords_h5_path))


class TestSamplePatchesToMemmap(unittest.TestCase):
    def _run(self, tmp, threshold_scope, percentile=0.0, n_patches=5, global_threshold=None, seed=42):
        coords = np.array([[i * 224, 0] for i in range(10)], dtype=np.int32)
        scores = np.arange(10, dtype=np.float32)  # 0..9, ascending sharpness
        with mock.patch("lib.analysis.blur_qc.OpenSlide", _FakeOpenSlide):
            meta = sample_patches_to_memmap(
                wsi_path="wsi001.svs",
                coords=coords,
                scores=scores,
                threshold_scope=threshold_scope,
                percentile=percentile,
                n_patches=n_patches,
                patch_size=224,
                out_dir=tmp,
                seed=seed,
                global_threshold=global_threshold,
            )
        return meta

    def test_per_slide_threshold_filters_by_percentile(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._run(tmp, "per_slide", percentile=50.0, n_patches=3)
            self.assertEqual(len(meta["coords"]), 3)
            self.assertTrue(all(s >= 4.5 for s in meta["blur_score"]))

    def test_global_threshold_used_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._run(tmp, "global", n_patches=3, global_threshold=7.0)
            self.assertTrue(all(s >= 7.0 for s in meta["blur_score"]))

    def test_none_scope_uses_all_when_fewer_than_n_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._run(tmp, "none", n_patches=100)
            self.assertEqual(len(meta["coords"]), 10)

    def test_no_oversampling_beyond_available_passed_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._run(tmp, "per_slide", percentile=90.0, n_patches=100)
            # scores 0..9, keep top 10% (>= 9th percentile score) -> only 1 patch passes
            self.assertLessEqual(len(meta["coords"]), 10)
            self.assertEqual(len(meta["coords"]), len(set(map(tuple, meta["coords"]))))

    def test_reproducible_with_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            meta1 = self._run(tmp1, "none", n_patches=5, seed=42)
            meta2 = self._run(tmp2, "none", n_patches=5, seed=42)
            self.assertEqual(meta1["coords"], meta2["coords"])

    def test_memmap_roundtrip_matches_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = self._run(tmp, "none", n_patches=4)
            meta_path = Path(tmp) / "wsi001_patches.meta.json"
            loaded_meta = json.loads(meta_path.read_text())
            self.assertEqual(loaded_meta["shape"], meta["shape"])

            memmap_path = Path(tmp) / "wsi001_patches.memmap"
            shape = tuple(loaded_meta["shape"])
            mm = np.memmap(memmap_path, dtype=np.uint8, mode="r", shape=shape)
            self.assertEqual(mm.shape, (4, 224, 224, 3))

    def test_global_scope_without_threshold_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self._run(tmp, "global", n_patches=3, global_threshold=None)


if __name__ == "__main__":
    unittest.main()
