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
    build_offset_table,
    compute_blur_scores,
    init_shared_memmap,
    laplacian_variance,
    sample_patches_to_memmap,
    sample_patches_to_shared_memmap,
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


class TestBuildOffsetTable(unittest.TestCase):
    def test_offsets_follow_sorted_wsi_id_order(self):
        table = build_offset_table(["b", "a", "c"], n_patches_per_wsi=10)
        self.assertEqual(table, {"a": 0, "b": 10, "c": 20})


class TestInitSharedMemmap(unittest.TestCase):
    def test_creates_file_with_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.memmap"
            init_shared_memmap(str(path), total_n_patches=6, patch_size=224)
            self.assertTrue(path.exists())
            mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(6, 224, 224, 3))
            self.assertEqual(mm.shape, (6, 224, 224, 3))

    def test_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.memmap"
            init_shared_memmap(str(path), total_n_patches=2, patch_size=224)
            mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=(2, 224, 224, 3))
            mm[0] = 123
            mm.flush()
            del mm

            init_shared_memmap(str(path), total_n_patches=2, patch_size=224)

            mm2 = np.memmap(path, dtype=np.uint8, mode="r", shape=(2, 224, 224, 3))
            self.assertTrue(np.all(mm2[0] == 123))


class TestSamplePatchesToSharedMemmap(unittest.TestCase):
    def _run(self, memmap_path, wsi_id, threshold, offset, total_n_patches, n_patches=3, seed=42):
        coords = np.array([[i * 224, 0] for i in range(10)], dtype=np.int32)
        scores = np.arange(10, dtype=np.float32)  # 0..9, ascending sharpness
        with mock.patch("lib.analysis.blur_qc.OpenSlide", _FakeOpenSlide):
            rows = sample_patches_to_shared_memmap(
                wsi_path=f"{wsi_id}.svs",
                coords=coords,
                scores=scores,
                threshold=threshold,
                n_patches=n_patches,
                patch_size=224,
                memmap_path=memmap_path,
                total_n_patches=total_n_patches,
                offset=offset,
                seed=seed,
            )
        return rows

    def test_filters_by_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            memmap_path = str(Path(tmp) / "shared.memmap")
            init_shared_memmap(memmap_path, total_n_patches=3, patch_size=224)
            rows = self._run(memmap_path, "wsi001", threshold=7.0, offset=0, total_n_patches=3)
            self.assertTrue(all(r["blur_score"] >= 7.0 for r in rows))

    def test_writes_only_own_offset_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            memmap_path = str(Path(tmp) / "shared.memmap")
            total_n = 6
            init_shared_memmap(memmap_path, total_n_patches=total_n, patch_size=224)
            self._run(memmap_path, "wsi001", threshold=0.0, offset=3, total_n_patches=total_n, n_patches=3)

            mm = np.memmap(memmap_path, dtype=np.uint8, mode="r", shape=(total_n, 224, 224, 3))
            # 他WSIの区間(offset未満)は書き込まれていない(ゼロ初期化のまま)。
            self.assertTrue(np.all(mm[0:3] == 0))
            # 自分の区間には何かしら書き込まれている。
            self.assertFalse(np.all(mm[3:6] == 0))

    def test_raises_when_fewer_than_n_patches_pass_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            memmap_path = str(Path(tmp) / "shared.memmap")
            init_shared_memmap(memmap_path, total_n_patches=3, patch_size=224)
            with self.assertRaises(ValueError):
                self._run(memmap_path, "wsi001", threshold=9.0, offset=0, total_n_patches=3, n_patches=3)

    def test_reproducible_with_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            path1 = str(Path(tmp1) / "shared.memmap")
            path2 = str(Path(tmp2) / "shared.memmap")
            init_shared_memmap(path1, total_n_patches=3, patch_size=224)
            init_shared_memmap(path2, total_n_patches=3, patch_size=224)
            rows1 = self._run(path1, "wsi001", threshold=0.0, offset=0, total_n_patches=3, seed=42)
            rows2 = self._run(path2, "wsi001", threshold=0.0, offset=0, total_n_patches=3, seed=42)
            self.assertEqual([r["x"] for r in rows1], [r["x"] for r in rows2])


if __name__ == "__main__":
    unittest.main()
