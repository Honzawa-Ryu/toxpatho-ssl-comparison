# -*- coding: utf-8 -*-
"""lib/data_preprocess/calculate_blur.py の単体テスト。

Run:
    python -m unittest lib.data_preprocess.tests.test_calculate_blur
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from lib.data_preprocess.calculate_blur import (
    append_blur_scores_from_wsi,
    append_blur_scores_to_h5,
    compute_blur_scores,
    laplacian_blur_score,
    read_patches_from_wsi,
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


class TestLaplacianBlurScore(unittest.TestCase):
    def test_sharp_patch_scores_higher_than_flat_patch(self):
        rng = np.random.default_rng(0)
        sharp = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        flat = np.full((224, 224, 3), 128, dtype=np.uint8)
        self.assertGreater(laplacian_blur_score(sharp), laplacian_blur_score(flat))

    def test_flat_patch_has_zero_variance(self):
        flat = np.full((64, 64, 3), 10, dtype=np.uint8)
        self.assertEqual(laplacian_blur_score(flat), 0.0)


class TestComputeBlurScores(unittest.TestCase):
    def test_scores_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        patches = rng.integers(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)
        scores = compute_blur_scores(patches)
        self.assertEqual(scores.shape, (5,))
        self.assertEqual(scores.dtype, np.float32)
        self.assertTrue(np.all(scores >= 0))

    def test_rejects_wrong_shape(self):
        bad = np.zeros((5, 32, 32), dtype=np.uint8)  # channel次元が無い
        with self.assertRaises(ValueError):
            compute_blur_scores(bad)


class TestAppendBlurScoresToH5(unittest.TestCase):
    def test_appends_to_existing_coords_h5(self):
        rng = np.random.default_rng(0)
        coords = np.array([[0, 0], [224, 0], [448, 224]], dtype=np.int64)
        patches = rng.integers(0, 256, size=(3, 224, 224, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "wsi001_patches.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("coords", data=coords)

            scores = append_blur_scores_to_h5(patches, h5_path)

            with h5py.File(h5_path, "r") as f:
                self.assertEqual(list(f.keys()), ["blur_score", "coords"])
                np.testing.assert_array_equal(f["coords"][:], coords)
                np.testing.assert_array_equal(f["blur_score"][:], scores)

    def test_creates_new_file_when_missing(self):
        rng = np.random.default_rng(0)
        patches = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "new.h5"
            scores = append_blur_scores_to_h5(patches, h5_path)

            with h5py.File(h5_path, "r") as f:
                np.testing.assert_array_equal(f["blur_score"][:], scores)

    def test_rerun_is_idempotent(self):
        rng = np.random.default_rng(0)
        patches = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "wsi001_patches.h5"
            append_blur_scores_to_h5(patches, h5_path)
            scores2 = append_blur_scores_to_h5(patches, h5_path)

            with h5py.File(h5_path, "r") as f:
                np.testing.assert_array_equal(f["blur_score"][:], scores2)

    def test_mismatched_patch_count_raises(self):
        coords = np.array([[0, 0], [224, 0], [448, 224]], dtype=np.int64)
        rng = np.random.default_rng(0)
        patches = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)  # coordsは3件

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "wsi001_patches.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("coords", data=coords)

            with self.assertRaises(ValueError):
                append_blur_scores_to_h5(patches, h5_path)


class TestReadPatchesFromWsi(unittest.TestCase):
    def test_reads_patches_at_given_coords(self):
        coords = np.array([[0, 0], [224, 0], [448, 224]], dtype=np.int32)
        with mock.patch("lib.data_preprocess.calculate_blur.OpenSlide", _FakeOpenSlide):
            patches = read_patches_from_wsi("wsi001.svs", coords, patch_size=224)
        self.assertEqual(patches.shape, (3, 224, 224, 3))
        self.assertEqual(patches.dtype, np.uint8)


class TestAppendBlurScoresFromWsi(unittest.TestCase):
    def test_reads_coords_h5_and_appends_scores(self):
        coords = np.array([[0, 0], [224, 0], [448, 224]], dtype=np.int32)
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "wsi001_patches.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("coords", data=coords)

            with mock.patch("lib.data_preprocess.calculate_blur.OpenSlide", _FakeOpenSlide):
                scores = append_blur_scores_from_wsi("wsi001.svs", h5_path, patch_size=224)

            with h5py.File(h5_path, "r") as f:
                self.assertEqual(list(f.keys()), ["blur_score", "coords"])
                np.testing.assert_array_equal(f["blur_score"][:], scores)
            self.assertEqual(scores.shape, (3,))
            self.assertEqual(scores.dtype, np.float32)

    def test_missing_coords_key_raises_keyerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "bad.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("not_coords", data=np.zeros((1, 2), dtype=np.int32))

            with self.assertRaises(KeyError):
                append_blur_scores_from_wsi("wsi001.svs", h5_path)


if __name__ == "__main__":
    unittest.main()
