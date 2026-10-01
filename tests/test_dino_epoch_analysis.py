"""CPU tests for sample correspondence and descriptive CKA controls."""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from lib.analysis.cka import linear_cka
from lib.analysis.dino_epoch_analysis import sample_tar, slide_center


class DinoEpochAnalysisTests(unittest.TestCase):
    def test_cka_ignores_rotation_translation_and_uniform_scaling(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(100, 12))
        rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        self.assertAlmostEqual(linear_cka(x, 3*x@rotation + 11), 1, places=10)
        self.assertLess(linear_cka(x, rng.normal(size=x.shape)), .3)

    def test_slide_center_removes_slide_offsets_preserving_patch_variation(self):
        x = np.random.default_rng(7).normal(size=(100, 12))
        slides = np.repeat(['A', 'B'], 50)
        shifted = x + np.where((slides == 'A')[:, None], 10, -5)
        residual = slide_center(x, slides)
        np.testing.assert_allclose(residual, slide_center(shifted, slides), atol=1e-12)
        self.assertGreater(np.std(residual), .5)
        np.testing.assert_allclose(residual[:50].mean(axis=0), 0, atol=1e-12)

    def test_balanced_selection_reads_corresponding_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'images.tar'
            with tarfile.open(path, 'w') as archive:
                for i, slide in enumerate(['B', 'A', 'B', 'A', 'B', 'A']):
                    for suffix, data in (
                        ('jpg', f'image{i}'.encode()),
                        ('json', json.dumps({'wsi': slide, 'x': i, 'y': 0}).encode()),
                    ):
                        item = tarfile.TarInfo(f'{i:08d}.{suffix}')
                        item.size = len(data)
                        archive.addfile(item, io.BytesIO(data))
            selected, offsets = sample_tar(path, 6, 2, 2)
            self.assertEqual([r['key'] for r in selected],
                             ['00000000', '00000001', '00000002', '00000003'])
            with path.open('rb') as source:
                for record in selected:
                    offset, size = offsets[record['key']]
                    source.seek(offset)
                    self.assertEqual(source.read(size), f"image{int(record['key'])}".encode())
            self.assertEqual(sample_tar(path, 6, 2, 2)[0], selected)
            with self.assertRaises(ValueError):
                sample_tar(path, 6, 4, 2)


if __name__ == '__main__':
    unittest.main()
