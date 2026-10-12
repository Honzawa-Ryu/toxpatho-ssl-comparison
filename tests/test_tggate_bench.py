"""CPU tests for the tggate-bench overlay and the Exp 0010-compatible .npz."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from lib.analysis import tggate_bench as tb


def fake_bench(root, slides, tiles_per_slide):
    (root / "lib").mkdir(parents=True)
    labels_dir = root / "outputs" / tb.LABELS_EXP / "min20"
    labels_dir.mkdir(parents=True)
    pd.DataFrame({"slide_id": slides}).to_parquet(labels_dir / "labels.parquet")
    chunk = root / "outputs" / tb.TILES_EXP / "min20__chunk000"
    chunk.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for sid, n in zip(slides, tiles_per_slide):
        np.save(chunk / f"{sid}.npy", rng.integers(0, 256, size=(n, 256, 256, 3), dtype=np.uint8))


class TinyBackbone:
    """Stands in for the ViT: a fixed linear map of the mean pixel to 768 dims."""

    def __init__(self, scale):
        import torch
        self.weight = torch.full((3, 768), float(scale))

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        return x.mean(dim=(2, 3)) @ self.weight


class TggateBenchTests(unittest.TestCase):
    def test_extract_writes_bench_layout_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bench, overlay, ckpts = tmp / "bench", tmp / "overlay", tmp / "ckpt"
            fake_bench(bench, ["11", "12"], [5, 3])
            ckpts.mkdir()
            for e in (20, 480):
                (ckpts / f"model_ep{e}.pt").write_bytes(b"x")
            loader = mock.patch("lib.analysis.dino_epoch_analysis.load_student_backbone",
                                side_effect=lambda p: TinyBackbone(int(Path(p).stem[8:])))
            with loader:
                tb.extract(bench, overlay, "min20", ckpts, [20, 480], batch_size=2, workers=0,
                           device_name="cpu", precision="fp32")
            self.assertTrue((overlay / "lib").is_symlink())
            npz = overlay / "outputs" / tb.EMB_EXP / "min20__dino0028-ep480" / "dino0028-ep480.npz"
            with np.load(npz) as z:
                self.assertEqual(z["tile_embeddings"].shape, (2, 5, 768))
                np.testing.assert_array_equal(z["n_tiles"], [5, 3])
                np.testing.assert_array_equal(z["slide_ids"], ["11", "12"])
                self.assertFalse(z["tile_embeddings"][1, 3:].any())
                np.testing.assert_allclose(z["mean_embeddings"][1], z["tile_embeddings"][1, :3].mean(0), rtol=1e-6)
                ep480 = z["tile_embeddings"]
            with np.load(npz.parents[1] / "min20__dino0028-ep020" / "dino0028-ep020.npz") as z:
                np.testing.assert_allclose(ep480, z["tile_embeddings"] * 24, rtol=1e-5)
            # Every shard exists, so a rerun embeds nothing and reuses the overlay links.
            with mock.patch("lib.analysis.dino_epoch_analysis.load_student_backbone",
                            side_effect=lambda p: TinyBackbone(0)):
                tb.extract(bench, overlay, "min20", ckpts, [20, 480], batch_size=2, workers=0,
                           device_name="cpu", precision="fp32")
            with np.load(npz) as z:
                np.testing.assert_array_equal(z["tile_embeddings"], ep480)

    def test_missing_tiles_stop_before_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = Path(tmp) / "bench"
            fake_bench(bench, ["11"], [2])
            labels = bench / "outputs" / tb.LABELS_EXP / "min20" / "labels.parquet"
            pd.DataFrame({"slide_id": ["11", "99"]}).to_parquet(labels)
            with self.assertRaisesRegex(FileNotFoundError, "1 slides have no tiles"):
                tb.extract(bench, Path(tmp) / "overlay", "min20", tmp, [480], device_name="cpu")

    def test_eval_tasks_cover_anomaly_and_each_pooler_per_epoch(self):
        tasks = tb.eval_tasks("min20", [20, 480])
        self.assertEqual(len(tasks), 8)
        self.assertEqual(tasks[0], (tb.ANOMALY_EXP, ["--dataset", "min20", "--model", "dino0028-ep020",
                                                     "--variant", "study"]))
        self.assertEqual([t[1][-1] for t in tasks[5:]], list(tb.POOLERS))


if __name__ == "__main__":
    unittest.main()
