"""Independent controls for spectra, cached CKA and resumed log ordering."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lib.analysis.cka import linear_cka
from lib.analysis.dino_dense_analysis import prepare_cka, prepared_cka, spectral_metrics, training_records


class DenseAnalysisTests(unittest.TestCase):
    def test_isotropic_and_rank_one_features(self):
        x=np.array([[1,0],[-1,0],[0,1],[0,-1]],dtype=float)
        metrics,_,_=spectral_metrics(x)
        for key in ('singular_effective_rank','covariance_entropy_rank','participation_ratio','stable_rank'):
            self.assertAlmostEqual(metrics[key],2)
        self.assertEqual(metrics['components90'],2)
        rank_one=np.column_stack([np.arange(4),np.arange(4)])
        one,_,_=spectral_metrics(rank_one)
        for key in ('singular_effective_rank','covariance_entropy_rank','participation_ratio','stable_rank'):
            self.assertAlmostEqual(one[key],1,places=6)

    def test_uniform_scale_preserves_rank_but_not_total_variance(self):
        x=np.random.default_rng(7).normal(size=(100,12))
        original,_,_=spectral_metrics(x)
        scaled,_,_=spectral_metrics(10*x+50)
        self.assertAlmostEqual(original['singular_effective_rank'],scaled['singular_effective_rank'])
        self.assertAlmostEqual(scaled['total_centered_variance']/original['total_centered_variance'],100)

    def test_cached_cka_matches_existing_implementation(self):
        rng=np.random.default_rng(7)
        x,y=rng.normal(size=(40,8)),rng.normal(size=(40,12))
        self.assertAlmostEqual(prepared_cka(prepare_cka(x),prepare_cka(y)),linear_cka(x,y),places=12)

    def test_constant_features_are_rejected(self):
        with self.assertRaises(ValueError):
            spectral_metrics(np.ones((8,4)))

    def test_resume_dates_override_filename_sort(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            (base/'log_195710.txt').write_text('[20260917-195710] Epoch: 420, train_loss: 2.0, lr: 1e-4, grad_norm: 3.0, ln_gain: 1.0\n')
            (base/'log_151710.txt').write_text('[20260923-151710] Resumed from state.pt\n[20260923-152310] Epoch: 420, train_loss: 1.0, lr: 5.4e-5, grad_norm: 4.0, ln_gain: 1.2\n[20260923-152311] eff_rank: 542.05, feat_std: 2.63\n')
            (base/'log_151710_rank1.txt').write_text('[20260923-151710] Epoch: 420, train_loss: 99.0\n')
            rows,events,sources=training_records(base)
            self.assertEqual(rows[0]['train_loss'],1)
            self.assertEqual(rows[0]['eff_rank'],542.05)
            self.assertEqual(events[0]['first_logged_epoch'],420)
            self.assertEqual(len(sources),2)


if __name__=='__main__':
    unittest.main()
