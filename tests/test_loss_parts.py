"""BT の損失の内訳(on_diag / off_diag、2026-10-14)。コンテナ内:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_loss_parts"
"""
import unittest

import torch

from lib.sslmodel.models.barlowtwins import BarlowTwinsLoss
from lib.sslmodel.sslutils import SimSiam
from lib.trainer.loop import LossPartsMeter


class TestLossParts(unittest.TestCase):
    def test_bt_parts_sum_to_loss_and_epoch_mean(self):
        torch.manual_seed(0)
        crit = BarlowTwinsLoss(lambda_param=5e-3)
        meter = LossPartsMeter(crit)
        losses = []
        for _ in range(3):
            meter.before(crit)
            loss = crit(torch.randn(32, 8), torch.randn(32, 8))
            meter.add(crit)
            on, off = crit.last_parts.tolist()
            self.assertAlmostEqual(on + 5e-3 * off, loss.item(), places=3, msg="on + λ·off = 損失")
            losses.append((on, off))
        r = meter.result('train', crit)
        self.assertAlmostEqual(r['train_on_diag'], sum(o for o, _ in losses) / 3, places=4)
        self.assertAlmostEqual(r['train_off_diag'], sum(f for _, f in losses) / 3, places=2)
        self.assertAlmostEqual(r['train_off_diag_weighted'], 5e-3 * r['train_off_diag'], places=6)

    def test_other_methods_report_nothing(self):
        crit = torch.nn.MSELoss()                 # PART_NAMES を持たない損失
        meter = LossPartsMeter(crit)
        meter.before(crit)
        crit(torch.randn(3), torch.randn(3))
        meter.add(crit)
        self.assertEqual(meter.result('train', crit), {})
        self.assertEqual(LossPartsMeter(None).result('val', None), {})   # MAE は criterion が None


if __name__ == "__main__":
    unittest.main()


class TestStepSummary(unittest.TestCase):
    def test_aggregates_and_skips_nonfinite(self):
        from lib.trainer.loop import summarize_steps
        cols = ['gn_raw', 'ln_gain_min', 'act_b0', 'act_b11', 'z_erank', 'unrelated']
        rows = [[1.0, 0.5, 2.0, 4.0, 30.0, 9.0], [3.0, 0.4, float('nan'), 6.0, 10.0, 9.0]]
        s = summarize_steps(cols, rows, [1.0, 5.0, float('nan')])
        self.assertEqual(s['steps/gn_raw_mean'], 2.0)
        self.assertEqual(s['steps/gn_raw_max'], 3.0)
        self.assertEqual(s['steps/ln_gain_min_min'], 0.4)
        self.assertEqual(s['steps/act_b0_mean'], 2.0, "NaN の step は除く")
        self.assertEqual(s['steps/act_b11_max'], 6.0, "最後の block も出す")
        self.assertEqual(s['steps/z_erank_min'], 10.0)
        self.assertEqual(s['steps/grad_norm_max'], 5.0)
        self.assertNotIn('steps/unrelated_mean', s)
