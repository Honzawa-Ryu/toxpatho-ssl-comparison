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
