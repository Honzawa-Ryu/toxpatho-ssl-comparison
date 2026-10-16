"""BT の val を学習と同じ条件(batch 統計の BN・running 統計は更新しない)で測る(2026-10-16)。コンテナ内:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_val_batch_stats"
"""
import types
import unittest

import torch

from lib.sslmodel.models.barlowtwins import BarlowTwins, BarlowTwinsLoss
from lib.trainer.loop import batch_stat_bn, compute_val_loss


def _bt():
    bb = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 16))
    return BarlowTwins(bb, head_size=[16, 32, 32]), BarlowTwinsLoss(lambda_param=5e-3)


class _SSL:
    def calc_loss(self, model, data, criterion):
        return criterion(model(data[0]), model(data[1]))


class TestValBatchStats(unittest.TestCase):
    def test_batch_stat_bn_keeps_running_stats_and_restores_state(self):
        model, _ = _bt()
        model.eval()
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        for b in bns:                      # running 統計を適当な値にしておく
            b.running_mean.fill_(3.0); b.running_var.fill_(7.0)
        before = [(b.running_mean.clone(), b.running_var.clone(), b.momentum) for b in bns]
        x = torch.randn(32, 3, 8, 8)
        with batch_stat_bn(model):
            self.assertTrue(model.training)
            y_batch = model(x)
        for b, (rm, rv, mom) in zip(bns, before):
            torch.testing.assert_close(b.running_mean, rm, msg="running_mean が更新されていない")
            torch.testing.assert_close(b.running_var, rv)
            self.assertEqual(b.momentum, mom, "momentum が元に戻る")
        self.assertFalse(model.training, "モードが元に戻る")
        model.eval()
        y_eval = model(x)
        self.assertFalse(torch.allclose(y_batch, y_eval), "batch 統計と running 統計で出力が違う(= 測り方が変わっている)")

    def test_compute_val_loss_uses_batch_stats_for_bt(self):
        model, crit = _bt()
        model.train()
        ctx = types.SimpleNamespace(ssl_class=_SSL())
        data = [(torch.randn(32, 3, 8, 8), torch.randn(32, 3, 8, 8)) for _ in range(3)]
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
        rm0 = [b.running_mean.clone() for b in bns]
        v = compute_val_loss(ctx, model, crit, data, max_batches=2)
        self.assertTrue(torch.isfinite(torch.tensor(v)))
        self.assertTrue(model.training, "呼ぶ前のモードに戻る")
        for b, rm in zip(bns, rm0):
            torch.testing.assert_close(b.running_mean, rm, msg="val で running 統計が動かない")
        self.assertIn('val_on_diag', ctx.val_loss_parts)
        # batch 統計を使わない損失(属性なし)は従来どおり eval モードで測る
        crit2 = torch.nn.MSELoss()

        class _S2:
            def calc_loss(self, model, data, criterion):
                return criterion(model(data[0]), model(data[1]))
        ctx2 = types.SimpleNamespace(ssl_class=_S2())
        seen = []
        orig = model.forward
        model.forward = lambda x: (seen.append(model.training), orig(x))[1]
        compute_val_loss(ctx2, model, crit2, data, max_batches=1)
        self.assertEqual(seen, [False, False], "BT 以外は eval モード")


if __name__ == "__main__":
    unittest.main()
