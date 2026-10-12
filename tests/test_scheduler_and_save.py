"""レビュー指摘(2026-10-12)の 3 点を固定する。コンテナ内で実行する(torch / timm が要る):
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_scheduler_and_save"

  1. 実効ランク: 完全崩壊(全サンプル同じ特徴)で最大値 d ではなく 0 を返す。丸め誤差だけのばらつきも 0。
  2. state.pt 等の保存: 保存中に失敗しても元のファイルが残る(一時ファイル + os.replace)。state.json は保存済み epoch。
  3. lr スケジュール: warmup を含めて num_epoch で減衰が終わり、epoch e は _get_lr(e) で走る(1 epoch 遅れない)。
"""
import json
import math
import os
import tempfile
import unittest
from unittest import mock

import torch

from lib.sslmodel.evaluation import SSLEvaluator
from lib.trainer.loop import atomic_torch_save, write_state_meta
from lib.trainer.model import build_scheduler


class TestEffectiveRank(unittest.TestCase):
    def test_complete_collapse_is_zero(self):
        z = torch.randn(1, 768).repeat(512, 1)
        self.assertEqual(SSLEvaluator._effective_rank(z)[0], 0.0, "旧実装はここで 768 を返していた")

    def test_rounding_noise_only_is_zero(self):
        z = torch.randn(1, 768).repeat(512, 1) + 1e-7 * torch.randn(512, 768)
        self.assertEqual(SSLEvaluator._effective_rank(z)[0], 0.0)

    def test_healthy_is_large(self):
        r = SSLEvaluator._effective_rank(torch.randn(512, 768))[0]
        self.assertGreater(r, 300)

    def test_low_rank_is_low(self):
        z = torch.randn(512, 3) @ torch.randn(3, 768) + 5.0
        self.assertLess(SSLEvaluator._effective_rank(z)[0], 3.5)


class TestAtomicSave(unittest.TestCase):
    def test_failure_keeps_previous(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'state.pt')
            atomic_torch_save({'epoch': 1}, path)
            real_save = torch.save

            def broken(obj, f, *a, **k):
                with open(f, 'wb') as h:
                    h.write(b'partial')
                raise KeyboardInterrupt('walltime')
            with mock.patch('torch.save', broken):
                with self.assertRaises(KeyboardInterrupt):
                    atomic_torch_save({'epoch': 2}, path)
            self.assertEqual(torch.load(path)['epoch'], 1, "直前の正常な state.pt が残る")
            torch.save = real_save

    def test_state_meta(self):
        with tempfile.TemporaryDirectory() as d:
            write_state_meta(d, completed_epochs=65, num_epoch=1600)
            m = json.load(open(os.path.join(d, 'state.json')))
        self.assertEqual((m['completed_epochs'], m['num_epoch']), (65, 1600))


class TestSchedule(unittest.TestCase):
    def _lrs(self, num_epoch, warmup_t, peak=1.0):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=peak)
        sch = build_scheduler(opt, num_epoch, warmup_t, 0.0, 1e-5)
        used = []
        for e in range(num_epoch):            # loop.py と同じ: epoch e を走らせてから step(e + 1)
            used.append(opt.param_groups[0]['lr'])
            sch.step(e + 1)
        return used, sch

    def test_simsiam_like(self):
        used, sch = self._lrs(160, 16)
        self.assertAlmostEqual(used[0], 1e-5, places=8)
        self.assertAlmostEqual(used[16], 1.0, places=6, msg="warmup 明けの epoch がピーク")
        self.assertLess(used[15], 1.0)
        self.assertLess(used[-1], 1e-3, "最終 epoch で lr がほぼ 0(旧実装は 3.1% 残っていた)")
        self.assertTrue(all(a >= b for a, b in zip(used[16:], used[17:])), "warmup 後は単調減少")
        for e in (0, 10, 100, 159):
            self.assertAlmostEqual(used[e], sch._get_lr(e)[0], places=9, msg="epoch e は _get_lr(e)(1 epoch 遅れない)")

    def test_mae_and_bt_like(self):
        for n, w in ((2568, 64), (1600, 16), (480, 10)):
            used, _ = self._lrs(n, w)
            self.assertAlmostEqual(used[w], 1.0, places=6)
            self.assertLess(used[-1], 1e-4)

    def test_no_warmup(self):
        used, _ = self._lrs(100, 0)
        self.assertAlmostEqual(used[0], 1.0, places=6)
        self.assertLess(used[-1], 1e-3)


if __name__ == "__main__":
    unittest.main()
