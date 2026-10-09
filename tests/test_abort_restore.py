"""abort 時の重みの復元元(lib/trainer/loop.py:restore_healthy_weights, 2026-10-09 レビュー指摘 P2)。
コンテナ内で実行する(torch が要る):
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_abort_restore"

別ディレクトリへ state.pt だけコピーして再開した診断ランには checkpoint.pt が無い。その場合に
state.pt の model_state_dict(最後の健全 epoch)へ戻れること、checkpoint.pt があればそちらが優先されること、
どちらも無ければ None を返すこと(呼び出し側が NaN 重みの書き出しを拒否する)を固定する。
"""
import logging
import os
import tempfile
import unittest

import torch

from lib.trainer.loop import restore_healthy_weights


def _net(fill):
    m = torch.nn.Linear(3, 2)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(fill)
    return m


class TestRestoreHealthyWeights(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger("t")

    def test_falls_back_to_state_pt_when_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            torch.save({'epoch': 63, 'model_state_dict': _net(2.0).state_dict(), 'criterion': object()},
                       os.path.join(d, 'state.pt'))
            m = _net(float('nan'))
            src = restore_healthy_weights(m, os.path.join(d, 'checkpoint.pt'), d, self.log)
        self.assertEqual(src, 'state.pt')
        self.assertTrue(all(torch.all(p == 2.0) for p in m.parameters()))

    def test_checkpoint_takes_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            torch.save({'epoch': 63, 'model_state_dict': _net(2.0).state_dict()}, os.path.join(d, 'state.pt'))
            torch.save(_net(3.0).state_dict(), os.path.join(d, 'checkpoint.pt'))
            m = _net(float('nan'))
            src = restore_healthy_weights(m, os.path.join(d, 'checkpoint.pt'), d, self.log)
        self.assertEqual(src, 'checkpoint.pt')
        self.assertTrue(all(torch.all(p == 3.0) for p in m.parameters()))

    def test_none_when_nothing_exists(self):
        with tempfile.TemporaryDirectory() as d:
            m = _net(1.0)
            self.assertIsNone(restore_healthy_weights(m, os.path.join(d, 'checkpoint.pt'), d, self.log))
        self.assertTrue(all(torch.all(p == 1.0) for p in m.parameters()), "何も無ければ重みを触らない")


if __name__ == "__main__":
    unittest.main()
