"""--step_log(2026-10-12, 全手法の step テレメトリ)のテスト。コンテナ内で実行する(torch が要る):
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_step_log"

守っていること:
  1. SimSiam の崩壊統計: 健全(ランダム)と崩壊(全サンプルが同じ方向)を z_erank / cos_zz で見分けられる。
     出力 BN が次元ごとの std を揃えるので、std 系の指標だけでは見逃す(だから erank を足した)。
  2. MAE はパラメータ名に backbone. 接頭辞が無い。エンコーダ / デコーダで gn_backbone / gn_head を分ける。
  3. 書き出しは BT 以外 steps_ep*、統計名は手法ごと。
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from lib.sslmodel.sslutils import SimSiam
from lib.trainer import bt_telemetry


class TestSimSiamStats(unittest.TestCase):
    def test_names_match(self):
        z = torch.randn(64, 32)
        st = SimSiam._stats(z + 0.1, z - 0.1, z, z)
        self.assertEqual(len(st), len(bt_telemetry.SIMSIAM_STAT_NAMES))

    def test_rank_collapse_is_visible(self):
        torch.manual_seed(0)
        healthy = dict(zip(bt_telemetry.SIMSIAM_STAT_NAMES,
                           SimSiam._stats(torch.randn(256, 64), torch.randn(256, 64),
                                          torch.randn(256, 64), torch.randn(256, 64)).tolist()))
        v = torch.randn(64)
        s1, s2 = torch.randn(256, 1), torch.randn(256, 1)
        z1, z2 = s1 * v, s2 * v                       # 全サンプルが 1 方向(ランク 1)
        z1 = (z1 - z1.mean(0)) / z1.std(0)            # 出力 BN(affine なし)で次元ごとに標準化
        z2 = (z2 - z2.mean(0)) / z2.std(0)
        collapsed = dict(zip(bt_telemetry.SIMSIAM_STAT_NAMES, SimSiam._stats(z1, z2, z1, z2).tolist()))
        self.assertGreater(healthy["z_erank"], 30)
        self.assertLess(collapsed["z_erank"], 2.0)
        self.assertAlmostEqual(collapsed["z_std_min"], 1.0, places=2, msg="std 系は BN で 1 のまま = 見逃す")
        self.assertGreater(abs(collapsed["cos_zz"]), 0.0)   # 符号は s1·s2 次第、大きさで見る


class TestProbeGroupingAndWrite(unittest.TestCase):
    def test_mae_encoder_decoder_split(self):
        m = torch.nn.ModuleDict({
            "blocks": torch.nn.Linear(4, 4), "decoder_pred": torch.nn.Linear(4, 4)})
        m.mask_token = torch.nn.Parameter(torch.zeros(1, 1, 4))
        out = m["decoder_pred"](m["blocks"](torch.randn(3, 4))) + m.mask_token.sum()
        out.sum().backward()
        probe = bt_telemetry.GradGroupProbe(m)
        params = [p for p in m.parameters() if p.grad is not None]
        norms = torch._foreach_norm([p.grad for p in params])
        probe.record(params, norms)
        row = dict(zip(bt_telemetry.GROUP_NAMES, probe.rows()[0]))
        enc = math.sqrt(sum(p.grad.norm().item() ** 2 for p in m["blocks"].parameters()))
        self.assertAlmostEqual(row["gn_backbone"], enc, places=5)
        self.assertGreater(row["gn_head"], 0)

    def test_write_prefix_and_stat_names(self):
        with tempfile.TemporaryDirectory() as d:
            stats = [torch.arange(len(bt_telemetry.SIMSIAM_STAT_NAMES), dtype=torch.float32)]
            bt_telemetry.write_epoch(d, 3, [-0.5], [1.0], stats, stat_names=bt_telemetry.SIMSIAM_STAT_NAMES,
                                     prefix="steps")
            self.assertEqual(len(list(Path(d).glob("bt_steps_*"))), 0)
            row = json.loads(next(Path(d).glob("steps_ep0003_*.jsonl")).read_text())
        self.assertEqual(row["z_erank"], 5.0)
        self.assertNotIn("on_diag", row)


if __name__ == "__main__":
    unittest.main()
