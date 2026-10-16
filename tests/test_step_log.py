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


class TestStepProbeDynamics(unittest.TestCase):
    """更新量・重み・LN ゲイン・block 出力(2026-10-12)。"""

    def _model(self):
        class Blk(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc, self.norm1 = torch.nn.Linear(4, 4), torch.nn.LayerNorm(4)

            def forward(self, x):
                return self.norm1(self.fc(x))

        class BB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = torch.nn.Linear(4, 4)
                self.blocks = torch.nn.ModuleList([Blk(), Blk()])

            def forward(self, x):
                x = self.patch_embed(x)
                for b in self.blocks:
                    x = b(x)
                return x
        m = torch.nn.Module()
        m.backbone, m.projection_head = BB(), torch.nn.Linear(4, 2)
        m.forward = lambda x: m.projection_head(m.backbone(x))
        return m

    def test_updates_activations_and_hooks_removed(self):
        torch.manual_seed(0)
        m = self._model()
        probe = bt_telemetry.StepProbe(m)
        self.assertEqual(probe.act_names, ("act_patch", "act_b0", "act_b1"))
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        x = torch.randn(8, 4)
        probe.active = True
        out = m.forward(x)
        out = out + m.forward(x)           # 2 ビュー相当: 2 回 forward
        probe.active = False
        out.sum().backward()
        params = [p for p in m.parameters() if p.grad is not None]
        probe.record(params, torch._foreach_norm([p.grad for p in params]))
        before = [p.detach().clone() for p in params]
        probe.before_step(params)
        opt.step()
        probe.after_step(params)
        row = dict(zip(probe.column_names, probe.rows()[0]))
        dn = [(p.detach() - b) for p, b in zip(params, before)]
        upd_w = math.sqrt(sum(d.norm().item() ** 2 for d, p in zip(dn, params) if p.ndim > 1))
        self.assertAlmostEqual(row["upd_w"], upd_w, places=5)
        self.assertGreater(row["act_b1"], 0.5, "LayerNorm 出力の RMS は ≈1")
        gains = [b.norm1.weight.detach().mean().item() for b in m.backbone.blocks]
        self.assertAlmostEqual(row["ln_gain_mean"], sum(gains) / 2, places=5, msg="backbone の LN ゲインだけ(head は含まない)")
        self.assertAlmostEqual(row["ln_gain_min"], min(gains), places=5)
        m.forward(x)                       # active でない forward は数えない
        self.assertEqual(float(probe._act_cnt.sum()), 0.0)
        probe.close()
        self.assertEqual(len(probe._handles), 0)

    def test_row_padded_when_step_skipped(self):
        m = self._model()
        probe = bt_telemetry.StepProbe(m)
        m.forward(torch.randn(3, 4)).sum().backward()
        params = [p for p in m.parameters() if p.grad is not None]
        probe.record(params, torch._foreach_norm([p.grad for p in params]))   # after_step 無し(非有限で抜けた)
        row = probe.rows()[0]
        self.assertEqual(len(row), len(probe.column_names))
        self.assertTrue(math.isnan(row[-1]))
        probe.close()


class TestSimSiamProjectorWidth(unittest.TestCase):
    """--simsiam_proj_hidden(2026-10-16): projector の隠れ層の幅。既定は backbone 次元(従来どおり)。"""

    def test_hidden_dim(self):
        from lib.sslmodel.models.simsiam import SimSiam
        bb = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 768))
        default = SimSiam(bb, head_size=768, dim=2048, pred_dim=512)
        wide = SimSiam(bb, head_size=768, dim=2048, pred_dim=512, hidden_dim=2048)
        self.assertEqual(default.fc[0].out_features, 768, "既定は従来どおり backbone 次元")
        self.assertEqual(wide.fc[0].out_features, 2048)
        self.assertEqual(wide.fc[3].in_features, 2048)
        self.assertEqual(wide.fc[6].out_features, 2048, "出力次元は変えない")
        p1, p2, z1, z2 = wide(torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8))
        self.assertEqual(tuple(p1.shape), (4, 2048))
        self.assertEqual(list(default.state_dict().keys()), list(wide.state_dict().keys()), "キーは同じ(形だけ違う)")
