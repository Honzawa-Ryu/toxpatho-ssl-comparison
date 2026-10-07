# -*- coding: utf-8 -*-
"""`--bt_fp32_head` / `--bt_step_log` の単体テスト（exp 0029 が ep65 で NaN になった件への対応）。

守っているのは次の点:

  1. **仮説 H1 の機構が数値として成立すること**: 各次元に大きな共通オフセットがあると、bf16 では
     バッチ内の値が同じ値に丸められて std が厳密に 0 になり、`(z-mean)/std` が 0/0 で損失が NaN になる。
     fp32 経路なら有限のまま。⚠️ これは「この機構がありうる」ことの確認であって、0029 の NaN が
     実際にこれだったことの証明ではない（その切り分けが --bt_step_log の目的）。
  2. fp32_head=True のとき、bf16 autocast の内側でもヘッド出力と損失が fp32 になること。
     False のときは従来どおり bf16 のままであること（既存の実験の挙動を変えない）。
  3. fp32 経路でも式を変えていないこと（損失値が bf16 経路とほぼ一致）。
  4. step 単位の統計(`last_stats`)が学習に影響せず、std==0 の次元を検出できること。
  5. 旧 state.pt から復元した criterion（新しい属性を持たない）でも動くこと。
     0029 の ep64 の state.pt から再開する診断ランで実際に起きる経路。

CPU で走るがコンテナ内で実行すること（torch が要る）。ヘッドは小さい次元で作るので数秒で終わる。

Run:
    python -m unittest lib.sslmodel.tests.test_barlowtwins_fp32
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from lib.sslmodel.models import barlowtwins
from lib.trainer import bt_telemetry


def _model(fp32, in_dim=16, proj=32):
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Linear(8, in_dim), nn.LayerNorm(in_dim))  # 末尾 LayerNorm = timm ViT と同じ
    model = barlowtwins.BarlowTwins(backbone, head_size=[in_dim, proj, proj], fp32_head=fp32)
    criterion = barlowtwins.BarlowTwinsLoss(fp32=fp32)
    return model, criterion


def _forward(model, criterion, n=64):
    torch.manual_seed(1)
    x1, x2 = torch.randn(n, 8), torch.randn(n, 8)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        z1, z2 = model(x1), model(x2)
        loss = criterion(z1, z2)
    return z1, z2, loss


class TestBf16Hazard(unittest.TestCase):
    """仮説 H1 の機構: オフセットが大きいと bf16 の std が 0 になる。"""

    def _offset_z(self, spread=0.1):
        # オフセット 300・ばらつき 0.1（比 3000）。bf16 の刻みは 256〜512 で 2 なので、値が
        # 刻みの半分(±1)を超えない限り全て 300 に丸まる。
        # ⚠️ 条件は厳しい: ばらつき 0.5（比 600）だと 2σ を超える ~5% の値が 298/302 に丸まって
        #    std は 0 にならない(粗く量子化されるだけ)。0029 でこの条件が満たされたかは未検証。
        torch.manual_seed(0)
        return torch.randn(256, 64) * spread + 300.0

    def test_bf16_collapses_std_to_zero_and_loss_is_nan(self):
        z = self._offset_z().bfloat16()          # bf16 の刻みは 256〜512 で 2 → 全値が 300 に丸まる
        self.assertTrue(bool((z.std(0) == 0).all()), "オフセット 300 では bf16 の std が 0 になるはず")
        loss = barlowtwins.BarlowTwinsLoss(fp32=False)(z, z)
        self.assertTrue(torch.isnan(loss).item(), "std==0 で 0/0 になり損失が NaN になる")

    def test_moderate_spread_is_only_coarsely_quantized_not_zero(self):
        # 比 600 ではまだ std は 0 にならない(= NaN にならない)が、値は数段階にしか分かれない。
        z = self._offset_z(spread=0.5).bfloat16()
        self.assertTrue(bool((z.std(0) > 0).all()))
        self.assertLessEqual(len(torch.unique(z[:, 0])), 5)

    def test_fp32_path_stays_finite(self):
        z = self._offset_z()
        loss = barlowtwins.BarlowTwinsLoss(fp32=True)(z, z)
        self.assertTrue(torch.isfinite(loss).item())

    def test_fp32_flag_rescues_what_bf16_input_cannot(self):
        # 損失の fp32 化は「bf16 に丸められた z」を元に戻せない。ヘッドごと fp32 にする
        # 必要がある理由(BarlowTwins.fp32_head)を明示するテスト。
        z_bf16 = self._offset_z().bfloat16()
        loss = barlowtwins.BarlowTwinsLoss(fp32=True)(z_bf16, z_bf16)
        self.assertTrue(torch.isnan(loss).item(), "丸め済みの入力は損失側で fp32 化しても救えない")


class TestFp32Head(unittest.TestCase):
    def test_default_keeps_bf16(self):
        model, crit = _model(fp32=False)
        z1, _, loss = _forward(model, crit)
        self.assertEqual(z1.dtype, torch.bfloat16)
        self.assertEqual(loss.dtype, torch.bfloat16, "既存の実験の挙動(bf16 の損失)を変えないこと")

    def test_fp32_head_promotes_head_output_and_loss(self):
        model, crit = _model(fp32=True)
        z1, _, loss = _forward(model, crit)
        self.assertEqual(z1.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)

    def test_backbone_stays_bf16_under_autocast(self):
        # backbone だけ bf16 のままにするのが狙い。「全部 fp32」でも困る。
        model, _ = _model(fp32=True)
        seen = {}
        def hook(module, inputs, output):
            seen.setdefault("dtype", output.dtype)   # 値を返すと出力が置き換わるので None を返す
        model.backbone[0].register_forward_hook(hook)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            model(torch.randn(8, 8))
        self.assertEqual(seen["dtype"], torch.bfloat16)

    def test_loss_matches_bf16_path(self):
        m32, c32 = _model(fp32=True)
        m16, c16 = _model(fp32=False)
        m16.load_state_dict(m32.state_dict())
        _, _, l32 = _forward(m32, c32)
        _, _, l16 = _forward(m16, c16)
        self.assertAlmostEqual(l32.item(), l16.float().item(), delta=0.05 * abs(l32.item()))

    def test_state_dict_is_identical_so_checkpoints_are_compatible(self):
        a, _ = _model(fp32=True)
        b, _ = _model(fp32=False)
        self.assertEqual(list(a.state_dict().keys()), list(b.state_dict().keys()))

    def test_backward_is_finite(self):
        model, crit = _model(fp32=True)
        _, _, loss = _forward(model, crit)
        loss.backward()
        for name, p in model.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all().item(), name)


class TestStats(unittest.TestCase):
    def test_stats_do_not_change_the_loss(self):
        m, c0 = _model(fp32=True)
        c1 = barlowtwins.BarlowTwinsLoss(fp32=True, collect_stats=True)
        _, _, l0 = _forward(m, c0)
        _, _, l1 = _forward(m, c1)
        self.assertEqual(l0.item(), l1.item())

    def test_stats_shape_and_names(self):
        m, _ = _model(fp32=True)
        c = barlowtwins.BarlowTwinsLoss(fp32=True, collect_stats=True)
        _forward(m, c)
        self.assertEqual(tuple(c.last_stats.shape), (len(bt_telemetry.STAT_NAMES),))
        self.assertTrue(torch.isfinite(c.last_stats).all().item())

    def test_off_by_default(self):
        m, c = _model(fp32=True)
        _forward(m, c)
        self.assertIsNone(c.last_stats)

    def test_detects_zero_std_dimensions(self):
        c = barlowtwins.BarlowTwinsLoss(fp32=False, collect_stats=True)
        z = (torch.randn(128, 16) * 0.1 + 300.0).bfloat16()
        c(z, z)
        stats = dict(zip(bt_telemetry.STAT_NAMES, c.last_stats.tolist()))
        self.assertEqual(stats["n_zero_std"], 32.0, "両ビューの 16 次元 x 2 が全て std==0")
        self.assertEqual(stats["z_std_min"], 0.0)
        self.assertGreater(stats["z_ratio_max"], 1e6, "std==0 なら オフセット/ばらつき比 は極端に大きい")
        self.assertAlmostEqual(stats["z_mean_abs_max"], 300.0, delta=2.0)


class TestOldStateCompat(unittest.TestCase):
    def test_criterion_unpickled_without_new_attributes_still_works(self):
        # 旧 state.pt の criterion は __init__ を通らず fp32 / collect_stats / last_stats を持たない。
        old = barlowtwins.BarlowTwinsLoss.__new__(barlowtwins.BarlowTwinsLoss)
        torch.nn.Module.__init__(old)
        old.lambda_param, old.gather_distributed = 5e-3, False
        self.assertFalse(old.fp32)
        self.assertFalse(old.collect_stats)
        self.assertIsNone(old.last_stats)
        z = torch.randn(32, 8)
        self.assertTrue(torch.isfinite(old(z, z.flip(0))).item())

    def test_flags_can_be_reapplied_after_restore(self):
        old = barlowtwins.BarlowTwinsLoss.__new__(barlowtwins.BarlowTwinsLoss)
        torch.nn.Module.__init__(old)
        old.lambda_param, old.gather_distributed = 5e-3, False
        old.fp32, old.collect_stats = True, True   # entry.py の resume 経路が行う設定
        z = torch.randn(32, 8)
        self.assertEqual(old(z.bfloat16(), z.bfloat16()).dtype, torch.float32)
        self.assertIsNotNone(old.last_stats)


class TestTelemetryFile(unittest.TestCase):
    def test_write_epoch_roundtrip_with_nonfinite(self):
        stats = [torch.tensor([0.5, 0.0, 3.0, 6.0, 0.9, 10.0, 20.0]),
                 torch.tensor([0.0, 4.0, 300.0, 1e30, float("nan"), float("nan"), float("nan")])]
        with tempfile.TemporaryDirectory() as d:
            bt_telemetry.write_epoch(d, 65, [272.0, float("nan")], [10.9, float("nan")], stats,
                                     first_nonfinite=1)
            files = list(Path(d).glob("bt_steps_ep0065_*.jsonl"))
            self.assertEqual(len(files), 1)
            rows = [json.loads(l) for l in files[0].read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["loss"], 272.0)
        self.assertEqual(rows[0]["z_std_min"], 0.5)
        self.assertEqual(rows[1]["loss"], "nan", "非有限は文字列で残す(JSON は NaN を表せない)")
        self.assertTrue(rows[1]["first_nonfinite"])
        self.assertEqual(rows[1]["n_zero_std"], 4.0)
        self.assertNotIn("first_nonfinite", rows[0])


class TestGradGroupProbe(unittest.TestCase):
    """グループ別 grad_norm(2026-10-07)。合計ノルムと一致し、4 分割が正しいこと。"""

    def _model(self):
        return torch.nn.ModuleDict({
            "backbone": torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3)),
            "projection_head": torch.nn.Linear(3, 2),
        })

    def test_groups_match_total_norm_and_names(self):
        model = self._model()
        out = model["projection_head"](model["backbone"](torch.randn(5, 4)))
        out.sum().backward()
        probe = bt_telemetry.GradGroupProbe(model)
        params = [p for p in model.parameters() if p.grad is not None]
        norms = torch._foreach_norm([p.grad for p in params])
        probe.record(params, norms)
        row = dict(zip(bt_telemetry.GROUP_NAMES, probe.rows()[0]))
        total = torch.norm(torch.stack(norms)).item()
        self.assertAlmostEqual(math.hypot(row["gn_lars"], row["gn_raw"]), total, places=5)
        self.assertAlmostEqual(math.hypot(row["gn_backbone"], row["gn_head"]), total, places=5)
        raw = math.sqrt(sum(p.grad.norm().item() ** 2 for p in params if p.ndim <= 1))
        self.assertAlmostEqual(row["gn_raw"], raw, places=5)
        head = math.sqrt(sum(p.grad.norm().item() ** 2 for p in model["projection_head"].parameters()))
        self.assertAlmostEqual(row["gn_head"], head, places=5)
        top1 = max(range(len(params)), key=lambda i: norms[i].item())
        self.assertEqual(int(row["top1_idx"]), top1)
        self.assertEqual(probe.names[top1], [n for n, _ in model.named_parameters()][top1])
        self.assertGreaterEqual(row["top1_norm"], row["top2_norm"])
        self.assertGreaterEqual(row["top2_norm"], row["top3_norm"])

    def test_module_prefix_is_stripped_and_mismatch_gives_nan(self):
        model = self._model()
        wrapped = torch.nn.ModuleDict({"module": model})   # DDP の "module." 接頭辞を模す
        probe = bt_telemetry.GradGroupProbe(wrapped)
        self.assertTrue(all(not n.startswith("module.") for n in probe.names))
        self.assertTrue(probe.names[0].startswith("backbone."))
        params = list(model.parameters())[:2]               # 集合が食い違う
        probe.record(params, [torch.tensor(1.0), torch.tensor(2.0)])
        self.assertTrue(all(math.isnan(v) for v in probe.rows()[0]))

    def test_write_epoch_with_groups(self):
        groups = [[1.0, 2.0, 3.0, 4.0, 5, 0.9, 6, 0.8, 7, 0.7]]
        with tempfile.TemporaryDirectory() as d:
            bt_telemetry.write_param_names(d, ["a", "b"])
            bt_telemetry.write_param_names(d, ["zzz"])     # 2 回目は上書きしない
            bt_telemetry.write_epoch(d, 66, [1.0, 2.0], [0.1, 0.2], [], groups=groups)
            rows = [json.loads(l) for l in next(Path(d).glob("bt_steps_ep0066_*.jsonl")).read_text().splitlines()]
            self.assertEqual(json.loads((Path(d) / "bt_param_names.json").read_text()), ["a", "b"])
        self.assertEqual(rows[0]["gn_raw"], 2.0)
        self.assertEqual(rows[0]["top1_idx"], 5)
        self.assertNotIn("gn_raw", rows[1], "groups が無い step には列を足さない")


if __name__ == "__main__":
    unittest.main()
