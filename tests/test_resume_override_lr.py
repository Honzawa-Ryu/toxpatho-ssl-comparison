"""--resume_override_lr（2026-10-09, exp 0032 用）のテスト。コンテナ内で実行する（torch / timm が要る）:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_resume_override_lr"

守っていること:
  1. 落とし穴の再現: state.pt(optimizer + timm scheduler)を復元すると、CLI で lr_bias を変えても
     保存時の値に戻る（optimizer.load_state_dict が param_groups の lr を、timm の load_state_dict が
     base_values を上書きする）。これが無ければオーバーライドは不要なので、落とし穴自体をテストで固定する。
  2. override_lr_after_resume の後は、bias グループだけ新しい lr_bias、重みグループは --lr のまま、
     scheduler.base_values も更新され、以後の scheduler.step でも元に戻らない。
"""
import types
import unittest

import torch
from timm.scheduler import CosineLRScheduler

from lib.trainer.optim import LARS, override_lr_after_resume


def _build(lr, lr_bias, num_epoch=1600, warmup_t=16, with_role=False):
    # with_role=False が既定: 0029 の state.pt は 'lr_role' を持たずに保存されている。
    # optimizer.load_state_dict は param_groups の dict を保存時のもので置き換えるので、
    # 現在の構築で付けた 'lr_role' は復元後に消える(0032 が踏んだ落とし穴)。
    w = torch.nn.Parameter(torch.randn(4, 4))
    b = torch.nn.Parameter(torch.randn(4))
    bias_group = {'params': [b], 'lr': lr_bias, 'weight_decay': 0.0, 'wd_exempt': True,
                  'weight_decay_filter': True, 'lars_adaptation_filter': True}
    if with_role:
        bias_group['lr_role'] = 'bias'
    opt = LARS([{'params': [w]}, bias_group], lr=lr, weight_decay=1.5e-6)
    sch = CosineLRScheduler(opt, t_initial=num_epoch, lr_min=0.0, warmup_t=warmup_t,
                            warmup_lr_init=0.0, warmup_prefix=True)
    return opt, sch


class TestResumeOverrideLr(unittest.TestCase):
    def _saved_state(self):
        opt, sch = _build(1.6, 0.0384)
        for e in range(64):
            sch.step(e)
        return opt.state_dict(), sch.state_dict()

    def test_pitfall_resume_reverts_cli_lr_bias(self):
        o_sd, s_sd = self._saved_state()
        opt, sch = _build(1.6, 0.0096)              # CLI で lr_bias を 1/4 にしたつもり
        opt.load_state_dict(o_sd)
        sch.load_state_dict(s_sd)
        self.assertAlmostEqual(sch.base_values[1], 0.0384, msg="timm の load_state_dict が base_values を戻す")
        sch.step(64)
        self.assertGreater(opt.param_groups[1]['lr'], 0.038, "再開後の bias lr は保存時の 0.0384 に戻る(落とし穴)")

    def test_saved_state_without_lr_role_drops_the_key(self):
        # 0032 の事故の再現: 保存側に 'lr_role' が無いと、復元後の param_groups からも消える
        o_sd, s_sd = self._saved_state()
        opt, sch = _build(1.6, 0.0096, with_role=True)
        self.assertEqual(opt.param_groups[1].get('lr_role'), 'bias')
        opt.load_state_dict(o_sd)
        self.assertIsNone(opt.param_groups[1].get('lr_role'), "復元で自前のキーは消える")
        sch.load_state_dict(s_sd)
        args = types.SimpleNamespace(lr=1.6, lr_bias=0.0096, layer_wise_lr=False, fix_pred_lr=False)
        override_lr_after_resume(args, opt, sch, start_epoch=64, log=lambda m: None)
        self.assertEqual(sch.base_values, [1.6, 0.0096], "キーに頼らず params の形で bias グループを見分ける")
        self.assertLess(opt.param_groups[1]['lr'], 0.0097)

    def test_override_applies_and_survives_scheduler_step(self):
        o_sd, s_sd = self._saved_state()
        opt, sch = _build(1.6, 0.0096, with_role=True)
        opt.load_state_dict(o_sd)
        sch.load_state_dict(s_sd)
        args = types.SimpleNamespace(lr=1.6, lr_bias=0.0096, layer_wise_lr=False, fix_pred_lr=False)
        msgs = []
        now = override_lr_after_resume(args, opt, sch, start_epoch=64, log=msgs.append)
        self.assertEqual(sch.base_values, [1.6, 0.0096])
        self.assertAlmostEqual(now[0], 1.6 * sch._get_lr(63)[0] / 1.6, places=9)
        self.assertLess(now[1], 0.0097)
        self.assertAlmostEqual(now[1] / now[0], 0.0096 / 1.6, places=9, msg="同じ cosine 位置で比が lr_bias/lr")
        sch.step(64)
        self.assertLess(opt.param_groups[1]['lr'], 0.0097, "以後の scheduler.step でも戻らない")
        self.assertGreater(opt.param_groups[0]['lr'], 1.59, "重みグループは --lr のまま")
        self.assertTrue(msgs and '--resume_override_lr' in msgs[0])

    def test_refuses_layer_wise_lr(self):
        opt, sch = _build(1.6, 0.0384)
        args = types.SimpleNamespace(lr=1.6, lr_bias=0.0096, layer_wise_lr=True, fix_pred_lr=False)
        with self.assertRaises(RuntimeError):
            override_lr_after_resume(args, opt, sch, 64, print)


if __name__ == "__main__":
    unittest.main()
