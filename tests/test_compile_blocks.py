"""--compile_blocks(2026-10-13)。コンテナ内(CPU の inductor で動く):
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_compile_blocks"

  - 出力と勾配がコンパイル前と一致する
  - state_dict のキーが変わらない(既存のチェックポイントと互換)
  - step テレメトリの block 出力のフック(StepProbe)がコンパイル後も動く
"""
import copy
import unittest

import timm
import torch

from lib.trainer import bt_telemetry
from lib.trainer.model import compile_blocks


def _tiny():
    vit = timm.create_model("vit_tiny_patch16_224", num_classes=0, depth=2, img_size=32)
    m = torch.nn.Module()
    m.backbone = vit
    return m


class TestCompileBlocks(unittest.TestCase):
    def test_equal_outputs_keys_and_hooks(self):
        torch.manual_seed(0)
        eager = _tiny()
        comp = copy.deepcopy(eager)
        keys = list(eager.state_dict().keys())
        self.assertEqual(compile_blocks(comp), 2)
        self.assertEqual(list(comp.state_dict().keys()), keys, "state_dict のキーが変わらない")
        x = torch.randn(4, 3, 32, 32)
        ye, yc = eager.backbone(x), comp.backbone(x)
        torch.testing.assert_close(yc, ye, rtol=1e-4, atol=1e-4)
        ye.sum().backward()
        yc.sum().backward()
        for (n, a), (_, b) in zip(eager.named_parameters(), comp.named_parameters()):
            torch.testing.assert_close(b.grad, a.grad, rtol=1e-3, atol=1e-4, msg=n)
        probe = bt_telemetry.StepProbe(comp)
        self.assertIn("act_b1", probe.act_names)
        probe.active = True
        comp.backbone(x)
        probe.active = False
        self.assertEqual(int(probe._act_cnt.sum()), len(probe.act_names), "フックがコンパイル後も呼ばれる")
        probe.close()


if __name__ == "__main__":
    unittest.main()
