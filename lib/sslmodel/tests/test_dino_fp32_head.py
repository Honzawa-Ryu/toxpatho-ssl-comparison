# -*- coding: utf-8 -*-
"""`--dino_fp32_head` / `--dino_freeze_last_layer` の単体テスト。

守っているのは次の3点:

  1. fp32_head=True のとき、bf16 autocast の内側でもヘッド出力が fp32 になること。
     backbone だけ bf16 のままにするのが狙いなので、「全部 fp32 になった」でも
     「実は bf16 のままだった」でも困る（docs/HANDOFF_dino_next.md タスク2-B）。
  2. fp32 経路でも損失値が bf16 経路とほぼ一致すること（式を変えていないこと）。
  3. `cancel_last_layer_gradients` が epoch 0始まりで効くこと。公式 main_dino.py の
     `--freeze_last_layer` と同義にするため、N を指定したら epoch 0..N-1 が凍結。

CPU で走るが timm/torch が要るのでコンテナ内で実行すること（`pytest lib/`）。
ViT-tiny を使うので数秒で終わる。

Run:
    python -m unittest lib.sslmodel.tests.test_dino_fp32_head
"""
import unittest

import torch

from lib.sslmodel.models import dino


def _build(fp32, out_dim=128):
    torch.manual_seed(0)
    model = dino.DINO(backbone_name="vit_tiny_patch16_224", out_dim=out_dim,
                      n_global_crops=2, fp32_head=fp32)
    criterion = dino.DINOLoss(out_dim=out_dim, fp32=fp32)
    return model, criterion


def _forward(model, criterion, views):
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        student = model(views)
        teacher = model.forward_teacher(views)
        loss = criterion(student, teacher)
    return student, teacher, loss


class TestDinoFp32Head(unittest.TestCase):
    def setUp(self):
        self.views = [torch.randn(2, 3, 224, 224) for _ in range(2)]

    def test_default_keeps_bf16_head(self):
        """既定(fp32_head=False)は従来どおり bf16 autocast のまま。"""
        model, criterion = _build(False)
        student, teacher, loss = _forward(model, criterion, self.views)
        self.assertEqual(student[0].dtype, torch.bfloat16)
        self.assertEqual(teacher[0].dtype, torch.bfloat16)
        # log_softmax は autocast のfp32ポリシー対象なので、損失自体は元から fp32。
        # fp32_head が効くのはその手前(ヘッドの Linear と `t - center` の引き算)。
        self.assertEqual(loss.dtype, torch.float32)

    def test_fp32_head_promotes_head_output(self):
        model, criterion = _build(True)
        student, teacher, loss = _forward(model, criterion, self.views)
        self.assertEqual(student[0].dtype, torch.float32)
        self.assertEqual(teacher[0].dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)

    def test_fp32_head_keeps_backbone_in_bf16(self):
        """狙いは「ヘッドだけ fp32」。backbone まで fp32 に戻ると速度が出ない。

        backbone の *出力* は timm ViT 末尾の LayerNorm が autocast の fp32
        ポリシー対象なので fp32_head によらず元から fp32 になる（＝出力dtypeでは
        判定できない）。計算の主体である行列積が bf16 のままかを、最初のブロックの
        MLP 出力で見る。
        """
        model, criterion = _build(True)
        seen = []
        handle = model.student_backbone.blocks[0].mlp.fc1.register_forward_hook(
            lambda m, i, o: seen.append(o.dtype))
        try:
            _forward(model, criterion, self.views)
        finally:
            handle.remove()
        self.assertTrue(seen, "backbone のフックが呼ばれていない")
        self.assertEqual(set(seen), {torch.bfloat16})

    def test_fp32_head_promotes_head_linear(self):
        """ヘッド側の Linear は fp32 で回る（fp32_head の実効部分）。"""
        for fp32, want in ((False, torch.bfloat16), (True, torch.float32)):
            with self.subTest(fp32_head=fp32):
                model, criterion = _build(fp32)
                seen = []
                handle = model.student_head.last_layer.register_forward_hook(
                    lambda m, i, o: seen.append(o.dtype))
                try:
                    _forward(model, criterion, self.views)
                finally:
                    handle.remove()
                self.assertEqual(set(seen), {want})

    def test_backward_is_finite(self):
        model, criterion = _build(True)
        _, _, loss = _forward(model, criterion, self.views)
        loss.backward()
        grad = model.student_head.last_layer.weight_v.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_loss_matches_bf16_path(self):
        """精度を上げるだけで、損失の定義は変えていないこと。"""
        m0, c0 = _build(False)
        m1, c1 = _build(True)
        _, _, l0 = _forward(m0, c0, self.views)
        _, _, l1 = _forward(m1, c1, self.views)
        self.assertAlmostEqual(l0.item(), l1.item(), delta=0.1)


class TestFreezeLastLayerEpochs(unittest.TestCase):
    def test_freeze_window_is_zero_based(self):
        model, _ = _build(False)
        model.freeze_last_layer_epochs = 3
        params = list(model.student_head.last_layer.parameters())

        for epoch in (0, 2):
            for p in params:
                p.grad = torch.ones_like(p)
            model.cancel_last_layer_gradients(epoch)
            self.assertTrue(all(p.grad is None for p in params),
                            f"epoch {epoch} は凍結対象のはず")

        for p in params:
            p.grad = torch.ones_like(p)
        model.cancel_last_layer_gradients(3)
        self.assertTrue(all(p.grad is not None for p in params),
                        "epoch 3 は凍結解除されているはず")


if __name__ == "__main__":
    unittest.main()
