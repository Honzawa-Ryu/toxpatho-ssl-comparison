"""--resume で WandB の run をつなぐ(lib/trainer/entry.py:init_wandb、2026-10-14)。コンテナ内・WANDB_MODE=offline で実行:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && WANDB_MODE=offline PYTHONPATH=. .venv/bin/python -m unittest tests.test_wandb_resume"
外部には送らない(offline のまま、ディレクトリは一時ディレクトリ)。
"""
import os
import tempfile
import types
import unittest

os.environ["WANDB_MODE"] = "offline"
import wandb  # noqa: E402

from lib.trainer.entry import init_wandb  # noqa: E402


def _args(resume):
    return types.SimpleNamespace(note="t", ssl_name="mae", model_name="ViTB16", lr=1e-3, num_epoch=10, patience=7,
                                 delta=0, freeze_backbone=False, resume=resume)


class TestWandbResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["WANDB_DIR"] = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_resume_reuses_id_only_with_state(self):
        d = self.tmp.name
        r1 = init_wandb(_args(resume=True), d)            # 初回: state.pt 無し → 新しい id を保存
        first = r1.id
        r1.finish()
        self.assertEqual(open(os.path.join(d, "wandb_run_id.txt")).read(), first)
        r2 = init_wandb(_args(resume=True), d)            # state.pt 無し → まだ再開ではない(新しい run)
        self.assertNotEqual(r2.id, first)
        second = r2.id
        r2.finish()
        open(os.path.join(d, "state.pt"), "w").close()
        r3 = init_wandb(_args(resume=True), d)            # state.pt あり → 保存した id で再開
        self.assertEqual(r3.id, second)
        r3.finish()
        r4 = init_wandb(_args(resume=False), d)           # --resume なし → 新しい run
        self.assertNotEqual(r4.id, second)
        r4.finish()


if __name__ == "__main__":
    unittest.main()
