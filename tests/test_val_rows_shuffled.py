"""val の行は固定 seed で並べ替える(2026-10-16)。純 Python(numpy のみ): python3 tests/test_val_rows_shuffled.py
memmap の行は WSI ごとに連続なので、並べ替えないと先頭 2,048 枚が 3 スライドしか含まない。"""
import ast
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(REPO, "lib/trainer/data.py"), encoding="utf-8").read()
assert "val_rows = np.random.RandomState(getattr(args, 'seed', 0)).permutation(val_rows)" in src, "val の並べ替えが無い"
i = src.index("permutation(val_rows)")
assert i < src.index("eval_dataset = MemmapPatchDataset"), "並べ替えは eval_dataset を作る前"
# 1,000 行/WSI × 200 WSI を模した行で、先頭 2,048 枚に含まれるスライド数が増えることを確かめる
rows = np.arange(200_000)
wsi = rows // 1000
head = wsi[rows[:2048]]
perm = np.random.RandomState(42).permutation(rows)
head_perm = wsi[perm[:2048]]
assert len(np.unique(head)) == 3, len(np.unique(head))
assert len(np.unique(head_perm)) > 190, len(np.unique(head_perm))
assert (np.random.RandomState(42).permutation(rows) == perm).all(), "同じ seed なら同じ順序(epoch・rank で一致)"
print("ok: val rows permuted with fixed seed; 2048-head spans", len(np.unique(head_perm)), "slides (was 3)")
