# -*- coding: utf-8 -*-
"""
# 解析スクリプト共通のユーティリティ

`scripts/analysis/` 配下の3ファイル（batch_color_influence / compare_representations /
visualize）が同じ `l2norm` をそれぞれ定義していたため、ここに集約した
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。
"""
import numpy as np

# 解析全体で共有する乱数シード
SEED = 0


def l2norm(x):
    """行ごとに L2 正規化する。ゼロ行はそのまま（ゼロ除算を避ける）。"""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n
