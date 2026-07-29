# -*- coding: utf-8 -*-
"""
# バッチ（スライド）/ 色が表現をどれだけ駆動しているかの定量

`scripts/analysis/batch_color_influence.py` から指標計算部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。CLI は呼び出し側に残してある。

指標:
  - eta2                  : ある grouping で説明される埋め込み分散の割合
  - within_group_cohesion : スライド内コサイン類似度の平均 - 全体平均
"""
import numpy as np

from lib.analysis.common import SEED


def eta2(X, labels):
    """Multivariate variance fraction explained by a grouping (trace ratio)."""
    X = X - X.mean(0, keepdims=True)
    ss_tot = float((X ** 2).sum())
    ss_between = 0.0
    for g in np.unique(labels):
        Xg = X[labels == g]
        if len(Xg) == 0:
            continue
        diff = Xg.mean(0)
        ss_between += len(Xg) * float((diff ** 2).sum())
    return ss_between / (ss_tot + 1e-12)


def within_group_cohesion(Xn, labels, min_n=5):
    """Mean within-slide cosine similarity minus global mean cosine similarity."""
    glob = float((Xn @ Xn.mean(0)).mean())  # ~ mean cos to centroid direction proxy
    sims = []
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        if len(idx) < min_n:
            continue
        G = Xn[idx]
        # mean pairwise cosine (excluding self)
        S = G @ G.T
        n = len(idx)
        sims.append((S.sum() - n) / (n * (n - 1)))
    within = float(np.mean(sims)) if sims else float("nan")
    # global mean pairwise cosine on a sample
    rng = np.random.default_rng(SEED)
    s = Xn[rng.choice(len(Xn), min(len(Xn), 800), replace=False)]
    Sg = s @ s.T
    m = len(s)
    global_pair = (Sg.sum() - m) / (m * (m - 1))
    return within - float(global_pair)
