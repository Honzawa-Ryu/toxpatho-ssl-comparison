# -*- coding: utf-8 -*-
"""
# スライド単位の t-SNE 可視化

`scripts/analysis/visualize.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。
学習ログのパース (parse_loss / parse_metrics) と各種プロットは RESULT_BASE や
LOG_EXPERIMENTS といったスクリプト固有の設定に強く依存するため、呼び出し側に残した。
"""
import os
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lib.analysis.common import SEED, l2norm

# 表示名の既定。呼び出し側から pretty= で上書きできる。
PRETTY = {"mae_vitb16": "MAE", "dino_vitb16": "DINO",
          "bt_vitb16": "Barlow Twins", "simsiam_vitb16": "SimSiam"}


def tsne_by_slide(input_dir, out_path, n=2000, top_slides=12, pretty=None):
    pretty = pretty or PRETTY
    meta = json.load(open(os.path.join(input_dir, "patches_meta.json")))
    slide = np.array([m["wsi"] for m in meta])
    embs = {}
    for f in sorted(glob.glob(os.path.join(input_dir, "emb_*.npy"))):
        arr = np.load(f)
        if np.isfinite(arr).all():
            embs[os.path.basename(f)[4:-4]] = arr
    names = list(embs)

    # highlight the most frequent slides; gray the rest
    uniq, counts = np.unique(slide, return_counts=True)
    top = uniq[np.argsort(-counts)][:top_slides]
    cmap = plt.get_cmap("tab20")
    color_of = {s: cmap(i % 20) for i, s in enumerate(top)}
    point_colors = np.array([color_of.get(s, (0.8, 0.8, 0.8, 0.35)) for s in slide], dtype=object)

    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 4.3))
    if len(names) == 1:
        axes = [axes]
    for ax, m in zip(axes, names):
        X = l2norm(embs[m])[:n]
        if X.shape[1] > 50:
            X = PCA(n_components=50, random_state=SEED).fit_transform(X)
        z = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=30).fit_transform(X)
        cols = list(point_colors[:len(z)])
        # draw gray first, highlighted on top
        is_top = np.array([slide[i] in color_of for i in range(len(z))])
        ax.scatter(z[~is_top, 0], z[~is_top, 1], c="#cccccc", s=6, alpha=0.35, linewidths=0)
        ax.scatter(z[is_top, 0], z[is_top, 1],
                   c=[cols[i] for i in range(len(z)) if is_top[i]], s=10, linewidths=0)
        ax.set_title(pretty.get(m, m), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=color_of[s],
                      markeredgecolor="none", markersize=6, label=str(s)) for s in top]
    fig.legend(handles=handles, title=f"slide (top {top_slides})", loc="center right",
               bbox_to_anchor=(1.14, 0.5), fontsize=7, title_fontsize=8, frameon=False)
    fig.suptitle("t-SNE of latent space colored by slide (WSI) — tighter same-color clusters = more batch-dominated", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] {out_path}")
