# -*- coding: utf-8 -*-
"""
# SSL表現の比較（埋め込みの読み込み・可視化）

`scripts/analysis/compare_representations.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。CLI と比較ロジック本体は呼び出し側に残してある。
"""
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lib.analysis.common import SEED


def load_embeddings(input_dir):
    embs = {}
    dropped = []
    for f in sorted(glob.glob(os.path.join(input_dir, "emb_*.npy"))):
        name = os.path.basename(f)[len("emb_"):-len(".npy")]
        arr = np.load(f)
        if not np.isfinite(arr).all():
            bad = (~np.isfinite(arr)).mean()
            print(f"[load] DROPPING {name}: {bad:.0%} non-finite values "
                  f"(diverged/corrupt checkpoint)")
            dropped.append(name)
            continue
        embs[name] = arr
    if not embs:
        raise FileNotFoundError(f"No usable emb_*.npy in {input_dir}")
    n = {k: v.shape[0] for k, v in embs.items()}
    assert len(set(n.values())) == 1, f"embeddings differ in #samples: {n}"
    print(f"[load] methods={list(embs)} n={list(n.values())[0]} "
          f"dims={ {k: v.shape[1] for k, v in embs.items()} }")
    return embs, dropped


def save_heatmap(mat, names, title, path, fmt=".2f", vmin=None, vmax=None, cmap="viridis"):
    plt.figure(figsize=(1.4 * len(names) + 2, 1.2 * len(names) + 1.5))
    sns.heatmap(mat, xticklabels=names, yticklabels=names, annot=True, fmt=fmt,
                vmin=vmin, vmax=vmax, cmap=cmap, square=True, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def prototype_montage(thumbs, emb_norm, labels, k, name, path, top=8):
    """Rows = clusters, cols = patches nearest each cluster centroid."""
    clusters = sorted(np.unique(labels))
    fig, axes = plt.subplots(len(clusters), top,
                             figsize=(top * 1.1, len(clusters) * 1.1))
    if len(clusters) == 1:
        axes = axes[None, :]
    for r, c in enumerate(clusters):
        idx = np.where(labels == c)[0]
        centroid = emb_norm[idx].mean(0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        order = idx[np.argsort(-(emb_norm[idx] @ centroid))][:top]
        for col in range(top):
            ax = axes[r, col]
            ax.axis("off")
            if col < len(order):
                ax.imshow(thumbs[order[col]])
            if col == 0:
                ax.set_ylabel(f"c{c}\n(n={len(idx)})", rotation=0, labelpad=22,
                              fontsize=8, va="center")
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{name}: prototype patches per cluster", y=1.001)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def tsne_plot(embs_norm, labels_by_method, path, n=1500):
    methods = list(embs_norm)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        sub = embs_norm[m][:n]
        # PCA-reduce before t-SNE (much faster and standard practice on high-dim SSL feats)
        if sub.shape[1] > 50:
            sub = PCA(n_components=50, random_state=SEED).fit_transform(sub)
        z = TSNE(n_components=2, init="pca", random_state=SEED,
                 perplexity=30).fit_transform(sub)
        ax.scatter(z[:, 0], z[:, 1], c=labels_by_method[m][:n], s=5,
                   cmap="tab10", alpha=0.7)
        ax.set_title(m); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
