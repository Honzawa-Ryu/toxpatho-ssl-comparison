# -*- coding: utf-8 -*-
"""
# バッチ（スライド）/ 色が表現をどれだけ駆動しているかの定量

`scripts/analysis/batch_color_influence.py` から指標計算部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。集計フロー（run）も Phase 4 でここへ移し、
scripts 側は CLI シムだけになった（§5-0）。

指標:
  - eta2                  : ある grouping で説明される埋め込み分散の割合
  - within_group_cohesion : スライド内コサイン類似度の平均 - 全体平均
"""
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score as ami

from lib.analysis.common import SEED, l2norm


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


def run(input_dir, k=10):
    """各手法について、表現がスライド/色にどれだけ引っ張られているかを集計する。

    旧 `scripts/analysis/batch_color_influence.py` の main() 本体をそのまま
    関数化したもの（REFACTOR_PLAN.md §5-0 / Phase 4）。
    """
    out = input_dir
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    meta = json.load(open(os.path.join(out, "patches_meta.json")))
    thumbs = np.load(os.path.join(out, "patches_thumbs.npy"))
    slide = np.array([m["wsi"] for m in meta])

    # color grouping from mean patch RGB
    mean_rgb = thumbs.reshape(len(thumbs), -1, 3).mean(1)  # (N,3)
    color_lab = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(mean_rgb)

    embs = {}
    for f in sorted(os.listdir(out)):
        if f.startswith("emb_") and f.endswith(".npy"):
            arr = np.load(os.path.join(out, f))
            if np.isfinite(arr).all():
                embs[f[4:-4]] = arr

    print(f"[info] {len(embs)} methods, N={len(slide)}, slides={len(set(slide))}")
    print(f"[coupling] AMI(color,slide) = {ami(color_lab, slide):.3f} "
          "(staining and slide are coupled if high)\n")

    rows = []
    for name, emb in embs.items():
        Xn = l2norm(emb)
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Xn)
        rows.append({
            "method": name,
            "eta2_slide": round(eta2(Xn, slide), 3),
            "eta2_color": round(eta2(Xn, color_lab), 3),
            "AMI_clust_slide": round(ami(km, slide), 3),
            "AMI_clust_color": round(ami(km, color_lab), 3),
            "within_slide_cohesion": round(within_group_cohesion(Xn, slide), 3),
        })
    df = pd.DataFrame(rows).sort_values("eta2_slide", ascending=False)
    csv_path = os.path.join(out, "batch_color_influence.csv")
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))

    # figure: t-SNE recolored by mean RGB for healthy methods + collapsed one
    show = [n for n in ["rn50_barlowtwins", "densenet121_barlowtwins",
                        "rn50_simsiam_collapsed"] if n in embs]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(4.3 * len(show), 4.2))
        if len(show) == 1:
            axes = [axes]
        rgb = (mean_rgb / 255.0).clip(0, 1)
        for ax, n in zip(axes, show):
            Xn = l2norm(embs[n])[:1500]
            z = TSNE(n_components=2, init="pca", random_state=SEED,
                     perplexity=30).fit_transform(PCA(50, random_state=SEED).fit_transform(Xn)
                                                   if Xn.shape[1] > 50 else Xn)
            ax.scatter(z[:, 0], z[:, 1], c=rgb[:len(z)], s=7)
            ax.set_title(f"{n}\n(点色 = パッチ平均RGB)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("t-SNE を『パッチの平均色』で着色 — 色でまとまるほど色/バッチ駆動", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "tsne_by_color.png"), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\n[fig] {os.path.join(fig_dir, 'tsne_by_color.png')}")

    return csv_path
