# -*- coding: utf-8 -*-
"""
# SSL表現の比較（埋め込みの読み込み・可視化・比較レポート生成）

`scripts/analysis/compare_representations.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。比較ロジック本体（run）も Phase 4 でここへ
移し、scripts 側は CLI シムだけになった（§5-0）。
"""
import os
import glob
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from lib.analysis import cka as ckalib
from lib.analysis.common import SEED, l2norm


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


def run(input_dir, k=10, rbf=False, label_csv="", label_key="wsi"):
    """extract の出力を読み、CKA / ARI / プロトタイプ / t-SNE とレポートを作る。

    旧 `scripts/analysis/compare_representations.py` の main() 本体をそのまま
    関数化したもの（REFACTOR_PLAN.md §5-0 / Phase 4）。
    """
    out = input_dir
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    embs, dropped = load_embeddings(out)
    names = list(embs)
    embs_norm = {kk: l2norm(v) for kk, v in embs.items()}
    thumbs = np.load(os.path.join(out, "patches_thumbs.npy"))
    meta = json.load(open(os.path.join(out, "patches_meta.json")))

    report = ["# 表現比較レポート (Representation Comparison)\n",
              f"- methods: {', '.join(names)}",
              f"- #patches: {embs[names[0]].shape[0]}",
              f"- embedding dims: {{ {', '.join(f'{kk}:{v.shape[1]}' for kk, v in embs.items())} }}",
              f"- k-means clusters: {k}"]
    if dropped:
        report += [f"- ⚠️ 除外(NaN/Inf=発散/破損チェックポイント): {', '.join(dropped)}"]
    report += [""]

    # (a) CKA -----------------------------------------------------------------
    _, cka_lin = ckalib.cka_matrix(embs, kernel="linear")
    pd.DataFrame(cka_lin, index=names, columns=names).to_csv(os.path.join(out, "cka_linear.csv"))
    save_heatmap(cka_lin, names, "Linear CKA (latent-space similarity)",
                 os.path.join(fig_dir, "cka_linear.png"), vmin=0, vmax=1)
    report += ["## (a) CKA — 潜在空間の類似度",
               "![linear CKA](figures/cka_linear.png)",
               "1に近いほど2手法の表現が線形変換で一致。", ""]
    if rbf:
        _, cka_rbf = ckalib.cka_matrix(embs, kernel="rbf")
        pd.DataFrame(cka_rbf, index=names, columns=names).to_csv(os.path.join(out, "cka_rbf.csv"))
        save_heatmap(cka_rbf, names, "RBF-kernel CKA",
                     os.path.join(fig_dir, "cka_rbf.png"), vmin=0, vmax=1)
        report += ["![rbf CKA](figures/cka_rbf.png)", ""]

    # (b) clustering + ARI ----------------------------------------------------
    labels = {}
    for m in names:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels[m] = km.fit_predict(embs_norm[m])
    pd.DataFrame(labels).to_csv(os.path.join(out, "cluster_labels.csv"), index=False)

    ari = np.eye(len(names))
    nmi = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            ari[i, j] = ari[j, i] = adjusted_rand_score(labels[a], labels[b])
            nmi[i, j] = nmi[j, i] = normalized_mutual_info_score(labels[a], labels[b])
    pd.DataFrame(ari, index=names, columns=names).to_csv(os.path.join(out, "ari_methods.csv"))
    pd.DataFrame(nmi, index=names, columns=names).to_csv(os.path.join(out, "nmi_methods.csv"))
    save_heatmap(ari, names, f"ARI between methods' k-means (k={k})",
                 os.path.join(fig_dir, "ari_methods.png"), vmin=0, vmax=1, cmap="magma")
    report += ["## (b) クラスタ構造の一致度 (ARI, 手法間)",
               "![ARI](figures/ari_methods.png)",
               "各手法の潜在空間を独立にk-meansし、割り当ての一致をARIで比較（ラベル不変）。", ""]

    # (c) prototype patches ---------------------------------------------------
    report += ["## (c) プロトタイプパッチ（各クラスタ中心に近いパッチ）"]
    for m in names:
        p = os.path.join(fig_dir, f"prototypes_{m}.png")
        prototype_montage(thumbs, embs_norm[m], labels[m], k, m, p)
        report += [f"### {m}", f"![prototypes {m}](figures/prototypes_{m}.png)", ""]

    # (d) t-SNE ---------------------------------------------------------------
    tsne_plot(embs_norm, labels, os.path.join(fig_dir, "tsne.png"))
    report += ["## (d) t-SNE（各手法・自クラスタで着色）",
               "![tsne](figures/tsne.png)", ""]

    # (e) optional ground-truth labels ---------------------------------------
    if label_csv and os.path.exists(label_csv):
        ldf = pd.read_csv(label_csv)
        keycol = "key" if "key" in ldf.columns else label_key
        lut = dict(zip(ldf[keycol].astype(str), ldf["label"].astype(str)))
        gt = np.array([lut.get(str(mm.get(keycol, mm.get(label_key, mm.get("key", "")))), None)
                       for mm in meta], dtype=object)
        valid = gt != None  # noqa: E711
        cov = valid.mean()
        report += ["## (e) ラベル付き評価（A/B/C 近接・意味ARI）",
                   f"- ラベル被覆率: {cov:.1%}（{keycol} で結合）"]
        if valid.sum() > 10:
            uniq = sorted(set(gt[valid]))
            gt_int = np.array([uniq.index(g) if g in uniq else -1 for g in gt])
            rows = []
            for m in names:
                a = adjusted_rand_score(gt_int[valid], labels[m][valid])
                nn = normalized_mutual_info_score(gt_int[valid], labels[m][valid])
                rows.append({"method": m, "ARI_vs_label": round(a, 3), "NMI_vs_label": round(nn, 3)})
            pd.DataFrame(rows).to_csv(os.path.join(out, "ari_vs_label.csv"), index=False)
            report += ["", "| method | ARI_vs_label | NMI_vs_label |",
                       "|---|---|---|"]
            report += [f"| {r['method']} | {r['ARI_vs_label']} | {r['NMI_vs_label']} |" for r in rows]
            report += [""]
            # per-label centroid cosine-distance ("A/B/C proximity") per method
            for m in names:
                cents = np.stack([l2norm(embs_norm[m][valid][gt[valid] == u].mean(0, keepdims=True))[0]
                                  for u in uniq])
                dist = 1 - cents @ cents.T
                save_heatmap(dist, uniq, f"{m}: 1-cos between label centroids",
                             os.path.join(fig_dir, f"labeldist_{m}.png"), cmap="rocket_r")
                report += [f"### {m}: ラベル重心間コサイン距離",
                           f"![labeldist {m}](figures/labeldist_{m}.png)", ""]
    else:
        report += ["## (e) ラベル付き評価",
                   "ラベルCSV未指定のためスキップ。`--label_csv key,label` を渡すと意味ARIとA/B/C近接を出力。", ""]

    report_path = os.path.join(out, "representation_comparison_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))
    print(f"[done] report -> {report_path}")
    return report_path
