# -*- coding: utf-8 -*-
"""
# 学習曲線・collapse監視・スライド単位の t-SNE 可視化

`scripts/analysis/visualize.py` から再利用可能な部分を切り出したもの
（REFACTOR_PLAN.md §5-3 / Phase 1-b）。Phase 1-b では「RESULT_BASE や
LOG_EXPERIMENTS といったスクリプト固有の設定に依存する」という理由で
ログのパースとプロットを呼び出し側に残していたが、Phase 4 で
それらもここへ移した（§5-0「研究の実処理は lib/」）。設定値は引数で渡す。
"""
import os
import re
import json
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import yaml

from lib.analysis.common import SEED, l2norm

# 表示名の既定。呼び出し側から pretty= で上書きできる。
PRETTY = {"mae_vitb16": "MAE", "dino_vitb16": "DINO",
          "bt_vitb16": "Barlow Twins", "simsiam_vitb16": "SimSiam"}

# preview で表示する手法名 -> 学習の実験ディレクトリ名
LOG_EXPERIMENTS = {
    "mae_vitb16": "20260710_000001_tggate_vitb16_mae",
    "dino_vitb16": "20260711_000001_tggate_vitb16_dino_v2",
    "bt_vitb16": "20260711_000002_tggate_vitb16_barlowtwins",
    "simsiam_vitb16": "20260711_000003_tggate_vitb16_simsiam",
}
COLORS = {"mae_vitb16": "#c0648f", "dino_vitb16": "#6a4c93",
          "bt_vitb16": "#3f8f6b", "simsiam_vitb16": "#a9762f"}
PALETTE = ["#6a4c93", "#c0648f", "#3f8f6b", "#a9762f", "#3d7fc1", "#c1543d"]


def default_result_base():
    """学習ログ (log_*.txt) の探索起点。

    以前は "result/workspace/andre01/honzawa/wsi-ad/outputs" が直接書かれていた。
    これは --dir_result の二重ネストバグ（REFACTOR_PLAN.md §1-3）が作っていた
    {PROJECT_ROOT}/result/{PROJECT_ROOT}/outputs/ という異常なパスそのもので、
    その場所を読むよう合わせ込まれていた。

    §1-3 の修正後、新しい学習の出力は {PROJECT_ROOT}/outputs/ に落ちるため、
    既定をそちらに変更した（§5-1 の「outputs/ に統一」決定に一致）。
    旧パスにある過去の実験を見たい場合は環境変数 RESULT_BASE で上書きする。
    """
    return os.environ.get(
        "RESULT_BASE",
        os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "outputs"),
    )


def experiments_from_methods_config(path):
    """Derive {name: exp_dir} / pretty-name / color from a methods.yaml used by
    extract_embeddings, so loss/eff_rank plots track whatever checkpoints the
    embedding step actually used (model_path=.../outputs/<EXP_DIR>/model_epN.pt)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    log_exp, pretty, colors = {}, {}, {}
    for i, m in enumerate(cfg["methods"]):
        mp = m.get("model_path", "")
        mobj = re.search(r"outputs/([^/]+)/", mp)
        if not mobj:
            continue
        log_exp[m["name"]] = mobj.group(1)
        pretty[m["name"]] = m["name"]
        colors[m["name"]] = PALETTE[i % len(PALETTE)]
    return log_exp, pretty, colors


def parse_loss(exp_dir, result_base=None):
    result_base = result_base or default_result_base()
    logs = glob.glob(os.path.join(result_base, exp_dir, "log_*.txt"))
    if not logs:
        return [], []
    log = max(logs, key=os.path.getmtime)
    ep, loss = [], []
    with open(log) as f:
        for line in f:
            m = re.search(r"Epoch:\s*(\d+),\s*train_loss:\s*([-0-9.eE+]+)", line)
            if m:
                ep.append(int(m.group(1)))
                loss.append(float(m.group(2)))
    return ep, loss


def parse_metrics(exp_dir, result_base=None):
    """Return {epoch:[...], eff_rank:[...], feat_std:[...]} parsed from the log.
    eff_rank/feat_std are logged every rank_monitor_interval epochs, on the line
    following the 'Epoch: N' line, so we track the most recent epoch seen."""
    result_base = result_base or default_result_base()
    logs = glob.glob(os.path.join(result_base, exp_dir, "log_*.txt"))
    if not logs:
        return {"epoch": [], "eff_rank": [], "feat_std": []}
    log = max(logs, key=os.path.getmtime)
    cur_ep, ep, er, fs = 0, [], [], []
    with open(log) as f:
        for line in f:
            me = re.search(r"Epoch:\s*(\d+),\s*train_loss", line)
            if me:
                cur_ep = int(me.group(1))
                continue
            mm = re.search(r"eff_rank:\s*([0-9.]+),\s*feat_std:\s*([0-9.]+)", line)
            if mm:
                ep.append(cur_ep); er.append(float(mm.group(1))); fs.append(float(mm.group(2)))
    return {"epoch": ep, "eff_rank": er, "feat_std": fs}


def plot_loss_curves(names, out_path, log_exp=None, pretty=None, result_base=None):
    log_exp = log_exp or LOG_EXPERIMENTS
    pretty = pretty or PRETTY
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 3.4))
    if len(names) == 1:
        axes = [axes]
    for ax, n in zip(axes, names):
        ep, loss = parse_loss(log_exp.get(n, ""), result_base)
        if ep:
            ax.plot(ep, loss, color="#6a4c93", lw=1.8)
            ax.set_title(f"{pretty.get(n, n)}  (ep{ep[-1]})", fontsize=10)
        else:
            ax.set_title(f"{pretty.get(n, n)}  (no log)", fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("train loss")
    fig.suptitle("Training loss per method (loss scale is method-specific)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] {out_path}")


def _feat_dim(input_dir, name, default=768):
    """Backbone feature dimensionality (to normalize effective rank across backbones)."""
    p = os.path.join(input_dir, f"emb_{name}.npy")
    if os.path.exists(p):
        return int(np.load(p, mmap_mode="r").shape[1])
    return default


def plot_effrank(names, out_path, input_dir, log_exp=None, pretty=None, colors=None,
                 result_base=None):
    log_exp = log_exp or LOG_EXPERIMENTS
    pretty = pretty or PRETTY
    colors = colors or COLORS
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for m in names:
        d = parse_metrics(log_exp.get(m, ""), result_base)
        if not d["epoch"]:
            continue
        c = colors.get(m, "#555")
        dim = _feat_dim(input_dir, m)
        norm_er = [e / dim for e in d["eff_rank"]]  # backbone-agnostic: eff_rank / feature_dim in [0,1]
        lbl = f"{pretty.get(m, m)} (d={dim})"
        axes[0].plot(d["epoch"], norm_er, "-o", ms=3, lw=1.6, color=c, label=lbl)
        axes[1].plot(d["epoch"], d["feat_std"], "-o", ms=3, lw=1.6, color=c, label=pretty.get(m, m))
    axes[0].set_title("normalized effective rank = eff_rank / feature_dim  (backbone-agnostic, [0,1])")
    axes[0].set_ylabel("normalized effective rank")
    axes[0].set_ylim(0, 1)
    axes[1].axhline(0.0, color="#b0454a", lw=0.8, ls="--", alpha=0.6)
    axes[1].set_title("feature dim std  (scale-dependent; -> 0 signals dimensional collapse)")
    axes[1].set_ylabel("feature std")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("Collapse monitoring (eff_rank / feat_std; MAE now included via single-view eval)", y=1.03)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[fig] {out_path}")


def run(input_dir, n=2000, methods_config="", skip_tsne=False, result_base=None):
    """学習曲線 / collapse監視 / スライド着色 t-SNE をまとめて描く。

    旧 `scripts/analysis/visualize.py` の main() 本体をそのまま関数化したもの
    （REFACTOR_PLAN.md §5-0 / Phase 4）。
    """
    fig_dir = os.path.join(input_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    if methods_config:
        log_exp, pretty, colors = experiments_from_methods_config(methods_config)
    else:
        log_exp, pretty, colors = LOG_EXPERIMENTS, PRETTY, COLORS

    plot_loss_curves(list(log_exp), os.path.join(fig_dir, "loss_curves.png"),
                     log_exp, pretty, result_base)
    plot_effrank(list(log_exp), os.path.join(fig_dir, "effrank_featstd.png"), input_dir,
                 log_exp, pretty, colors, result_base)
    if not skip_tsne:
        tsne_by_slide(input_dir, os.path.join(fig_dir, "tsne_by_slide.png"),
                      n=n, pretty=pretty)
    return fig_dir


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
