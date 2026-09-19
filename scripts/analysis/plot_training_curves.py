"""学習ログ(log_HHMMSS.txt)から train_loss / grad_norm / eff_rank を抜いて実験間で比較する。

学習ログのパースだけなので GPU も計算ノードも要らない。ログインノードで動く。

    uv run --with matplotlib python scripts/analysis/plot_training_curves.py \
        --exp 0025_20260831_dino_paper_faithful \
        --exp 0026_20260901_dino_clipgrad03 \
        --exp 0028_20260917_dino_lr_half \
        --out outputs/_analysis/curves_0025_0026_0028.png

--json を付けると素の系列も吐く(プロット環境が無いとき用)。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"

# [20260917-203014] [INFO] Epoch: 5, train_loss: 7.6981, lr: 4.06e-04, grad_norm: 2.9491, ...
EPOCH_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*train_loss:\s*(?P<train_loss>[\d.]+)"
    r".*?lr:\s*(?P<lr>[\d.e+-]+)"
    r".*?grad_norm:\s*(?P<grad_norm>[\d.]+)"
)
# [20260917-203033] [INFO] eff_rank: 165.77, feat_std: 0.7510, alignment: ..., uniformity: ...
RANK_RE = re.compile(
    r"eff_rank:\s*(?P<eff_rank>[\d.]+),\s*feat_std:\s*(?P<feat_std>[\d.]+)"
)
VAL_RE = re.compile(r"val_loss:\s*(?P<val_loss>[\d.]+)")

# DINO ヘッドの出力次元が 65536 のとき、完全崩壊(全サンプル同一分布)の loss は ln(65536)
COLLAPSE_LOSS = 11.0904


def main_logs(exp_dir: Path) -> list[Path]:
    """rank1-3 を除いた rank0 のログを時刻順に返す。resume で複数に分かれるため。"""
    logs = [p for p in exp_dir.glob("log_*.txt") if not re.search(r"_rank\d+\.txt$", p.name)]
    return sorted(logs, key=lambda p: p.name)


def parse_exp(exp_dir: Path) -> dict:
    """1実験ぶんのログを epoch をキーにマージする。

    resume すると同じ epoch が複数ログに出るので、後から読んだ(=新しい)ログで上書きする。
    eff_rank は rank_monitor_interval ごとにしか出ず、直前の Epoch 行に属する。
    """
    by_epoch: dict[int, dict] = {}
    for log in main_logs(exp_dir):
        current: int | None = None
        for line in log.read_text(errors="replace").splitlines():
            m = EPOCH_RE.search(line)
            if m:
                current = int(m.group("epoch"))
                by_epoch.setdefault(current, {})["epoch"] = current
                by_epoch[current]["train_loss"] = float(m.group("train_loss"))
                by_epoch[current]["grad_norm"] = float(m.group("grad_norm"))
                by_epoch[current]["lr"] = float(m.group("lr"))
                continue
            if current is None:
                continue
            m = RANK_RE.search(line)
            if m:
                by_epoch[current]["eff_rank"] = float(m.group("eff_rank"))
                by_epoch[current]["feat_std"] = float(m.group("feat_std"))
                continue
            m = VAL_RE.search(line)
            if m:
                # val は次の epoch の頭で出るので、直前の epoch の値として持つ
                by_epoch[current]["val_loss"] = float(m.group("val_loss"))

    config_path = exp_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "name": exp_dir.name,
        "config": {k: config.get(k) for k in ("lr", "clip_grad", "batch_size", "epochs")},
        "records": [by_epoch[e] for e in sorted(by_epoch)],
    }


def collapse_epoch(records: list[dict], tol: float = 1e-3) -> int | None:
    """train_loss が ln(65536) に張り付いた最初の epoch。崩壊していなければ None。"""
    for r in records:
        if abs(r.get("train_loss", 0.0) - COLLAPSE_LOSS) < tol:
            return r["epoch"]
    return None


def label_for(exp: dict) -> str:
    cfg = exp["config"]
    tag = exp["name"].split("_")[0]
    return f"{tag} (lr={cfg.get('lr')}, clip_grad={cfg.get('clip_grad')})"


def plot(exps: list[dict], out: Path, zoom_epochs: int = 100) -> None:
    """上段=全範囲、下段=序盤 zoom_epochs まで。

    崩壊した実験は数十 epoch で終わるので、全範囲だけだと潰れて比較にならない。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    keys = [("train_loss", "train loss"), ("grad_norm", "grad_norm (pre-clip)"), ("eff_rank", "effective rank")]

    for row in (0, 1):
        for col, (key, title) in enumerate(keys):
            ax = axes[row][col]
            for i, exp in enumerate(exps):
                color = colors[i % len(colors)]
                pts = [(r["epoch"], r[key]) for r in exp["records"] if key in r]
                if row == 1:
                    pts = [(e, v) for e, v in pts if e <= zoom_epochs]
                if pts:
                    ax.plot(*zip(*pts), label=label_for(exp), color=color, lw=1.4)
                dead = collapse_epoch(exp["records"])
                if dead is not None and (row == 0 or dead <= zoom_epochs):
                    ax.axvline(dead, color=color, ls=":", lw=1.0, alpha=0.7)
            ax.set_title(title if row == 0 else f"{title}  [ep<={zoom_epochs}]")
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            if key == "train_loss":
                ax.axhline(COLLAPSE_LOSS, color="0.4", ls="--", lw=0.9)
                ax.annotate("ln(65536)=11.09 (collapse)", (0.02, COLLAPSE_LOSS),
                            xycoords=("axes fraction", "data"), va="bottom", fontsize=8, color="0.3")
            if key == "eff_rank":
                ax.set_yscale("log")
    axes[0][0].legend(fontsize=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", action="append", required=True,
                   help="outputs/ 以下の実験ディレクトリ名。複数回指定で比較")
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "_analysis" / "training_curves.png")
    p.add_argument("--json", type=Path, help="抽出した系列を JSON で書き出す")
    p.add_argument("--no-plot", action="store_true", help="matplotlib を使わず JSON だけ出す")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exps = []
    for name in args.exp:
        exp_dir = Path(name) if Path(name).is_dir() else OUTPUTS_DIR / name
        if not exp_dir.is_dir():
            raise SystemExit(f"実験ディレクトリが無い: {exp_dir}")
        exp = parse_exp(exp_dir)
        dead = collapse_epoch(exp["records"])
        last = exp["records"][-1] if exp["records"] else {}
        print(f"{exp['name']}: {len(exp['records'])} epochs, last ep{last.get('epoch')} "
              f"loss={last.get('train_loss')} eff_rank={last.get('eff_rank')} "
              f"collapse={'ep%d' % dead if dead else 'none'}")
        exps.append(exp)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(exps, indent=2))
        print(f"wrote {args.json}")
    if not args.no_plot:
        plot(exps, args.out)


if __name__ == "__main__":
    main()
