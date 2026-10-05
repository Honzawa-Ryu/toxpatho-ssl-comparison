#!/usr/bin/env python3
"""bt_steps_ep*.jsonl（--bt_step_log が出す step テレメトリ）から、発散の起点を読み出す。

torch 不要・CPU のみ（ログインノードや andre01 で実行してよい）。

    python3 scripts/analysis/bt_step_report.py outputs/0030_20261005_bt_diag_resume_ep64 --epoch 66
    python3 scripts/analysis/bt_step_report.py <出力ディレクトリ> --epoch 66 --baseline-epoch 65

出力:
  1. 基準 epoch（健全な epoch）の各指標の min / median / max
  2. 対象 epoch で、基準の最大値を初めて明確に超えた step（grad_norm・loss それぞれ）と、その前後の表
     （どの指標が先に動いたかを見る: z の std/比が先なら標準化の数値問題、grad_norm が先なら最適化の発散）
  3. 最初の非有限 step（first_nonfinite）

注意: z の統計(std/mean)は rank0 のローカルバッチ(GPUあたり256)の値。c の対角・on/off 項は全 rank 集約後。
"""
import argparse
import glob
import json
import statistics as st
from pathlib import Path

COLS = ["loss", "grad_norm", "z_std_min", "n_zero_std", "z_mean_abs_max", "z_ratio_max",
        "c_diag_mean", "on_diag", "off_diag"]


def load(directory, epoch):
    files = sorted(glob.glob(str(Path(directory) / f"bt_steps_ep{epoch:04d}_*.jsonl")))
    if not files:
        raise SystemExit(f"bt_steps_ep{epoch:04d}_*.jsonl が {directory} に無い")
    return [json.loads(line) for line in open(files[-1])]   # 再開した試行が複数あれば最新


def finite(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def fmt(v):
    return f"{v:10.4g}" if isinstance(v, (int, float)) else f"{str(v):>10}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--epoch", type=int, required=True, help="調べる epoch（1始まり。NaN を出した epoch）")
    ap.add_argument("--baseline-epoch", type=int, default=None, help="健全な基準 epoch（既定: epoch-1）")
    ap.add_argument("--window", type=int, default=30, help="起点の後ろに表示する step 数")
    ap.add_argument("--gn-factor", type=float, default=2.0, help="基準の最大 grad_norm のこの倍を超えたら異常とみなす")
    a = ap.parse_args()

    base_ep = a.baseline_epoch or a.epoch - 1
    base, rows = load(a.directory, base_ep), load(a.directory, a.epoch)
    print(f"基準 ep{base_ep}: {len(base)} step / 対象 ep{a.epoch}: {len(rows)} step\n")
    print(f"== 基準 ep{base_ep} の step 統計 ==")
    for k in COLS:
        v = finite(base, k)
        if v:
            print(f"{k:16s} min {min(v):10.4g}  med {st.median(v):10.4g}  max {max(v):10.4g}")

    gn_lim = a.gn_factor * max(finite(base, "grad_norm"))
    loss_lim = 1.3 * max(finite(base, "loss"))
    onset_gn = next((i for i, r in enumerate(rows) if isinstance(r["grad_norm"], (int, float)) and r["grad_norm"] > gn_lim), None)
    onset_loss = next((i for i, r in enumerate(rows) if isinstance(r["loss"], (int, float)) and r["loss"] > loss_lim), None)
    nonfinite = next((i for i, r in enumerate(rows) if r.get("first_nonfinite")), None)
    print(f"\n基準を超えた最初の step: grad_norm>{gn_lim:.3g} -> {onset_gn} / loss>{loss_lim:.4g} -> {onset_loss}"
          f" / 最初の非有限 -> {nonfinite}")
    cands = [i for i in (onset_gn, onset_loss) if i is not None]
    if not cands:
        print("基準を超えた step は無い。")
        return
    lo = max(0, min(cands) - 8)
    print(f"\n== ep{a.epoch} step {lo}〜{lo + a.window} ==")
    print("step  " + " ".join(f"{c[:10]:>10}" for c in COLS))
    for r in rows[lo: lo + a.window + 1]:
        print(f"{r['step']:4d}  " + " ".join(fmt(r.get(c)) for c in COLS))
    pre = rows[:min(cands)]
    print(f"\n起点より前（step 0〜{min(cands) - 1}）: grad_norm 中央値 {st.median(finite(pre, 'grad_norm')):.3g} "
          f"最大 {max(finite(pre, 'grad_norm')):.3g}; loss 中央値 {st.median(finite(pre, 'loss')):.4g}")


if __name__ == "__main__":
    main()
