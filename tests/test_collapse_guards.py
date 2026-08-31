# -*- coding: utf-8 -*-
"""2026-08-31のDINO恒久崩壊の再発防止テスト（純Python、torch/GPU不要）。

守っているのは次の2点:

  1. `lib/trainer/model.py:_wd_groups` が bias / ndim<=1 のパラメータを
     weight decay から必ず外すこと。外れていないと backbone の LayerNorm
     ゲインが毎step `γ <- γ(1 - lr*wd)` で削られ、ep100で初期値の0.2%まで
     消えて恒久崩壊する（0017/0021/0023/0024 が全滅した実際の原因）。
  2. `lib/sslmodel/utils.py:CollapseMonitor` が「loss が ln(out_dim) に
     張り付いた一様崩壊」を検出できること。旧実装は effective_rank だけを
     見ていたため、0024 が36 epoch崩壊し続けても最後まで発火しなかった。

対象はどちらも外部依存の無い純粋ロジックだが、定義元のモジュールは
torch/timm をimportするためコンテナ外ではimportできない。そこでASTから
当該定義だけを取り出してexecする（テストのためにロジックを複製すると
本体と乖離するので、必ず本体ソースから読むこと）。

実行: python3 tests/test_collapse_guards.py
依存: python3 / numpy のみ
"""
import ast
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILURES = 0
PASSES = 0


def _ok(msg):
    global PASSES
    PASSES += 1
    print(f"  ✅ {msg}")


def _fail(msg):
    global FAILURES
    FAILURES += 1
    print(f"  ❌ {msg}")


def _check(desc, cond):
    _ok(desc) if cond else _fail(desc)


def load_defs(relpath, names, namespace=None):
    """モジュール本体をimportせずに、指定した関数/クラス定義だけを読み込む。"""
    src = open(os.path.join(REPO_ROOT, relpath), encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = [n for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    missing = set(names) - {n.name for n in wanted}
    if missing:
        raise AssertionError(f"{relpath} に {sorted(missing)} が見つからない（改名された？）")
    ns = dict(namespace or {})
    exec(compile(ast.Module(body=wanted, type_ignores=[]), relpath, "exec"), ns)
    return ns


class FakeParam:
    """ndim だけを持つパラメータのスタブ（_wd_groups が見るのは ndim のみ）。"""

    def __init__(self, name, ndim):
        self.name = name
        self.ndim = ndim

    def __repr__(self):
        return self.name


class Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# =====================================================
# 1. weight decay の param group 分割
# =====================================================
print("lib/trainer/model.py:_wd_groups")
_wd_groups = load_defs("lib/trainer/model.py", ["_wd_groups"])["_wd_groups"]

# ViT-B/16 相当の内訳: 2次元以上の重み / bias / LayerNormゲイン
w = [FakeParam("qkv.weight", 2), FakeParam("proj.weight", 2)]
b = [FakeParam("qkv.bias", 1), FakeParam("norm1.weight", 1), FakeParam("norm1.bias", 1)]
tok = [FakeParam("cls_token", 3), FakeParam("pos_embed", 3)]

groups = _wd_groups(Args(wd_apply_to_bias_norm=False), [{"params": w + b + tok}])
decayed = [p for g in groups if not g.get("wd_exempt") for p in g["params"]]
exempt = [p for g in groups if g.get("wd_exempt") for p in g["params"]]

_check("bias と ndim<=1（LayerNormゲイン）が wd から外れる", set(exempt) == set(b))
_check("ndim>=2 の重みには wd が掛かったまま", set(decayed) == set(w + tok))
_check("除外グループの weight_decay は 0.0",
       all(g["weight_decay"] == 0.0 for g in groups if g.get("wd_exempt")))
_check("全パラメータがどちらかのグループに1回だけ現れる",
       sorted(p.name for p in decayed + exempt) == sorted(p.name for p in w + b + tok)
       and len(decayed) + len(exempt) == len(w + b + tok))
# cls_token/pos_embed は ndim=3 なので decay 側。DINO公式(get_params_groups)も
# `len(shape)==1 or name.endswith('.bias')` で分けており同じ挙動（MAEのみ別扱い）。
_check("cls_token / pos_embed は DINO公式と同じく decay 側", set(tok) <= set(decayed))

# グループ固有の設定（lr / fix_lr など）は分割後も両側に引き継がれること
split = _wd_groups(Args(wd_apply_to_bias_norm=False),
                   [{"params": w + b, "lr": 0.1, "fix_lr": True}])
_check("分割してもグループ固有のキー(lr/fix_lr)が失われない",
       len(split) == 2 and all(g.get("lr") == 0.1 and g.get("fix_lr") for g in split))

# opt-out（SimSiam原論文のResNetレシピ再現用）
same = _wd_groups(Args(wd_apply_to_bias_norm=True), [{"params": w + b}])
_check("--wd_apply_to_bias_norm を付けると従来通り分割しない",
       len(same) == 1 and not any(g.get("wd_exempt") for g in same))

# 重みだけ / biasだけのグループで空グループを作らないこと
_check("空グループを作らない",
       len(_wd_groups(Args(wd_apply_to_bias_norm=False), [{"params": w}])) == 1
       and len(_wd_groups(Args(wd_apply_to_bias_norm=False), [{"params": b}])) == 1)


# =====================================================
# 2. 崩壊判定
# =====================================================
print("lib/sslmodel/utils.py:CollapseMonitor")
CollapseMonitor = load_defs("lib/sslmodel/utils.py", ["CollapseMonitor"],
                            namespace={"np": np})["CollapseMonitor"]

LN8192 = float(np.log(8192))  # = 9.0109: DINO out_dim=8192 の一様崩壊値

# 0024 の実測値: eff_rank は 16〜40 を維持したまま36 epoch 崩壊し続けた
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=8192)
m.update(26.35, uniformity=-0.1757, train_loss=9.0103)
_check("一様崩壊: 1回目は patience 未達で発火しない", not m.collapsed)
m.update(20.63, uniformity=-0.1750, train_loss=9.0104)
_check("一様崩壊: eff_rank 20 でも train_loss で検出できる（旧実装では検出不能）",
       m.collapsed and "uniform collapse" in m.reason)

# 0023 の ep27-41: loss 1.7〜4.6 まで下がっていた区間は崩壊と判定しないこと
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=8192)
for loss in (4.6179, 2.9739, 1.7228):
    m.update(142.81, uniformity=-0.3084, train_loss=loss)
_check("学習が進んでいる区間を誤検知しない", not m.collapsed)

# 全サンプルが1点に潰れた場合（uniformity -> 0）。out_dim を持たない手法用の経路
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=0)
_check("out_dim 未指定なら loss 判定は無効", m.loss_ceiling is None)
m.update(50.0, uniformity=-0.01, train_loss=0.5)
m.update(50.0, uniformity=-0.02, train_loss=0.5)
_check("uniformity ~ 0 で検出できる", m.collapsed and "one point" in m.reason)

# 断続的な悪化ではカウンタがリセットされること
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=8192)
m.update(3.0, uniformity=-0.3, train_loss=5.0)
m.update(30.0, uniformity=-0.3, train_loss=5.0)
m.update(3.0, uniformity=-0.3, train_loss=5.0)
_check("連続していない低下ではカウンタがリセットされる", not m.collapsed)

# NaN は「未観測」として無視（NaN < threshold は常に False なので明示処理が要る）
m = CollapseMonitor(rank_threshold=5.0, patience=1, out_dim=8192)
m.update(float("nan"), uniformity=float("nan"), train_loss=float("nan"))
_check("NaN では発火しない", not m.collapsed)
m.update(None, uniformity=None, train_loss=None)
_check("None では発火しない", not m.collapsed)

# 後方互換: 位置引数1つだけの旧呼び出し
m = CollapseMonitor(rank_threshold=5.0, patience=1)
m.update(1.2)
_check("旧来の update(effective_rank) 単独呼び出しも動く", m.collapsed)

# 閾値のちょうど境界（ln(out_dim) * 0.999）
m = CollapseMonitor(rank_threshold=5.0, patience=1, out_dim=8192)
m.update(50.0, uniformity=-0.5, train_loss=LN8192 * 0.998)
_check("閾値をわずかに下回る loss では発火しない", not m.collapsed)
m.update(50.0, uniformity=-0.5, train_loss=LN8192)
_check("loss == ln(out_dim) で発火する", m.collapsed)


print()
print(f"passed: {PASSES}, failed: {FAILURES}")
sys.exit(1 if FAILURES else 0)
