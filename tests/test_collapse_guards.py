# -*- coding: utf-8 -*-
"""2026-08-31のDINO恒久崩壊の再発防止テスト（純Python、torch/GPU不要）。

守っているのは次の2点:

  1. `lib/trainer/model.py:_wd_groups` が bias / ndim<=1 のパラメータを
     weight decay から必ず外すこと。外れていないと backbone の LayerNorm
     ゲインが毎step `γ <- γ(1 - lr*wd)` で削られ、ep100で初期値の0.2%まで
     消えて恒久崩壊する（0017/0021/0023/0024 が全滅した実際の原因）。
  2. `lib/sslmodel/utils.py:CollapseMonitor` が「loss が ln(out_dim) に
     張り付いた一様崩壊」を検出できること。加えて NaN/inf の train_loss を
     patience なしで即検出できること（Barlow Twins 用, 2026-09-30）。さらに BT 用に
     uniformity 判定の無効化と損失の跳ね返り(発散)判定を持つこと（2026-10-01）。旧実装は effective_rank だけを
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
from typing import Optional
CollapseMonitor = load_defs("lib/sslmodel/utils.py", ["CollapseMonitor"],
                            namespace={"np": np, "Optional": Optional})["CollapseMonitor"]

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

# 2026-09-01の誤検知の再発防止。0026 の ep15/ep20 実測値:
# loss は天井 ln(65536)=11.0904 の26%まで下降、eff_rank も 195->237 と上昇して
# いたのに、uniformity -0.0108 だけで abort した。補助指標は損失が天井付近の
# ときだけ有効にする。
LN65536 = float(np.log(65536))
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=65536)
m.update(195.46, uniformity=-0.0157, train_loss=3.4830)
m.update(237.18, uniformity=-0.0108, train_loss=2.8539)
_check("損失が下降中なら uniformity だけでは発火しない（0026の誤検知）",
       not m.collapsed)
# 同じ uniformity でも、損失が天井付近なら補助指標として機能すること
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=65536)
m.update(237.18, uniformity=-0.0108, train_loss=LN65536 * 0.95)
m.update(237.18, uniformity=-0.0108, train_loss=LN65536 * 0.95)
_check("損失が天井付近なら uniformity が補助指標として効く",
       m.collapsed and "one point" in m.reason)
# eff_rank も同じゲートに従うこと（損失が健全なら単独で止めない）
m = CollapseMonitor(rank_threshold=5.0, patience=1, out_dim=65536)
m.update(1.5, uniformity=-0.9, train_loss=2.8539)
_check("損失が下降中なら eff_rank だけでも発火しない", not m.collapsed)
# 一次指標は補助指標のゲートに関係なく単独で発火すること
m = CollapseMonitor(rank_threshold=5.0, patience=1, out_dim=65536)
m.update(237.18, uniformity=-0.9, train_loss=LN65536)
_check("train_loss が天井なら他が健全でも発火する",
       m.collapsed and "uniform collapse" in m.reason)

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


# =====================================================
# 3. NaN/inf の train_loss（2026-09-30, Barlow Twins ViT-B/16 投入前に追加）
#    BT は出力が定数に潰れると z.std(0)->0 で損失が NaN になる。update() は NaN を
#    「無効な測定値」として無視する設計なので、別経路で即停止する必要がある。
# =====================================================
print("CollapseMonitor.check_loss_finite (NaN/inf)")

for bad in (float("nan"), float("inf"), float("-inf")):
    m = CollapseMonitor(rank_threshold=5.0, patience=2)
    _check(f"train_loss={bad} は patience を待たず即崩壊", m.check_loss_finite(bad) and m.collapsed)
    _check(f"train_loss={bad} の reason に回復不能と残る", "not finite" in m.reason)

m = CollapseMonitor(rank_threshold=5.0, patience=2)
for ok in (8000.0, 1.0e-3, 0.0, -1.5, np.float32(12.5), None):
    m.check_loss_finite(ok)
_check("有限値・None では発火しない", not m.collapsed and m.reason == "")

# update() が NaN を無視する既存仕様は変えていない（eff_rank の SVD 発散でカウンタを
# リセットされないための仕様）。NaN の検知は check_loss_finite の担当。
m = CollapseMonitor(rank_threshold=5.0, patience=2, out_dim=8192)
m.update(float("nan"), uniformity=float("nan"), train_loss=float("nan"))
_check("update() は従来どおり NaN を無視する（検知は check_loss_finite 側）", not m.collapsed)

# ---- 呼び出し順の回帰テスト ----
# NaN 判定が state.pt 保存・early_stopping() より「後」にあると、EarlyStopping が
# `NaN > best == False` で「改善」扱いとなり NaN 重みを checkpoint.pt(abort時の復元元)へ
# 保存し、state.pt も NaN 重みで上書きされる。順序そのものが防御なのでソースで固定する。
loop_src = open(os.path.join(REPO_ROOT, "lib/trainer/loop.py"), encoding="utf-8").read()
i_check = loop_src.index("collapse_monitor.check_loss_finite(")
i_state = loop_src.index("torch.save(state, f'{DIR_NAME}/state.pt')")
i_early = loop_src.index("early_stopping(val_epoch_loss")
_check("check_loss_finite は state.pt 保存より前", i_check < i_state)
_check("check_loss_finite は early_stopping() より前", i_check < i_early)
_check("復元元が無ければ NaN 重みを書き出さず例外で止める",
       "refusing to write NaN weights" in loop_src)


# =====================================================
# 4. BT 用の設定（2026-10-01, 投入前に追加）
#    (a) uniformity 判定の無効化: BT は out_dim を持たないので補助指標が常時有効になり、
#        DINO の健全時でさえ -0.006 の uniformity で最初の判定(ep10)に健全なランを誤停止しうる。
#    (b) 損失の持続的な跳ね返り(NaN にならない発散)の判定。
# =====================================================
print("CollapseMonitor: uniformity 無効化 / 損失の跳ね返り")

# (a) 既定(-0.05)では、out_dim 無しだと健全な uniformity ~ -0.006 でも発火する = 誤停止の再現
m = CollapseMonitor(rank_threshold=5.0, patience=2)
for _ in range(2):
    m.update(440.0, uniformity=-0.0056, train_loss=1.0)
_check("[再現] 既定だと out_dim 無しで健全な uniformity(-0.006)でも誤停止する", m.collapsed)

m = CollapseMonitor(rank_threshold=5.0, patience=2, uniformity_threshold=None)
for _ in range(5):
    m.update(440.0, uniformity=-0.0056, train_loss=1.0)
_check("uniformity_threshold=None なら uniformity ~ 0 でも停止しない", not m.collapsed)
m = CollapseMonitor(rank_threshold=5.0, patience=2, uniformity_threshold=None)
for _ in range(2):
    m.update(1.2, uniformity=-0.0056, train_loss=1.0)
_check("uniformity を無効にしても eff_rank < 閾値の判定は生きている", m.collapsed)

# (b) 跳ね返り判定
def run_rebound(losses, start_epoch=1, **kw):
    mm = CollapseMonitor(rank_threshold=5.0, patience=2, loss_rebound_ratio=2.0, **kw)
    fired_at = None
    for i, l in enumerate(losses):
        if mm.check_loss_rebound(start_epoch + i, l):
            fired_at = start_epoch + i
            break
    return mm, fired_at

healthy = [8000.0 - 100.0 * i for i in range(60)]  # 単調に下がる健全な曲線
mm, at = run_rebound(healthy)
_check("単調に下がる健全な損失では発火しない", at is None and not mm.collapsed)

# ep20 から基準(最小値)を取る: ep1-19 の値は基準にも判定にも使わない
early_spike = [50000.0] * 19 + [3000.0] * 10
mm, at = run_rebound(early_spike)
_check("min_epoch(20) より前の大きな値は無視する(warmup の動きに引きずられない)", at is None)

base = [1000.0] * 25   # ep1..25 の最小値は 1000
mm, at = run_rebound(base + [2500.0, 2500.0, 2500.0])
_check("最小値の2倍超が3 epoch続くと発散と判定する", at == 28 and "diverged" in mm.reason)
mm, at = run_rebound(base + [2500.0, 2500.0, 1500.0, 2500.0, 2500.0])
_check("途中で戻ればカウンタがリセットされる(連続でなければ発火しない)", at is None)
mm, at = run_rebound(base + [1999.0] * 10)
_check("ちょうど2倍未満なら発火しない", at is None)

mm, at = run_rebound(base + [float("nan"), 2500.0, 2500.0])
_check("非有限値は rebound 側では扱わない(check_loss_finite の担当)", at is None)

m = CollapseMonitor(rank_threshold=5.0, patience=2)  # 既定 ratio=0 = 無効
_check("既定(ratio=0)では無効 = 既存の実験の挙動は変わらない",
       not any(m.check_loss_rebound(e, l) for e, l in enumerate([1.0] * 30 + [1e6] * 10, start=1)))

# 配線(entry.py のCLI -> model.py の構築 -> loop.py の呼び出し順)
entry_src = open(os.path.join(REPO_ROOT, "lib/trainer/entry.py"), encoding="utf-8").read()
model_src = open(os.path.join(REPO_ROOT, "lib/trainer/model.py"), encoding="utf-8").read()
for flag in ("--collapse_ignore_uniformity", "--collapse_loss_rebound",
             "--collapse_loss_rebound_min_epoch", "--collapse_loss_rebound_patience"):
    _check(f"CLI {flag} が定義されている", f"add_argument('{flag}'" in entry_src)
for kw in ("uniformity_threshold=None if args.collapse_ignore_uniformity",
           "loss_rebound_ratio=args.collapse_loss_rebound",
           "loss_rebound_min_epoch=args.collapse_loss_rebound_min_epoch",
           "loss_rebound_patience=args.collapse_loss_rebound_patience"):
    _check(f"model.py が {kw.split('=')[0]} を CollapseMonitor へ渡している", kw in model_src)
i_reb = loop_src_after = open(os.path.join(REPO_ROOT, "lib/trainer/loop.py"), encoding="utf-8").read()
i_rebound = i_reb.index("collapse_monitor.check_loss_rebound(")
_check("check_loss_rebound は state.pt 保存より前", i_rebound < i_reb.index("torch.save(state, f'{DIR_NAME}/state.pt')"))
_check("check_loss_rebound は early_stopping() より前", i_rebound < i_reb.index("early_stopping(val_epoch_loss"))


# =====================================================
# 5. BT の fp32 ヘッド / step テレメトリの配線（2026-10-05, 0029 の NaN 対応）
#    torch が要る挙動は lib/sslmodel/tests/test_barlowtwins_fp32.py（コンテナ内）が担当。
#    ここでは「配線を忘れると黙って効かなくなる」箇所をソースで固定する。
# =====================================================
print("BT fp32 head / step telemetry wiring")
sslutils_src = open(os.path.join(REPO_ROOT, "lib/sslmodel/sslutils.py"), encoding="utf-8").read()
for flag in ("--bt_fp32_head", "--bt_step_log"):
    _check(f"CLI {flag} が定義されている", f"add_argument('{flag}'" in entry_src)
_check("model.py が fp32_head を BT の prepare_model へ渡している",
       'model_kwargs["fp32_head"] = True' in model_src)
_check("model.py が collect_stats を criterion へ設定している",
       "criterion.collect_stats = args.bt_step_log" in model_src)
_check("sslutils が fp32_head を モデルと損失の両方へ渡している(片方だけでは桁落ちが残る)",
       "fp32_head=fp32_head)" in sslutils_src and "fp32=fp32_head)" in sslutils_src)
# 旧 state.pt から復元した criterion は pickle 時点の属性しか持たない。再開時に CLI の値を
# 設定し直さないと、--bt_fp32_head を付けても損失が bf16 のまま走る（診断ランで実際に踏む経路）。
i_restore = entry_src.index("criterion = state['criterion']")
i_reapply = entry_src.index("criterion.fp32 = args.bt_fp32_head")
_check("resume 経路で復元後の criterion に fp32 / collect_stats を設定し直す", i_reapply > i_restore)
_check("resume 経路で collect_stats も設定し直す", "criterion.collect_stats = args.bt_step_log" in entry_src)
bt_src = open(os.path.join(REPO_ROOT, "lib/sslmodel/models/barlowtwins.py"), encoding="utf-8").read()
for attr in ("    fp32 = False", "    collect_stats = False", "    last_stats = None"):
    _check(f"旧 pickle 互換のクラス属性 `{attr.strip()}` がある", attr in bt_src)
# グループ別 grad_norm(2026-10-07): probe は grads と同じパラメータ列で record し、epoch 末に names と rows を書く
for kw in ("bt_probe = GradGroupProbe(model) if bt_stats is not None else None",
           "bt_probe.record(grad_params, per_param_norms)",
           "write_param_names(ctx.dir_name, bt_probe.names)",
           "groups=bt_probe.rows()"):
    _check(f"loop.py にグループ別 grad_norm の配線 `{kw[:40]}` がある", kw in loop_src)
_check("record は grads(= grad_params)と同じ列を渡す", "grads = [p.grad.detach() for p in grad_params]" in loop_src)
# 非有限の損失で epoch を打ち切る処理は、loss_value を append した直後・DDP 集合通信の前にあること
i_append = loop_src.index("train_batch_loss.append(loss_value)")
i_break = loop_src.index("first_nonfinite = i")
_check("非有限の損失での打ち切りは loss_value の append 直後（全 rank 同一の値で判定）",
       0 < i_break - i_append < 300)
_check("打ち切りは --collapse_early_stop のときだけ（既存の実験の挙動は変えない）",
       "stop_on_nonfinite = getattr(ctx.args, 'collapse_early_stop', False)" in loop_src)
_check("テレメトリは rank0 だけが貯める",
       "getattr(criterion, 'collect_stats', False) and distributed.is_main_process()" in loop_src)

i_es = entry_src.index("early_stopping = state['early_stopping']")
_check("resume 経路で復元した EarlyStopping の path を今回の DIR_NAME に合わせ直す(元のランの成果物を上書きしない)",
       entry_src.index("early_stopping.path = f'{DIR_NAME}/checkpoint.pt'") > i_es)
_check("resume 経路で復元した EarlyStopping の save_enabled を rank0 限定に合わせ直す(全rankの同時書き込みを防ぐ)",
       entry_src.index("early_stopping.save_enabled = distributed.is_main_process()") > i_es)


print()
print(f"passed: {PASSES}, failed: {FAILURES}")
sys.exit(1 if FAILURES else 0)
