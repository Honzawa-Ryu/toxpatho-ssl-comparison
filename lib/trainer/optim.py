# -*- coding: utf-8 -*-
"""
# オプティマイザ

LARS の実装と --optimizer によるビルダー。
分散学習に移行する際は、ここに world_size による学習率スケーリングを足す
（REFACTOR_PLAN.md §6-4）。

"""
import torch
import torch.optim as optim

from lib.trainer.context import RunContext


class LARS(optim.Optimizer):
    """LARS (You et al. 2017), matching the Barlow Twins / SwAV official recipe.

    Per-group flags (True = skip): weight_decay_filter, lars_adaptation_filter.
    Used to exclude bias & BatchNorm params (ndim<=1) from weight decay and LARS
    trust-ratio adaptation, exactly as in the Barlow Twins reference implementation.
    """
    def __init__(self, params, lr, weight_decay=0.0, momentum=0.9, eta=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta,
                        weight_decay_filter=False, lars_adaptation_filter=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        # グループ内のテンソルをまとめて計算する(torch._foreach_*)。旧実装はパラメータごとの Python ループで、
        # BT の GPU 時間の約 9% を占めていた(exp 0041 の profiler)。計算は旧実装と同じ
        # (tests/test_lars_foreach.py が旧実装との一致を固定している)。
        for g in self.param_groups:
            params = [p for p in g['params'] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            if not g.get('weight_decay_filter', False):
                dps = torch._foreach_add(grads, params, alpha=g['weight_decay'])
            elif not g.get('lars_adaptation_filter', False):
                dps = [d.clone() for d in grads]      # 下で in-place に q を掛けるので勾配そのものは書き換えない
            else:
                dps = grads                           # 読むだけ(bias/LN グループ: wd も LARS 適応も無し)
            if not g.get('lars_adaptation_filter', False):
                pn = torch.stack(torch._foreach_norm(params))
                un = torch.stack(torch._foreach_norm(dps))
                one = torch.ones_like(pn)
                q = torch.where(pn > 0., torch.where(un > 0., g['eta'] * pn / un, one), one)
                torch._foreach_mul_(dps, list(q.unbind(0)))
            mus = []
            for p in params:
                state = self.state[p]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(p)
                mus.append(state['mu'])
            torch._foreach_mul_(mus, g['momentum'])
            torch._foreach_add_(mus, dps)
            torch._foreach_add_(params, mus, alpha=-g['lr'])


def build_optimizer(ctx: RunContext, params, lr, weight_decay):
    """Optimizer per --optimizer. AdamW (decoupled weight decay) is the standard
    for ViT self-supervised learning (MAE/DINO/MoCo-v3/iBOT); kept as the default.
    LARS is used for the large-batch Barlow Twins / SwAV recipes."""
    args = ctx.args
    name = args.optimizer.lower()
    if name == 'adamw':
        return optim.AdamW(params, lr=lr, betas=(0.9, args.beta2), weight_decay=weight_decay, fused=True)
    if name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, fused=True)
    if name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, fused=True)
    if name == 'lars':
        return LARS(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def override_lr_after_resume(args, optimizer, scheduler, start_epoch, log):
    """--resume_override_lr: 再開後に CLI の --lr / --lr_bias を optimizer と scheduler に掛け直す。

    `optimizer.load_state_dict` は param_groups の lr/initial_lr を保存時の値に戻し、timm の
    `Scheduler.load_state_dict` は `__dict__.update` で `base_values`（各グループの基準 lr）も
    保存時の値に戻す。そのため state.pt から再開するとき CLI で lr_bias を変えても、
    **黙って保存時の値で走る**（0029 の ep64 から lr_bias を変えて A/B する exp 0032 で踏む経路）。
    通常の walltime 再開（0028）は値を変えないので、このフラグが無ければ何もしない。

    ⚠️ グループの役割は **params の形で見分ける**（全テンソルが ndim<=1 なら bias/BN グループ）。
    `optimizer.load_state_dict` は param_groups の dict を保存時のものに丸ごと置き換える
    （`Optimizer.update_group` は saved group を返し、現在の構築で付けたキーは捨てる）ため、
    model.py が付ける `lr_role` のような自前のキーは古い state.pt から再開すると消える。
    exp 0032（job 3513691）はこれを踏み、bias/LN グループに lr 1.6 が入って step 5 で NaN になった（無効）。
    layer_wise_lr / fix_pred_lr の lr 比はここでは扱わない（未対応として止める）。
    """
    if getattr(args, 'layer_wise_lr', False) or getattr(args, 'fix_pred_lr', False):
        raise RuntimeError('--resume_override_lr は layer_wise_lr / fix_pred_lr と併用できない')
    lr_bias = args.lr_bias if args.lr_bias > 0 else args.lr
    base = []
    for g in optimizer.param_groups:
        is_bias_group = len(g['params']) > 0 and all(p.ndim <= 1 for p in g['params'])
        g['lr_role'] = 'bias' if is_bias_group else 'weights'   # ログ用に付け直す(復元で消えている)
        v = lr_bias if is_bias_group else args.lr
        g['lr'] = v
        g['initial_lr'] = v
        base.append(v)
    scheduler.base_values = base
    if hasattr(scheduler, 'warmup_steps') and getattr(scheduler, 'warmup_t', 0):
        scheduler.warmup_steps = [(v - scheduler.warmup_lr_init) / scheduler.warmup_t for v in base]
    # 再開する epoch の cosine 位置の値を、timm 自身の計算で各グループへ入れ直す
    # (loop.py は epoch 末に scheduler.step(epoch + 1) を呼ぶので、epoch start_epoch の値は _get_lr(start_epoch))。
    scheduler.update_groups(scheduler._get_lr(start_epoch))
    now = [g['lr'] for g in optimizer.param_groups]
    roles = [g['lr_role'] for g in optimizer.param_groups]
    log(f'--resume_override_lr: base lr per group -> {base} (roles {roles}); lr at epoch {start_epoch + 1}: {now}')
    if args.lr_bias > 0 and args.lr_bias != args.lr and 'bias' not in roles:
        raise RuntimeError('--resume_override_lr: bias/BN グループが見つからない(--lr_bias が掛からない)')
    return now
