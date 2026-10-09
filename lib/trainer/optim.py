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
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue
                if not g.get('weight_decay_filter', False):
                    dp = dp.add(p, alpha=g['weight_decay'])
                if not g.get('lars_adaptation_filter', False):
                    pn = torch.norm(p)
                    un = torch.norm(dp)
                    one = torch.ones_like(pn)
                    q = torch.where(pn > 0., torch.where(un > 0., g['eta'] * pn / un, one), one)
                    dp = dp.mul(q)
                state = self.state[p]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(p)
                mu = state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


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

    グループの役割は lib/trainer/model.py が付ける `lr_role`（'bias' = LARS 除外の bias/BN グループ）
    で見分ける。layer_wise_lr / fix_pred_lr の lr 比はここでは扱わない（未対応として止める）。
    """
    if getattr(args, 'layer_wise_lr', False) or getattr(args, 'fix_pred_lr', False):
        raise RuntimeError('--resume_override_lr は layer_wise_lr / fix_pred_lr と併用できない')
    lr_bias = args.lr_bias if args.lr_bias > 0 else args.lr
    base = []
    for g in optimizer.param_groups:
        v = lr_bias if g.get('lr_role') == 'bias' else args.lr
        g['lr'] = v
        g['initial_lr'] = v
        base.append(v)
    scheduler.base_values = base
    if hasattr(scheduler, 'warmup_steps') and getattr(scheduler, 'warmup_t', 0):
        scheduler.warmup_steps = [(v - scheduler.warmup_lr_init) / scheduler.warmup_t for v in base]
    # 再開する epoch の cosine 位置の値を、timm 自身の計算で各グループへ入れ直す
    # (loop.py は epoch 末に scheduler.step(epoch) を呼ぶので、epoch start_epoch の開始時の値は _get_lr(start_epoch-1))。
    if start_epoch > 0:
        scheduler.update_groups(scheduler._get_lr(start_epoch - 1))
    now = [g['lr'] for g in optimizer.param_groups]
    log(f'--resume_override_lr: base lr per group -> {base} (roles '
        f'{[g.get("lr_role", "weights") for g in optimizer.param_groups]}); lr at epoch {start_epoch + 1}: {now}')
    return now
