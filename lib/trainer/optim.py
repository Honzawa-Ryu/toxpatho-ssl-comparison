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
