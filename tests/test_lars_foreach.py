"""LARS のまとめ計算(torch._foreach_*、2026-10-13)が旧実装(パラメータごとのループ)と一致すること。コンテナ内:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_lars_foreach"
"""
import copy
import unittest

import torch

from lib.trainer.optim import LARS


class LoopLARS(torch.optim.Optimizer):
    """旧実装(2026-10-13 まで)の写し。比較の基準。"""

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.9, eta=0.001):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta,
                                      weight_decay_filter=False, lars_adaptation_filter=False))

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
                    pn, un = torch.norm(p), torch.norm(dp)
                    one = torch.ones_like(pn)
                    q = torch.where(pn > 0., torch.where(un > 0., g['eta'] * pn / un, one), one)
                    dp = dp.mul(q)
                st = self.state[p]
                if 'mu' not in st:
                    st['mu'] = torch.zeros_like(p)
                st['mu'].mul_(g['momentum']).add_(dp)
                p.add_(st['mu'], alpha=-g['lr'])


def _groups(ps):
    w, b = [p for p in ps if p.ndim > 1], [p for p in ps if p.ndim <= 1]
    return [{'params': w}, {'params': b, 'lr': 0.0384, 'weight_decay': 0.0,
                             'weight_decay_filter': True, 'lars_adaptation_filter': True}]


class TestLarsForeach(unittest.TestCase):
    def test_matches_loop_implementation(self):
        torch.manual_seed(0)
        base = [torch.randn(8, 4), torch.randn(4), torch.zeros(6, 3), torch.randn(5), torch.randn(3, 3)]
        pa = [torch.nn.Parameter(t.clone()) for t in base]
        pb = [torch.nn.Parameter(t.clone()) for t in base]
        oa = LARS(_groups(pa), lr=1.6, weight_decay=1.5e-6)
        ob = LoopLARS(_groups(pb), lr=1.6, weight_decay=1.5e-6)
        for step in range(6):
            g = [torch.randn_like(t) for t in base]
            g[2] = torch.zeros_like(base[2]) if step % 2 == 0 else g[2]   # 勾配 0・重み 0 のテンソル(q=1 の枝)
            for p, q, gg in zip(pa, pb, g):
                p.grad, q.grad = gg.clone(), gg.clone()
            pa[4].grad = pb[4].grad = None if step == 3 else pa[4].grad  # 勾配を持たない step
            oa.step()
            ob.step()
            for p, q in zip(pa, pb):
                torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)
        for p, q in zip(pa, pb):
            if 'mu' in ob.state[q]:
                torch.testing.assert_close(oa.state[p]['mu'], ob.state[q]['mu'], rtol=1e-6, atol=1e-7)

    def test_state_dict_roundtrip(self):
        ps = [torch.nn.Parameter(torch.randn(4, 4)), torch.nn.Parameter(torch.randn(4))]
        o = LARS(_groups(ps), lr=1.6)
        for p in ps:
            p.grad = torch.randn_like(p)
        o.step()
        o2 = LARS(_groups(ps), lr=1.6)
        o2.load_state_dict(copy.deepcopy(o.state_dict()))
        torch.testing.assert_close(o2.state[ps[0]]['mu'], o.state[ps[0]]['mu'])


if __name__ == "__main__":
    unittest.main()
