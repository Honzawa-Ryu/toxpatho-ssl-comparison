"""BT 損失の標準化が global batch と一致すること(2026-10-12 レビュー指摘 P1)。コンテナ内で実行する:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. .venv/bin/python -m unittest tests.test_bt_global_standardize"

gloo で 2 プロセス(CPU)を立て、各 rank にバッチの半分を渡して損失と勾配を計算し、1 プロセスで全バッチに
公式実装と同じ式(BatchNorm1d(affine=False): biased 分散 + eps 1e-5、c = z_a^T z_b / N_total)を当てた値と比べる。
  - 損失は一致する(旧実装は rank ごとのローカル標準化で一致しない。数値例で相関の符号まで変わりうる)
  - 各 rank の z の勾配は、全バッチの勾配の対応する行と一致する(= SyncBN と同じ。標準化の統計にも勾配が通る)
  - std が 0 の次元があっても NaN にならない(eps)
"""
import os
import socket
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.sslmodel.models.barlowtwins import BarlowTwinsLoss

LAMBDA = 5e-3


def reference(z_a, z_b):
    bn = lambda z: (z - z.mean(0)) / (z.var(0, unbiased=False) + 1e-5).sqrt()
    c = bn(z_a).T @ bn(z_b) / z_a.size(0)
    diag = torch.diagonal(c)
    return (diag - 1).pow(2).sum() + LAMBDA * (c.pow(2).sum() - diag.pow(2).sum())


def _worker(rank, world, port, path, z_a, z_b):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    n = z_a.size(0) // world
    a = z_a[rank * n:(rank + 1) * n].clone().requires_grad_(True)
    b = z_b[rank * n:(rank + 1) * n].clone().requires_grad_(True)
    loss = BarlowTwinsLoss(lambda_param=LAMBDA, gather_distributed=True)(a, b)
    loss.backward()
    torch.save({'loss': loss.detach(), 'ga': a.grad, 'gb': b.grad}, f'{path}/r{rank}.pt')
    dist.destroy_process_group()


def _worker_param(rank, world, port, path, x_a, x_b, w0, ref):
    """DDP と同じく、パラメータの勾配を rank 間で平均したものを保存する。"""
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    n = x_a.size(0) // world
    w = w0.clone().requires_grad_(True)
    crit = BarlowTwinsLoss(lambda_param=LAMBDA, gather_distributed=True)
    crit.grad_scale = world / ref
    loss = crit(x_a[rank * n:(rank + 1) * n] @ w, x_b[rank * n:(rank + 1) * n] @ w)
    loss.backward()
    g = w.grad.clone()
    dist.all_reduce(g)
    torch.save({'loss': loss.detach(), 'grad': g / world}, f'{path}/p{rank}.pt')
    dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class TestGlobalStandardize(unittest.TestCase):
    def test_matches_full_batch_loss_and_grads(self):
        torch.manual_seed(0)
        N, D, W = 32, 6, 2
        z_a = torch.randn(N, D, dtype=torch.float64).float()
        z_b = (z_a + 0.5 * torch.randn(N, D)).float()
        z_a[: N // 2] += 3.0          # rank 間で分布をずらす(ローカル標準化だと結果が大きく変わる)
        z_b[N // 2:] *= 4.0
        ra, rb = z_a.clone().requires_grad_(True), z_b.clone().requires_grad_(True)
        ref = reference(ra, rb)
        ref.backward()
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(_worker, args=(W, _free_port(), d, z_a, z_b), nprocs=W, join=True)
            outs = [torch.load(f'{d}/r{r}.pt') for r in range(W)]
        n = N // W
        for r, o in enumerate(outs):
            self.assertAlmostEqual(o['loss'].item(), ref.item(), places=3)
            torch.testing.assert_close(o['ga'], ra.grad[r * n:(r + 1) * n], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(o['gb'], rb.grad[r * n:(r + 1) * n], rtol=1e-4, atol=1e-5)

    def test_param_grad_is_world_size_invariant_with_ref(self):
        """--bt_grad_ref_world: DDP 平均後のパラメータ勾配が world_size に依らず「真の勾配 / ref」になる。
        ref を使わない(grad_scale=1)と「真の勾配 / world_size」になり、ノード数で bias/LN の実効 lr が変わる。"""
        torch.manual_seed(2)
        N, D, H, ref = 32, 6, 5, 16
        x_a, x_b = torch.randn(N, D), torch.randn(N, D)
        w0 = torch.randn(D, H)
        w = w0.clone().requires_grad_(True)
        reference(x_a @ w, x_b @ w).backward()
        true_grad = w.grad.clone()
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(_worker_param, args=(2, _free_port(), d, x_a, x_b, w0, ref), nprocs=2, join=True)
            g2 = torch.load(f'{d}/p0.pt')['grad']
        torch.testing.assert_close(g2, true_grad / ref, rtol=1e-4, atol=1e-6)
        crit = BarlowTwinsLoss(lambda_param=LAMBDA)        # world 1、ref 16
        crit.grad_scale = 1 / ref
        w1 = w0.clone().requires_grad_(True)
        loss = crit(x_a @ w1, x_b @ w1)
        loss.backward()
        torch.testing.assert_close(w1.grad, true_grad / ref, rtol=1e-4, atol=1e-6)
        self.assertAlmostEqual(loss.item(), reference(x_a @ w0, x_b @ w0).item(), places=4, msg="ログの損失値は変わらない")

    def test_single_process_matches_reference(self):
        torch.manual_seed(1)
        z_a, z_b = torch.randn(16, 5), torch.randn(16, 5)
        got = BarlowTwinsLoss(lambda_param=LAMBDA)(z_a, z_b)
        self.assertAlmostEqual(got.item(), reference(z_a, z_b).item(), places=4)

    def test_constant_dimension_is_finite(self):
        z_a, z_b = torch.randn(16, 5), torch.randn(16, 5)
        z_a[:, 2] = 7.0                    # std == 0 の次元
        z_b[:, 0] = 300.0
        loss = BarlowTwinsLoss(lambda_param=LAMBDA)(z_a.bfloat16(), z_b.bfloat16())
        self.assertTrue(torch.isfinite(loss), "旧実装は 0/0 で NaN")


if __name__ == "__main__":
    unittest.main()
