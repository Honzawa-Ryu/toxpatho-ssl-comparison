#!/usr/bin/env python3
"""学習の律速を切り分けるプロファイル(1 GPU・計算ノードで実行。exp 0041 が呼ぶ)。

各実験の run_slurm.sh と同じ引数でモデル・optimizer・損失を組み、1 GPU あたりのバッチを変えながら次を測る:
  synth      合成データ(GPU 上の乱数)で forward+backward+optimizer.step。データ供給を除いた GPU 側の上限
  fwdbwd     そのうち forward+backward だけ / opt  optimizer.step だけ(LARS の Python ループの重さ)
  loader     DataLoader だけを回したときの供給能力(GPU を使わない。実データ・実拡張・16 worker)
  real       本番と同じ train_epoch(実データ・テレメトリ込み)を数十 step 回したときの速度
  profile    本番のバッチでだけ torch.profiler の内訳(CPU / CUDA 時間の上位)をテキストに書く
peak_mem はそのバッチでの GPU メモリのピーク。OOM はその旨を記録して次へ進む。

    python scripts/train/profile_throughput.py --out <dir> \
        --exp 0035_20261011_bt_vitb16_lrbias_quarter_clip1:256,512,1024 \
        --exp 0040_20261013_mae_vitb16_lr_half:512,1024,2048

読み方: real ≈ synth なら GPU 律速(→ バッチを上げて GPU を埋める / ノードを減らす)、
real ≈ loader なら CPU のデータ供給が律速(→ worker 数・拡張の軽量化)、どちらより遅ければ step ごとの
CPU 側オーバーヘッド(同期・テレメトリ・optimizer)。1 GPU なのでノード間通信は含まない
(本番の 1 GPU あたりの速度との差が通信のコスト)。
"""
import argparse
import contextlib
import gc
import json
import os
import re
import shlex
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.getcwd())
from lib.trainer.entry import build_parser                    # noqa: E402
from lib.trainer.context import build_context                 # noqa: E402
from lib.trainer.model import prepare_model                   # noqa: E402
from lib.trainer.data import prepare_data                     # noqa: E402
from lib.trainer.loop import train_epoch                      # noqa: E402


N_SYNTH, N_SUB, N_LOAD = 15, 10, 20


EXTRA_ARGS = []   # --compile 指定時に全構成へ足す引数


def exp_args(exp, batch, out_dir, base_batch=None):
    src = open(f'experiments/{exp}/run_slurm.sh').read()
    cmd = re.search(r'RUN_COMMAND="python \$\{PYTHON_PATH\} (.*?)"', src, re.S).group(1).replace('\\\n', ' ')
    argv = [a.replace('${PROJECT_ROOT}', os.getcwd()).replace('${EXP_NAME}', exp) for a in shlex.split(cmd)]
    argv[argv.index('--dir_result') + 1] = out_dir
    argv[argv.index('--batch_size') + 1] = str(batch)
    if '--resume' in argv:
        argv.remove('--resume')
    # compile の有無はプロファイル側(--compile → EXTRA_ARGS)だけで決める。本番の run_slurm.sh に --compile_blocks が
    # 入っていても読み込み時に外す(外さないと「compile なし」の測定にも compile が入り比較が成立しない。レビュー指摘)。
    argv = [x for x in argv if x != '--compile_blocks']
    if base_batch:
        # ホスト RAM の先読み量を本番のバッチと同じに保つ(4 × 本番バッチ / このバッチ)。
        # 0041 の初回は prefetch 4 のまま batch 512 にしてノードの RAM 上限(cgroup)で落ちた。
        argv += ['--loader_prefetch', str(max(1, round(4 * base_batch / batch)))]
    return build_parser().parse_args(argv + EXTRA_ARGS)


def synthetic_batch(args, ssl_class, b, device):
    if args.ssl_name == 'mae':
        return [torch.randn(b, 3, 224, 224, device=device)]
    if args.ssl_name == 'dino':
        return [torch.randn(b, 3, 224, 224, device=device) for _ in range(ssl_class.n_global_crops)]
    return [torch.randn(b, 3, 224, 224, device=device), torch.randn(b, 3, 224, 224, device=device)]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, n, warm):
    for _ in range(warm):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    _sync()
    return (time.perf_counter() - t0) / n


class Truncated:
    """train_epoch に渡す、先頭 n step だけの DataLoader。各 step の開始時刻を記録する。"""

    def __init__(self, loader, n):
        self.loader, self.n, self.t = loader, n, []

    def __len__(self):
        return self.n

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                break
            self.t.append(time.perf_counter())
            yield batch
        self.t.append(time.perf_counter())


def host_mem():
    """ノードのホスト RAM(GB): 使用中・ページキャッシュ込みの合計(/proc/meminfo)。"""
    m = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
    return round((m['MemTotal'] - m['MemAvailable']) / 2**20, 1)


def measure(exp, b, out_root, steps, profile, base_batch):
    rec = dict(exp=exp, batch_per_gpu=b)
    out_dir = tempfile.mkdtemp(dir=out_root, prefix=f'{exp[:4]}_b{b}_')
    args = exp_args(exp, b, out_dir, base_batch)
    rec['loader_prefetch'] = args.loader_prefetch
    rec['compile_blocks'] = bool(getattr(args, 'compile_blocks', False))
    rec['host_mem_start_gb'] = host_mem()
    ctx = build_context(args)
    dev = ctx.device
    cuda = torch.cuda.is_available()      # CPU でも動く(ログインノードでのスモーク用。数値は意味を持たない)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model, criterion, optimizer, scheduler, _, _ = prepare_model(
        ctx, model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr,
        weight_decay=args.weight_decay, num_epoch=args.num_epoch)
    ssl = ctx.ssl_class
    for g in optimizer.param_groups:      # warmup の 1e-5 ではなくピーク付近の lr で測る(更新の重さは同じ)
        g['lr'] = g.get('initial_lr', args.lr)
    data = synthetic_batch(args, ssl, b, dev)

    def fwdbwd():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = ssl.calc_loss(model, data, criterion)
        loss.backward()

    def full():
        fwdbwd()
        optimizer.step()

    try:
        warm = (10 if rec['compile_blocks'] else 5) if cuda else 1     # 初回のコンパイルを計測から外す
        t_c = time.perf_counter()
        full()
        _sync()
        rec['first_step_s'] = time.perf_counter() - t_c               # コンパイル込みの最初の 1 step
        rec['synth_s'] = timed(full, N_SYNTH, warm)
        rec['fwdbwd_s'] = timed(fwdbwd, N_SUB, 2 if cuda else 0)
        rec['opt_s'] = timed(optimizer.step, N_SUB, 2 if cuda else 0)
        rec['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2**30 if cuda else float('nan')
        rec['synth_img_s'] = b / rec['synth_s']
    except torch.cuda.OutOfMemoryError:
        rec['oom'] = True
        print(json.dumps(rec), flush=True)
        del model, optimizer, criterion, data
        torch.cuda.empty_cache()
        return rec
    del data
    torch.cuda.empty_cache()

    # 各フェーズで DataLoader を作り直す(persistent_workers の iterator を使い回すと reset 待ちで止まることがある)
    def fresh_loader():
        return prepare_data(ctx, batch_size=b)[0]

    def close(loader):
        it = getattr(loader, '_iterator', None)
        if it is not None:
            it._shutdown_workers()
        loader._iterator = None

    loader = fresh_loader()
    it = iter(loader)
    for _ in range(3):
        next(it)
    t0 = time.perf_counter()
    for _ in range(N_LOAD):
        next(it)
    rec['loader_img_s'] = N_LOAD * b / (time.perf_counter() - t0)
    rec['host_mem_loader_gb'] = host_mem()
    close(loader)
    del loader, it
    gc.collect()

    loader = fresh_loader()
    tr = Truncated(loader, steps)
    train_epoch(ctx, model, tr, criterion, optimizer, 0, num_epoch=args.num_epoch)
    rec['host_mem_real_gb'] = host_mem()
    close(loader)
    del loader
    gc.collect()
    skip = min(5, max(0, steps - 2))
    dts = [b2 - a for a, b2 in zip(tr.t[skip:-1], tr.t[skip + 1:])]   # 先頭 skip step を除く
    rec['real_s'] = sum(dts) / len(dts)
    rec['real_img_s'] = b / rec['real_s']

    if profile:
        from torch.profiler import profile as tprof, ProfilerActivity
        loader = fresh_loader()
        tr = Truncated(loader, 8 if cuda else 2)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        with tprof(activities=acts) as prof:
            train_epoch(ctx, model, tr, criterion, optimizer, 0, num_epoch=args.num_epoch)
        close(loader)
        pname = f"profile_{exp[:4]}_b{b}{'_compile' if rec['compile_blocks'] else ''}.txt"
        with open(os.path.join(out_root, pname), 'w') as f:
            ka = prof.key_averages()
            if cuda:
                f.write('### sort by self CUDA time\n' + ka.table(sort_by='self_cuda_time_total', row_limit=30) + '\n')
            f.write('### sort by self CPU time\n' + ka.table(sort_by='self_cpu_time_total', row_limit=30) + '\n')
        rec['profile'] = pname

    print(json.dumps(rec), flush=True)
    del model, optimizer, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--exp', action='append', required=True, help='<experiment dir>:<batch,batch,...> 先頭が本番のバッチ')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--smoke', action='store_true', help='CPU で配線だけ確かめる(回数を最小に)')
    ap.add_argument('--compile', action='store_true', help='全構成に --compile_blocks を付ける(初回のコンパイルは warmup に含まれる)')
    a = ap.parse_args()
    if a.compile:
        EXTRA_ARGS.append('--compile_blocks')
    if a.smoke:
        global N_SYNTH, N_SUB, N_LOAD
        N_SYNTH, N_SUB, N_LOAD = 1, 1, 2
        import lib.trainer.data as data_mod     # ログインノードでは worker を 2 に絞る(本番の 16 は計算ノード向け)
        _DL = data_mod.DataLoader
        data_mod.DataLoader = lambda *x, **k: _DL(*x, **{**k, 'num_workers': min(2, k.get('num_workers', 0)),
                                                       'prefetch_factor': 2 if k.get('num_workers') else None})
    os.makedirs(a.out, exist_ok=True)
    results = []
    for spec in a.exp:
        exp, batches = spec.split(':')
        bs = [int(x) for x in batches.split(',')]
        for k, b in enumerate(bs):
            try:
                r = measure(exp, b, a.out, a.steps, profile=(k == 0), base_batch=bs[0])
            except Exception as e:      # 1 構成の失敗で全体を止めない
                r = dict(exp=exp, batch_per_gpu=b, error=f'{type(e).__name__}: {e}')
                print(json.dumps(r), flush=True)
            results.append(r)
            with open(os.path.join(a.out, 'profile_results.jsonl'), 'a') as f:
                f.write(json.dumps(r) + '\n')
    print('\n exp  batch  synth img/s  fwd+bwd s  opt s  loader img/s  real img/s  peak GB')
    for r in results:
        if 'error' in r or r.get('oom'):
            print(f" {r['exp'][:4]}  {r['batch_per_gpu']:5d}  {'OOM' if r.get('oom') else r['error'][:60]}")
            continue
        print(f" {r['exp'][:4]}  {r['batch_per_gpu']:5d}  {r['synth_img_s']:11.0f}  {r['fwdbwd_s']:9.3f}  {r['opt_s']:5.3f}"
              f"  {r.get('loader_img_s', float('nan')):12.0f}  {r.get('real_img_s', float('nan')):10.0f}  {r['peak_mem_gb']:7.1f}")


if __name__ == '__main__':
    main()
