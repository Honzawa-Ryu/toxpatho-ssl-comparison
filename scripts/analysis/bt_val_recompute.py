#!/usr/bin/env python3
"""保存済みのスナップショット(model_ep*.pt)で、BT の val 損失を学習と同じ条件で測り直す(exp 0047、1 GPU・debug-g)。

0035 の学習中の val_loss は eval モードの BN・実効 N=256 で測っていて比較にならなかった(PROJECT_STATUS 2026-10-16)。
ここでは各スナップショットを読み、val fold の固定バッチ(batch 2048 = 学習の global batch と同じ N)に対して
  - batch 統計の BN(running 統計は更新しない)・bf16  … 学習と同じ条件(本命)
  - eval モードの BN・bf16                            … 学習中の val と同じ条件(参照)
で損失と内訳(on_diag / off_diag)を測る。勾配は取らないので 1 GPU に batch 2048 が載る。

    python scripts/analysis/bt_val_recompute.py --exp 0035_20261011_bt_vitb16_lrbias_quarter_clip1 \
        --methods scripts/analysis/bt_epochs.yaml --out outputs/0047_.../bt_val_recompute.jsonl [--batches 10] [--batch 2048]
"""
import argparse
import json
import os
import re
import shlex
import sys
import tempfile
import time

import torch
import yaml

sys.path.insert(0, os.getcwd())
from lib.trainer.entry import build_parser          # noqa: E402
from lib.trainer.context import build_context       # noqa: E402
from lib.trainer.model import prepare_model         # noqa: E402
from lib.trainer.data import prepare_data           # noqa: E402
from lib.trainer.loop import batch_stat_bn          # noqa: E402


def exp_args(exp, batch):
    src = open(f'experiments/{exp}/run_slurm.sh').read()
    cmd = re.search(r'RUN_COMMAND="python \$\{PYTHON_PATH\} (.*?)"', src, re.S).group(1).replace('\\\n', ' ')
    argv = [a.replace('${PROJECT_ROOT}', os.getcwd()).replace('${EXP_NAME}', exp) for a in shlex.split(cmd)]
    argv[argv.index('--dir_result') + 1] = tempfile.mkdtemp()
    argv[argv.index('--batch_size') + 1] = str(batch)
    argv = [a for a in argv if a not in ('--resume', '--compile_blocks', '--bt_step_log', '--step_log')]
    return build_parser().parse_args(argv)


@torch.no_grad()
def measure(model, crit, batches, ssl, device, mode):
    tot, parts, n = 0.0, torch.zeros(2), 0
    ctx = batch_stat_bn(model) if mode == 'batch' else _eval(model)
    with ctx:
        for data in batches:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = ssl.calc_loss(model, data, crit)
            tot += loss.item(); parts += crit.last_parts.cpu(); n += 1
    return tot / n, (parts / n).tolist()


class _eval:
    def __init__(self, m): self.m = m
    def __enter__(self): self.was = self.m.training; self.m.eval()
    def __exit__(self, *a): self.m.train(self.was)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--methods', required=True, help='model_path を列挙した yaml(scripts/analysis/bt_epochs.yaml 等)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batches', type=int, default=10)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--limit', type=int, default=0, help='先頭 N 本だけ(スモーク用)')
    a = ap.parse_args()
    args = exp_args(a.exp, a.batch)
    ctx = build_context(args)
    device = ctx.device
    model, crit, *_ = prepare_model(ctx, model_name=args.model_name, patience=args.patience, delta=args.delta,
                                    lr=args.lr, weight_decay=args.weight_decay, num_epoch=args.num_epoch)
    _, eval_loader = prepare_data(ctx, batch_size=a.batch)
    batches = []
    for i, d in enumerate(eval_loader):            # 固定バッチ(shuffle なし)を全スナップショットで使い回す
        batches.append([x.to(device, non_blocking=True) for x in d])
        if len(batches) >= a.batches:
            break
    print(f'val batches: {len(batches)} x {a.batch} (x2 views)', flush=True)
    methods = yaml.safe_load(open(a.methods))['methods']
    if a.limit:
        methods = methods[:a.limit]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'a') as f:
        for m in methods:
            t0 = time.time()
            sd = torch.load(m['model_path'], map_location=device)
            sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) and 'model_state_dict' in sd else sd
            model.load_state_dict(sd)
            rec = dict(name=m['name'], model_path=m['model_path'], n_batches=len(batches), batch=a.batch)
            for mode in ('batch', 'eval'):
                loss, (on, off) = measure(model, crit, batches, ctx.ssl_class, device, mode)
                rec[f'{mode}_loss'], rec[f'{mode}_on_diag'], rec[f'{mode}_off_diag'] = loss, on, off
            rec['sec'] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + '\n'); f.flush()
            print(f"{rec['name']}: batch-BN loss {rec['batch_loss']:.1f} (on {rec['batch_on_diag']:.1f}, "
                  f"λ·off {crit.lambda_param * rec['batch_off_diag']:.1f}) | eval-BN loss {rec['eval_loss']:.1f}  [{rec['sec']}s]",
                  flush=True)


if __name__ == '__main__':
    main()
