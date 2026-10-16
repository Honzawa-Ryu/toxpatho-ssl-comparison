#!/usr/bin/env python3
"""実験定義(run_slurm.sh)の引数そのままで model / optimizer / scheduler を組み、CPU で train_epoch を数 step 回すスモーク。
投入前に「構築が通る・損失と勾配が有限・テレメトリが書かれる」を確かめる(計算ノードは使わない)。コンテナ内で:
    apptainer exec --bind /work <SIF> bash -c "cd <repo> && PYTHONPATH=. WANDB_MODE=disabled .venv/bin/python tools/smoke_train_config.py <exp_name>"
"""
import glob, json, os, re, shlex, sys, tempfile
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.getcwd())
from lib.trainer.entry import build_parser
from lib.trainer.context import build_context
from lib.trainer.model import prepare_model
from lib.trainer.loop import train_epoch

exp = sys.argv[1]
src = open(f'experiments/{exp}/run_slurm.sh').read()
cmd = re.search(r'RUN_COMMAND="python \$\{PYTHON_PATH\} (.*?)"', src, re.S).group(1).replace('\\\n', ' ')
argv = [a.replace('${PROJECT_ROOT}', os.getcwd()).replace('${EXP_NAME}', exp) for a in shlex.split(cmd)]
out = tempfile.mkdtemp()
argv[argv.index('--dir_result') + 1] = out
argv[argv.index('--batch_size') + 1] = '4'
argv = [a for a in argv if a not in ('--resume', '--compile_blocks')]   # CPU の inductor は遅いので compile は外す
args = build_parser().parse_args(argv)
ctx = build_context(args)
model, criterion, optimizer, scheduler, early_stopping, cm = prepare_model(
    ctx, model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr,
    weight_decay=args.weight_decay, num_epoch=args.num_epoch)
print('groups:', [(len(g['params']), g['lr'], g['weight_decay'], g.get('fix_lr'), g.get('wd_exempt')) for g in optimizer.param_groups])
if args.ssl_name == 'simsiam':
    print('simsiam projector:', [m.out_features for m in model.fc if isinstance(m, torch.nn.Linear)])
tf = ctx.ssl_class.prepare_transform(color_plob=args.color_plob, blur_plob=args.blur_plob, solar_plob=args.solar_plob)
rng = np.random.default_rng(0)
imgs = [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(4)]
def batch():
    v = [tf(im) for im in imgs]
    if isinstance(v[0], (list, tuple)):
        return [torch.stack([x[k] for x in v]) for k in range(len(v[0]))]
    return [torch.stack(v)]
loader = [batch() for _ in range(3)]
if args.fix_pred_lr:
    for g in optimizer.param_groups:
        if g.get('fix_lr'): g['lr'] = args.lr
model, loss, gn = train_epoch(ctx, model, loader, criterion, optimizer, 0, num_epoch=args.num_epoch)
print(f'loss {loss:.4f} grad_norm {gn:.4f}')
f = glob.glob(f'{out}/*steps_ep0001_*.jsonl')
print('telemetry:', [os.path.basename(x) for x in f])
print('loss_parts:', getattr(ctx, 'train_loss_parts', None))
print('step_summary keys:', len(getattr(ctx, 'train_step_summary', {}) or {}))
