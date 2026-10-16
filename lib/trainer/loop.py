# -*- coding: utf-8 -*-
"""
# 学習ループ

分散学習に移行する際は、ログ・チェックポイント書き出しを rank0 のみに限定する
ガードをここに入れる（REFACTOR_PLAN.md §6-4）。

"""
import contextlib
import math
import os
import re
import time

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import lib.sslmodel.evaluation as ssl_eval
from lib.trainer import distributed
from lib.trainer.context import RunContext
from lib.trainer.data import prepare_data
from lib.trainer.clip_logging import write_epoch as write_clip_epoch
from lib.trainer.bt_telemetry import (write_epoch as write_bt_epoch, StepProbe, write_param_names,
                                     STAT_NAMES as BT_STAT_NAMES, SIMSIAM_STAT_NAMES)


class LossPartsMeter:
    """損失の内訳(criterion.PART_NAMES / criterion.last_parts を持つ損失、今は BT)の epoch 平均。

    step ごとにデバイス上で足すだけで .item() の同期は増やさない(epoch 末に 1 回だけ取り出す)。
    BT は on_diag(不変性の項)・off_diag(冗長性削減の項、λ を掛ける前)と、λ·off_diag(損失への寄与)を出す。
    """

    def __init__(self, criterion):
        self.names = getattr(criterion, 'PART_NAMES', None)
        self.sum, self.n = None, 0

    def before(self, criterion):
        if self.names:
            criterion.last_parts = None     # 古い値を二重に数えない

    def add(self, criterion):
        parts = getattr(criterion, 'last_parts', None) if self.names else None
        if parts is None:
            return
        self.sum = parts.clone() if self.sum is None else self.sum + parts
        self.n += 1

    def result(self, prefix, criterion):
        if self.sum is None or self.n == 0:
            return {}
        vals = (self.sum / self.n).tolist()
        out = {f'{prefix}_{k}': v for k, v in zip(self.names, vals)}
        lam = getattr(criterion, 'lambda_param', None)
        if lam is not None and 'off_diag' in self.names:
            out[f'{prefix}_off_diag_weighted'] = lam * vals[self.names.index('off_diag')]
        return out


# step テレメトリの epoch 要約として WandB に出す列(2026-10-14)。jsonl を開かなくても崩壊・発散の前兆をグラフで追えるように。
# 値は epoch 内の step の平均。スパイクを見たいものは max、潰れを見たいものは min も出す。
STEP_SUMMARY = {
    # 全手法(StepProbe)
    'gn_raw': ('mean', 'max'), 'gn_lars': ('mean',), 'gn_head': ('mean',),
    'upd_raw': ('mean', 'max'), 'upd_w': ('mean',), 'upd_ratio_max': ('max',),
    'ln_gain_mean': ('mean',), 'ln_gain_min': ('min',),
    'act_patch': ('mean', 'max'), 'act_b0': ('mean', 'max'),
    # BT
    'z_std_min': ('min',), 'n_zero_std': ('max',), 'z_ratio_max': ('max',), 'c_diag_mean': ('mean',),
    # SimSiam
    'z_erank': ('mean', 'min'), 'p_erank': ('mean', 'min'), 'cos_zz': ('mean', 'max'), 'cos_pz': ('mean',),
    'z_nstd_sqrtd': ('mean',),
}


def summarize_steps(columns, rows, grad_norms):
    """step ごとの行(columns と同じ並びの値のリスト)を epoch 要約の dict にする(WandB のキーは steps/<列>_<集計>)。
    非有限の値は除く(NaN の step があっても他の step の要約は出す)。最後の block の出力(act_bN)も平均と最大を出す。"""
    out = {}
    last_block = max((c for c in columns if re.fullmatch(r'act_b\d+', c)), key=lambda c: int(c[5:]), default=None)
    wanted = dict(STEP_SUMMARY)
    if last_block and last_block != 'act_b0':
        wanted[last_block] = ('mean', 'max')
    for j, c in enumerate(columns):
        if c not in wanted:
            continue
        v = [r[j] for r in rows if j < len(r) and isinstance(r[j], float) and math.isfinite(r[j])]
        if not v:
            continue
        for agg in wanted[c]:
            out[f'steps/{c}_{agg}'] = float(np.mean(v) if agg == 'mean' else (max(v) if agg == 'max' else min(v)))
    g = [x for x in grad_norms if math.isfinite(x)]
    if g:
        out['steps/grad_norm_max'] = float(max(g))
    return out


def _fmt_parts(parts):
    return ', '.join(f'{k}: {v:.4g}' for k, v in parts.items())


# train epoch

def train_epoch(ctx: RunContext, model, data_loader, criterion, optimizer, epoch, num_epoch=None):
    ssl_class = ctx.ssl_class
    clip_grad = getattr(ctx.args, 'clip_grad', 0.0)
    model.train()
    train_batch_loss = []
    grad_norms = []
    clip_norms = [] if clip_grad > 0 and distributed.is_main_process() else None
    # Barlow Twins の step 単位テレメトリ(--bt_step_log)。rank0 のローカルバッチの統計を貯め、
    # epoch 末に1回だけ host へ転送して書く(bt_telemetry.py)。学習の挙動には影響しない。
    # --step_log(全手法)/ --bt_step_log(旧名)で有効。MAE は criterion を持たないので統計は無く、損失と勾配だけ。
    step_log = (getattr(ctx.args, 'step_log', False) or getattr(ctx.args, 'bt_step_log', False)
                or getattr(criterion, 'collect_stats', False))
    bt_stats = [] if step_log and distributed.is_main_process() else None
    # グループ別 grad_norm(重み/bias・LN, backbone/head)と上位テンソル。0030 で「起点は勾配の急増」
    # まで絞れたが、どのグループが先に跳ねたかが合計ノルムからは分からなかった(bt_telemetry.py)。
    # 勾配に加えて、更新量・重み・LN ゲイン・backbone の各 block の出力の大きさも step ごとに残す(bt_telemetry.StepProbe)
    bt_probe = StepProbe(model) if bt_stats is not None else None
    # --collapse_early_stop のときは、最初の非有限の損失で epoch を打ち切る。どのみち epoch 末に
    # check_loss_finite が全 rank で停止するが、残りの step は NaN の重みで空回りするだけで、
    # 打ち切れば最大 1 epoch 分の計算を節約でき、テレメトリも最初の NaN の直後で止まる。
    # loss_value は all_reduce_mean 済みで全 rank 同一なので、全 rank が同じ step で抜ける
    # (片 rank だけ抜けると他 rank が DDP の集合通信で hang する)。
    stop_on_nonfinite = getattr(ctx.args, 'collapse_early_stop', False)
    first_nonfinite = None
    parts_meter = LossPartsMeter(criterion)
    # DINOのteacher momentum/温度スケジュールはstep単位で更新する(論文と同じ粒度)。
    # 通算stepの純関数として計算するのでresume時も正しい値になる。
    niter_per_ep = len(data_loader)
    total_steps = (num_epoch or 0) * niter_per_ep
    for i, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch + 1}")):
        global_step = epoch * niter_per_ep + i
        raw = distributed.unwrap(model)
        if hasattr(raw, 'update_teacher_momentum'):
            raw.update_teacher_momentum(global_step, total_steps)
        if hasattr(criterion, 'update_teacher_temp'):
            criterion.update_teacher_temp(global_step)
        optimizer.zero_grad(set_to_none=True)
        if bt_probe is not None:
            bt_probe.active = True    # この forward の block 出力だけを測る
        parts_meter.before(criterion)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ssl_class.calc_loss(model, data, criterion)
        parts_meter.add(criterion)
        if bt_probe is not None:
            bt_probe.active = False
        if bt_stats is not None and getattr(criterion, 'last_stats', None) is not None:
            bt_stats.append(criterion.last_stats)
        loss.backward()
        # total grad L2 norm with a SINGLE device sync (old code did .item() per
        # parameter -> ~150 GPU->CPU syncs/iter, serializing the step). Identical value.
        # クリッピングより前に測る(公式DINOと同じ)ので、ログのgrad_normは常に
        # 「クリップ前」の値=スパイク検出用の生の指標のまま。
        grad_params = [p for p in model.parameters() if p.grad is not None]
        grads = [p.grad.detach() for p in grad_params]
        if grads:
            per_param_norms = torch._foreach_norm(grads)
            grad_norm = torch.norm(torch.stack(per_param_norms)).item()
            if bt_probe is not None:
                bt_probe.record(grad_params, per_param_norms)
            if clip_norms is not None:
                clip_norms.append(torch.stack(per_param_norms).detach())
            if clip_grad > 0:
                # DINO公式(utils.clip_gradients)と同じ「パラメータ毎」のクリップ。
                # 全体ノルムでの clip_grad_norm_ ではない点に注意。
                # 係数はデバイス上のテンソルのまま掛ける(.item()しない)ので
                # GPU->CPU同期は増えない。
                for g_, n_ in zip(grads, per_param_norms):
                    g_.mul_(torch.clamp(clip_grad / (n_ + 1e-6), max=1.0))
        else:
            grad_norm = 0.0
            if clip_norms is not None:
                clip_norms.append(())
        grad_norms.append(grad_norm)
        # DDPラップ後、DINO等のカスタムメソッド(cancel_last_layer_gradients /
        # update_moving_average)は .module 側にしか存在しないため unwrap 経由で呼ぶ
        # （DistributedDataParallel は任意属性を .module へフォワードしない）。
        # 公式DINOと同じ順序: clip_gradients -> cancel_gradients_last_layer -> step
        # （逆にすると、grad=Noneにしたはずのlast_layerがクリップ処理で復活しうる）。
        raw_model = distributed.unwrap(model)
        if hasattr(raw_model, 'cancel_last_layer_gradients'):
            raw_model.cancel_last_layer_gradients(epoch)  # DINO anti-collapse (early epochs)
        if bt_probe is not None and grads:
            bt_probe.before_step(grad_params)
        optimizer.step()
        if bt_probe is not None and grads:
            bt_probe.after_step(grad_params)
        if hasattr(raw_model, 'update_moving_average'):
            raw_model.update_moving_average()
        # ログ用のlossはrank-local値だと「rank0がたまたま引いたshardのloss」に
        # なってしまうため、マルチGPU時は全rank平均に揃える(no-op when world_size==1)。
        loss_value = distributed.all_reduce_mean(loss.detach().clone()).item()
        train_batch_loss.append(loss_value)
        if stop_on_nonfinite and not math.isfinite(loss_value):
            first_nonfinite = i
            ctx.logger.logger.warning(
                f'Non-finite loss at Epoch {epoch + 1} step {i}/{niter_per_ep} (loss={loss_value}); '
                f'stopping the epoch early')
            break
    if bt_probe is not None:
        bt_probe.close()   # フックを外す(評価・val の forward を測らない、次 epoch で二重登録しない)
    ctx.train_step_summary = {}
    if bt_stats is not None:
        try:
            # epoch 要約(WandB 用): StepProbe の列 + 手法ごとの崩壊統計
            stat_names = BT_STAT_NAMES if ctx.args.ssl_name == 'barlowtwins' else SIMSIAM_STAT_NAMES
            g_rows = bt_probe.rows()
            s_rows = torch.stack(bt_stats).float().cpu().tolist() if bt_stats else []
            ctx.train_step_summary = summarize_steps(
                list(bt_probe.column_names) + list(stat_names),
                [list(a) + (list(s_rows[i]) if i < len(s_rows) else []) for i, a in enumerate(g_rows)],
                grad_norms)
        except Exception as exc:     # 要約は付加情報。失敗しても学習は止めない
            ctx.logger.logger.warning(f'step summary failed: {exc}')
        try:
            write_param_names(ctx.dir_name, bt_probe.names)
            is_bt = ctx.args.ssl_name == 'barlowtwins'
            write_bt_epoch(ctx.dir_name, epoch + 1, train_batch_loss, grad_norms, bt_stats, first_nonfinite,
                           groups=bt_probe.rows(), group_names=bt_probe.column_names,
                           stat_names=BT_STAT_NAMES if is_bt else SIMSIAM_STAT_NAMES,
                           prefix='bt_steps' if is_bt else 'steps')
        except OSError as exc:
            ctx.logger.logger.warning(f'step telemetry could not be saved: {exc}')
    if clip_norms is not None:
        try:
            write_clip_epoch(ctx.dir_name, epoch + 1, clip_grad, clip_norms, grad_norms)
        except OSError as exc:
            ctx.logger.logger.warning(f'Clipping telemetry could not be saved: {exc}')
    # 損失の内訳の epoch 平均(BT の on_diag / off_diag)。戻り値を変えずに ctx 経由で train() に渡す
    ctx.train_loss_parts = parts_meter.result('train', criterion)
    return model, np.mean(train_batch_loss), np.mean(grad_norms)

def atomic_torch_save(obj, path):
    """同じディレクトリの一時ファイルへ保存してから os.replace で置き換える(途中で殺されても元が残る)。"""
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_state_meta(dir_name, completed_epochs, num_epoch):
    """state.json: state.pt に保存済みの epoch 数(1 始まりの完了数)。resume.sh が読む。"""
    import json
    path = os.path.join(dir_name, 'state.json')
    with open(f'{path}.tmp', 'w') as f:
        json.dump({'completed_epochs': int(completed_epochs), 'num_epoch': int(num_epoch),
                   'saved_at': time.strftime('%Y-%m-%dT%H:%M:%S')}, f)
    os.replace(f'{path}.tmp', path)


def restore_healthy_weights(model, early_stopping_path, dir_name, logger):
    """abort(NaN / 発散 / 崩壊 / early stop)時に model_ssl.pt へ書く重みの復元元を選ぶ。

    優先順: checkpoint.pt(最良 val_loss) → state.pt の model_state_dict(最後の健全 epoch の末) → None。
    別ディレクトリへ state.pt だけコピーして再開した診断ラン(0030〜0032)では checkpoint.pt が無く、
    旧実装は NaN 時に RuntimeError、崩壊/early stop 時に FileNotFoundError で落ちていた(レビュー指摘 P2)。
    state.pt は NaN/発散の判定の**後**にしか保存されない(train() の順序)ので、常に健全な重みを持つ。
    戻り値は復元元の名前(None = 復元できる重みが無い)。
    """
    raw = distributed.unwrap(model)
    if os.path.exists(early_stopping_path):
        raw.load_state_dict(torch.load(early_stopping_path, map_location='cpu'))
        logger.info(f'restored weights from {early_stopping_path} (best val_loss)')
        return 'checkpoint.pt'
    state_path = os.path.join(dir_name, 'state.pt')
    if os.path.exists(state_path):
        # state.pt は criterion / early_stopping のオブジェクトも持つので weights_only=False
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        raw.load_state_dict(state['model_state_dict'])
        logger.info(f'restored weights from {state_path} (end of epoch {int(state["epoch"]) + 1}; '
                    f'no checkpoint.pt in this directory)')
        return 'state.pt'
    logger.warning(f'no healthy weights to restore: neither {early_stopping_path} nor {state_path} exists')
    return None


def diagnose_gpu_bound(ctx: RunContext, model, criterion, optimizer, batch_size, size=(224,224), n_steps=50):
    """データパイプラインを完全にバイパスして、GPU計算のみの速度を測る診断用関数"""
    args, ssl_class, DEVICE = ctx.args, ctx.ssl_class, ctx.device
    model.train()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_steps):
        # ssl_class.calc_loss が期待する data の形にダミーテンソルを合わせる
        if args.ssl_name in ("mae",):
            data = [torch.randn(batch_size, 3, *size, device=DEVICE)]
        elif args.ssl_name in ("dino",):
            data = [torch.randn(batch_size, 3, *size, device=DEVICE) for _ in range(ssl_class.n_global_crops)]
        elif args.ssl_name == "swav":
            data = [torch.randn(batch_size, 3, *size, device=DEVICE) for _ in range(2)]
        else:
            # barlowtwins, byol, simsiam, simclr, wsl 等: [x1, x2] のペア
            data = [torch.randn(batch_size, 3, *size, device=DEVICE),
                    torch.randn(batch_size, 3, *size, device=DEVICE)]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ssl_class.calc_loss(model, data, criterion)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - start
    samples_per_sec = n_steps * batch_size / elapsed
    print(f"[diagnose_gpu_bound] {n_steps} steps, {elapsed:.2f}s, {samples_per_sec:.1f} samples/sec (pure GPU compute, no dataloader)")
    return samples_per_sec

@torch.no_grad()
def norm_gain_mean(model):
    """backbone の LayerNorm/BatchNorm ゲイン(ndim==1 の `*.weight`)の平均。

    weight decay がこれらに掛かっていると、対抗する勾配がほとんど無いため
    毎step `γ <- γ(1 - lr*wd)` で単調に削られ、backboneが実質的に消滅する
    (2026-08-31にDINO恒久崩壊の主因として同定。0017はep100でこの値が0.0019
    ＝初期値の0.2%まで削られていた)。初期値1.0から明確に下がり続けていたら
    weight decayの param group 設定を疑うこと (lib/trainer/model.py:_wd_groups)。
    毎epochログに出るので、崩壊してから気付く事態を防げる。
    """
    raw = distributed.unwrap(model)
    base = raw
    for attr in ('student_backbone', 'backbone'):
        if hasattr(raw, attr):
            base = getattr(raw, attr)
            break
    gains = [p.detach().float().mean() for n, p in base.named_parameters()
             if p.ndim == 1 and n.endswith('.weight')]
    if not gains:
        return float('nan')
    return torch.stack(gains).mean().item()


@torch.no_grad()
@contextlib.contextmanager
def batch_stat_bn(model):
    """BatchNorm を batch 統計で動かしつつ running 統計は更新しない(momentum を一時的に 0 にする)。
    train モードにするので、Dropout / DropPath を持つモデルでは使わないこと(BT の ViT backbone は持たない)。"""
    was_training = model.training
    saved = []
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            saved.append((m, m.momentum))
            m.momentum = 0.0          # running = (1-0)·running + 0·batch → 変わらない
    model.train()
    try:
        yield
    finally:
        for m, mom in saved:
            m.momentum = mom
        model.train(was_training)


def sharded_val_loader(loader):
    """val のバッチを rank ごとに別のものにする(学習の DistributedSampler と同じ割り方、shuffle なし)。
    損失が rank 間で c を all_reduce する BT では、これで学習と同じ global batch(256×8=2048)の相関になる。
    単一プロセスではそのまま返す。"""
    if distributed.world_size() <= 1:
        return loader
    sampler = DistributedSampler(loader.dataset, num_replicas=distributed.world_size(), rank=distributed.rank(),
                                 shuffle=False, drop_last=True)
    return torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=sampler,
                                       num_workers=loader.num_workers, pin_memory=True, drop_last=True)


@torch.no_grad()
def compute_val_loss(ctx: RunContext, model, criterion, loader, max_batches=40):
    """Held-out pretext loss on the validation fold (fold_idx, excluded from training).
    Overfitting monitor: compare against train loss. Side-effect-safe for DINO
    (the DINOLoss center buffer is snapshotted/restored so val data doesn't drift it).

    損失がバッチ統計に依存する手法(BT: `criterion.batch_statistics_loss`)は、学習と同じ条件で測る(2026-10-16):
      - BatchNorm は batch 統計(eval モードの running 統計だと projector の出力が冗長になり、損失が D=8192 付近〜
        1 万超で大きく揺れた。0035 の val_loss は全 epoch これで比較にならなかった)。running 統計は更新しない。
      - val のバッチを rank ごとに別にして、相関を学習と同じ global batch(2048)で取る(同じ rank に同じバッチだと
        実効 N が 256 になり、ランダムな相関の下限 λ·D²/N だけで約 1,300 残る)。c は all_reduce されるので損失は全 rank 同一。
    それ以外の手法は従来どおり eval モード・全 rank 同じバッチ(値は rank 間で一致する)。"""
    ssl_class = ctx.ssl_class
    batch_stats = getattr(criterion, 'batch_statistics_loss', False)
    saved_center = criterion.center.clone() if hasattr(criterion, 'center') else None
    losses = []
    parts_meter = LossPartsMeter(criterion)
    if batch_stats:
        loader = sharded_val_loader(loader)
        mode = batch_stat_bn(model)
    else:
        was_training = model.training
        model.eval()
        mode = contextlib.nullcontext()
    with mode:
        for i, data in enumerate(loader):
            parts_meter.before(criterion)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = ssl_class.calc_loss(model, data, criterion)
            parts_meter.add(criterion)
            losses.append(loss.item())
            if i + 1 >= max_batches:
                break
    if saved_center is not None:
        criterion.center.copy_(saved_center)
    if not batch_stats and was_training:
        model.train()
    ctx.val_loss_parts = parts_meter.result('val', criterion)
    return float(np.mean(losses)) if losses else float("nan")

# train
def train(ctx: RunContext, model, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100, run=None, start_epoch:int=0, train_loss=None, collapse_monitor=None):
    """ train ssl model """
    args, DEVICE, DIR_NAME, LOGGER = ctx.args, ctx.device, ctx.dir_name, ctx.logger
    start = time.time()
    evaluator = ssl_eval.SSLEvaluator(samples_n=2048, uniformity_samples=512)
    train_loader, eval_loader = prepare_data(ctx, batch_size=args.batch_size)
    # teacher温度warmupはepoch単位で指定されるが更新はstep単位のため、
    # niter_per_ep が判明したここでstep数へ変換しておく(DINO以外はno-op)。
    if hasattr(criterion, 'set_schedule'):
        criterion.set_schedule(len(train_loader))
    train_loss = list(train_loss) if train_loss is not None else list()
    if args.fix_pred_lr:
        # SimSiam fix-pred-lr: timm の scheduler は構築時に全グループを warmup_lr_init にするので、
        # epoch 末の掛け直し(下)だけでは最初の epoch の predictor が 1e-5 で走る。最初から固定値にする
        # (公式 main_simsiam.py は predictor を常に init_lr)。
        for g in optimizer.param_groups:
            if g.get('fix_lr'):
                g['lr'] = args.lr
    for epoch in range(start_epoch, num_epoch):
        # DistributedSampler(world_size>1のみ設定される。lib/trainer/data.py)は
        # epochごとにset_epochしないと全epochで同じrank分割・同じシャッフルになるため必須。
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model, train_epoch_loss, grad_norm = train_epoch(
            ctx, model, train_loader, criterion, optimizer, epoch, num_epoch=num_epoch)
        # 次の epoch(添字 epoch+1)の lr を入れる。旧実装の step(epoch) は 1 epoch 遅れていた(epoch e が
        # _get_lr(e-1) で走る)。構築時に epoch 0 の値(warmup_lr_init、warmup 無しならピーク)が入っている。
        scheduler.step(epoch + 1)
        # SimSiam: hold the predictor at a constant LR (fix-pred-lr; not decayed).
        if args.fix_pred_lr:
            for g in optimizer.param_groups:
                if g.get('fix_lr'):
                    g['lr'] = args.lr
        # DINO: cosine weight-decay schedule (weight_decay -> weight_decay_end).
        # wd_exempt(bias/ndim<=1)のグループは絶対に上書きしないこと。ここで上書きすると
        # lib/trainer/model.py:_wd_groups の除外が毎epoch無効化され、LayerNormゲインが
        # 削られて恒久崩壊する経路がそのまま復活する。
        if args.weight_decay_end > 0:
            import math
            wd = args.weight_decay_end + 0.5 * (args.weight_decay - args.weight_decay_end) * \
                 (1 + math.cos(math.pi * epoch / max(1, num_epoch)))
            for g in optimizer.param_groups:
                if not g.get('fix_lr') and not g.get('wd_exempt'):
                    g['weight_decay'] = wd
        train_loss.append(train_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        # DINOのteacher momentum/温度は「スケジュールを有効にしたつもりで実は
        # 更新配線が無く固定のままだった」という事故が起きうる(0021の実装は
        # 更新呼び出し側が失われており、実際に適用されていたか事後検証できない)。
        # 毎epochの実測値をログに残して、後から適用状況を確認できるようにする。
        _raw = distributed.unwrap(model)
        teacher_state = ''
        if hasattr(_raw, 'momentum'):
            teacher_state += f', teacher_momentum: {_raw.momentum:.6f}'
        if hasattr(criterion, 'teacher_temp'):
            teacher_state += f', teacher_temp: {criterion.teacher_temp:.4f}'
        # LayerNormゲイン平均: weight decayによるbackbone消滅の早期検知用(毎epoch)
        ln_gain = norm_gain_mean(model)
        # グループごとに lr が違う(BT の bias/LN、SimSiam の predictor)ときは全グループの lr も残す
        group_lrs = sorted({round(g['lr'], 10) for g in optimizer.param_groups})
        lr_groups = f", lr_groups: {[f'{v:.2e}' for v in group_lrs]}" if len(group_lrs) > 1 else ''
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, '
            f'lr: {current_lr:.2e}, grad_norm: {grad_norm:.4f}, '
            f'ln_gain: {ln_gain:.4f}{teacher_state}{lr_groups}'
        )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        train_parts = getattr(ctx, 'train_loss_parts', {}) or {}
        if train_parts:
            LOGGER.logger.info(f'train_loss_parts: {_fmt_parts(train_parts)}')
        step_summary = getattr(ctx, 'train_step_summary', {}) or {}
        if step_summary:
            keys = ('steps/gn_raw_max', 'steps/ln_gain_min_min', 'steps/act_patch_mean', 'steps/act_b0_mean', 'steps/z_erank_mean', 'steps/cos_zz_mean')
            LOGGER.logger.info('step_summary: ' + ', '.join(f'{k[6:]}: {step_summary[k]:.4g}' for k in keys if k in step_summary))
        if run:
            run.log({
                **step_summary,
                **train_parts,
                "train_loss": train_epoch_loss,
                "learning_rate": current_lr,
                "grad_norm": grad_norm,
                "norm_gain_mean": ln_gain,
            }, step=epoch)
        # NaN/inf の train_loss は重みが回復不能なので、patience も rank_monitor_interval も
        # 待たず即停止する。**state.pt 保存・early_stopping() より前**に置くこと:
        #   - EarlyStopping は `NaN > best` が False になり「改善」側へ入るため、
        #     NaN の重みを checkpoint.pt(abort時の復元元)へ保存してしまう。
        #   - state.pt も NaN 重みで上書きされ、--resume しても壊れた状態から再開になる。
        # train_epoch_loss は all_reduce_mean 済みで全rank同一なので、追加の集合通信なしで
        # 全rankが同じ判断をする(片rankだけ抜けると他rankがDDPでhangする)。
        # check_loss_finite / check_loss_rebound は毎epoch・全rankで呼ぶ(内部状態を全rankで揃える)。
        nonfinite = collapse_monitor is not None and collapse_monitor.check_loss_finite(train_epoch_loss)
        rebound = (collapse_monitor is not None and not nonfinite
                   and collapse_monitor.check_loss_rebound(epoch + 1, train_epoch_loss))
        if nonfinite or rebound:
            LOGGER.logger.info(
                f'{"Non-finite train_loss" if nonfinite else "Diverged train_loss"} at Epoch: {epoch + 1} '
                f'— aborting to save compute [{collapse_monitor.reason}]'
            )
            source = restore_healthy_weights(model, early_stopping.path, DIR_NAME, LOGGER.logger)
            if nonfinite and source is None:
                # 健全な重みがどこにも無い(checkpoint.pt も state.pt も無い = 1 epoch も終えていない)。
                # NaN 重みを model_ssl.pt として書き出さないよう、例外で止める。
                raise RuntimeError(
                    f'train_loss became non-finite at Epoch {epoch + 1} and no healthy '
                    f'weights exist ({early_stopping.path} / {DIR_NAME}/state.pt); refusing to write NaN weights'
                )
            return model, train_loss, True
        # 崩壊監視(effective_rank等)はrank0のみで実行する。eval_loaderは全rank同一
        # (data.py: split_by_rank=False)なので結果はどのrankで計算しても同じであり、
        # 全rankで重複計算する意味がない（docs/multi_gpu_migration.md §5）。
        if distributed.is_main_process() and (
            (epoch + 1) % args.rank_monitor_interval == 0 or (epoch + 1) == num_epoch
        ):
            # DDPラップされたままだと SSLEvaluator._backbone_embed の
            # hasattr(model, 'backbone') 判定が通らず(DDPは.moduleの属性を
            # 自動転送しない)、backboneではなく投影後ベクトルで診断指標を
            # 計算してしまう。unwrap()して生モジュールを渡す(単一プロセス
            # 実行ではno-op)。
            ssl_metrics = evaluator.evaluate(distributed.unwrap(model), eval_loader, DEVICE)
            LOGGER.logger.info(
                f'eff_rank: {ssl_metrics.get("effective_rank", float("nan")):.2f}, '
                f'feat_std: {ssl_metrics.get("feature_dim_std", float("nan")):.4f}, '
                f'alignment: {ssl_metrics.get("alignment", float("nan")):.4f}, '
                f'uniformity: {ssl_metrics.get("uniformity", float("nan")):.4f}, '
                f'n_comp_90: {ssl_metrics.get("n_comp_90", float("nan")):.0f}, '
                f'pc1_ratio: {ssl_metrics.get("pc1_ratio", float("nan")):.3f}, '
                f'mean_cos: {ssl_metrics.get("mean_cos", float("nan")):.3f}'
            )
            if run:
                run.log(ssl_metrics, step=epoch)
            if collapse_monitor is not None:
                collapse_monitor.update(ssl_metrics.get("effective_rank"),
                                        uniformity=ssl_metrics.get("uniformity"),
                                        train_loss=train_epoch_loss)
        # held-out validation pretext loss on the val fold (overfitting monitor)
        val_epoch_loss = compute_val_loss(ctx, model, criterion, eval_loader, max_batches=args.val_max_batches)
        LOGGER.logger.info(
            f'val_loss: {val_epoch_loss:.4f}  (train {train_epoch_loss:.4f}, '
            f'train-val gap {train_epoch_loss - val_epoch_loss:+.4f})'
        )
        val_parts = getattr(ctx, 'val_loss_parts', {}) or {}
        if val_parts:
            LOGGER.logger.info(f'val_loss_parts: {_fmt_parts(val_parts)}')
        if run:
            run.log({**val_parts, "val_loss": val_epoch_loss,
                     "train_val_gap": train_epoch_loss - val_epoch_loss}, step=epoch)
        # チェックポイント保存・logger.pkl書き出しはrank0限定
        # （複数プロセスが同じファイルへ競合書き込みするのを防ぐ。
        # docs/multi_gpu_migration.md §1「loop.pyのcheckpoint保存・ログをrank0限定に」）。
        # state_dictは常にunwrap(=DDPの"module."prefixなし)して保存し、単一GPU実行時の
        # チェックポイント形式・下流の表現比較パイプライン(lib/analysis)との互換性を保つ。
        if distributed.is_main_process():
            state = {
                "epoch": epoch,
                "model_state_dict": distributed.unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "criterion": criterion,
                "early_stopping": early_stopping,
                "train_loss": train_loss
            }
            # 一時ファイルへ書いてから os.replace(同一ディレクトリ内なので atomic)。直接上書きすると、保存中に
            # walltime で殺されたとき直前の正常な state.pt も失い、optimizer 込みの再開ができなくなる(レビュー指摘)。
            atomic_torch_save(state, f'{DIR_NAME}/state.pt')
            # resume.sh が「どの epoch まで保存済みか」を 1.7GB の state.pt を読まずに知るための小さな記録。
            # state.pt の**後**に書くので、ここにある epoch は必ず state.pt に入っている。
            write_state_meta(DIR_NAME, completed_epochs=epoch + 1, num_epoch=num_epoch)
            # periodic model snapshots (kept, not overwritten) for trajectory analysis /
            # collapse recovery. Saves model weights only (not optimizer) as model_ep{N}.pt.
            if args.save_interval > 0 and ((epoch + 1) % args.save_interval == 0 or (epoch + 1) == num_epoch):
                atomic_torch_save(distributed.unwrap(model).state_dict(), f'{DIR_NAME}/model_ep{epoch + 1}.pt')
                LOGGER.logger.info(f'saved snapshot model_ep{epoch + 1}.pt')
            LOGGER.save_logger(fileout=ctx.file_log)
        # monitor VAL loss now (fixes the previous train-loss bug). checkpoint.pt = best held-out model.
        # 全rankで呼ぶ（early_stoppingの内部状態(counter/best_score/early_stop)を
        # 全rankで揃えて更新するため。val_epoch_lossはeval_loaderがsplit_by_rank=Falseで
        # 全rank同一なのでrank間で一致し、broadcastなしで判定を一致させられる。
        # ファイル書き込み自体はEarlyStopping.save_enabled(=is_main_process)でrank0に限定）。
        early_stopping(val_epoch_loss, distributed.unwrap(model))
        # default: fixed epoch budget (standard SSL: epoch + cosine). only stop early if explicitly requested.
        if args.early_stop and early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping (val_loss) at Epoch: {epoch}')
            if restore_healthy_weights(model, early_stopping.path, DIR_NAME, LOGGER.logger) is None:
                raise RuntimeError(f'early stop at Epoch {epoch + 1} but no weights to restore')
            return model, train_loss, True
        if collapse_monitor is not None and (
            (epoch + 1) % args.rank_monitor_interval == 0 or (epoch + 1) == num_epoch
        ):
            # effective_rankはrank0だけが計算している(上のrank_monitorブロック)ため、
            # collapse判定をall_reduceで全rankへ揃えてから抜ける。揃えずにrank0だけ
            # breakすると、他rankが次epochのDDP集合通信(backward等)を待ち続けてhangする。
            collapse_flag = torch.zeros(1, device=DEVICE)
            if distributed.is_main_process() and collapse_monitor.collapsed:
                collapse_flag[0] = 1.0
            collapse_flag = distributed.all_reduce_mean(collapse_flag)
            if collapse_flag.item() > 0:
                LOGGER.logger.info(
                    f'Collapse detected for {args.collapse_patience} consecutive checks '
                    f'at Epoch: {epoch} — aborting to save compute'
                    + (f' [{collapse_monitor.reason}]' if collapse_monitor.reason else '')
                )
                # checkpoint.pt(最良 val_loss) か state.pt(最後の健全 epoch)を復元して、
                # 崩壊した重みを model_ssl.pt に書かないようにする。
                if restore_healthy_weights(model, early_stopping.path, DIR_NAME, LOGGER.logger) is None:
                    raise RuntimeError(f'collapse at Epoch {epoch + 1} but no weights to restore')
                return model, train_loss, True
    return model, train_loss, True
