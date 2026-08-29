# -*- coding: utf-8 -*-
"""
# 学習ループ

分散学習に移行する際は、ログ・チェックポイント書き出しを rank0 のみに限定する
ガードをここに入れる（REFACTOR_PLAN.md §6-4）。

"""
import time

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import lib.sslmodel.evaluation as ssl_eval
from lib.trainer import distributed
from lib.trainer.context import RunContext
from lib.trainer.data import prepare_data


# train epoch

def train_epoch(ctx: RunContext, model, data_loader, criterion, optimizer, epoch, num_epoch=None):
    ssl_class = ctx.ssl_class
    model.train()
    train_batch_loss = []
    grad_norms = []
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
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ssl_class.calc_loss(model, data, criterion)
        loss.backward()
        # DDPラップ後、DINO等のカスタムメソッド(cancel_last_layer_gradients /
        # update_moving_average)は .module 側にしか存在しないため unwrap 経由で呼ぶ
        # （DistributedDataParallel は任意属性を .module へフォワードしない）。
        raw_model = distributed.unwrap(model)
        if hasattr(raw_model, 'cancel_last_layer_gradients'):
            raw_model.cancel_last_layer_gradients(epoch)  # DINO anti-collapse (early epochs)
        # total grad L2 norm with a SINGLE device sync (old code did .item() per
        # parameter -> ~150 GPU->CPU syncs/iter, serializing the step). Identical value.
        grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack(torch._foreach_norm(grads))).item() if grads else 0.0
        grad_norms.append(grad_norm)
        optimizer.step()
        if hasattr(raw_model, 'update_moving_average'):
            raw_model.update_moving_average()
        # ログ用のlossはrank-local値だと「rank0がたまたま引いたshardのloss」に
        # なってしまうため、マルチGPU時は全rank平均に揃える(no-op when world_size==1)。
        loss_value = distributed.all_reduce_mean(loss.detach().clone()).item()
        train_batch_loss.append(loss_value)
    return model, np.mean(train_batch_loss), np.mean(grad_norms)

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
def compute_val_loss(ctx: RunContext, model, criterion, loader, max_batches=40):
    """Held-out pretext loss on the validation fold (fold_idx, excluded from training).
    Overfitting monitor: compare against train loss. Side-effect-safe for DINO
    (the DINOLoss center buffer is snapshotted/restored so val data doesn't drift it)."""
    ssl_class = ctx.ssl_class
    was_training = model.training
    model.eval()
    saved_center = criterion.center.clone() if hasattr(criterion, 'center') else None
    losses = []
    for i, data in enumerate(loader):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = ssl_class.calc_loss(model, data, criterion)
        losses.append(loss.item())
        if i + 1 >= max_batches:
            break
    if saved_center is not None:
        criterion.center.copy_(saved_center)
    if was_training:
        model.train()
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
    for epoch in range(start_epoch, num_epoch):
        # DistributedSampler(world_size>1のみ設定される。lib/trainer/data.py)は
        # epochごとにset_epochしないと全epochで同じrank分割・同じシャッフルになるため必須。
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model, train_epoch_loss, grad_norm = train_epoch(
            ctx, model, train_loader, criterion, optimizer, epoch, num_epoch=num_epoch)
        scheduler.step(epoch)
        # SimSiam: hold the predictor at a constant LR (fix-pred-lr; not decayed).
        if args.fix_pred_lr:
            for g in optimizer.param_groups:
                if g.get('fix_lr'):
                    g['lr'] = args.lr
        # DINO: cosine weight-decay schedule (weight_decay -> weight_decay_end).
        if args.weight_decay_end > 0:
            import math
            wd = args.weight_decay_end + 0.5 * (args.weight_decay - args.weight_decay_end) * \
                 (1 + math.cos(math.pi * epoch / max(1, num_epoch)))
            for g in optimizer.param_groups:
                if not g.get('fix_lr'):
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
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, '
            f'lr: {current_lr:.2e}, grad_norm: {grad_norm:.4f}{teacher_state}'
        )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        if run:
            run.log({
                "train_loss": train_epoch_loss,
                "learning_rate": current_lr,
                "grad_norm": grad_norm,
            }, step=epoch)
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
                f'uniformity: {ssl_metrics.get("uniformity", float("nan")):.4f}'
            )
            if run:
                run.log(ssl_metrics, step=epoch)
            if collapse_monitor is not None:
                collapse_monitor.update(ssl_metrics.get("effective_rank"))
        # held-out validation pretext loss on the val fold (overfitting monitor)
        val_epoch_loss = compute_val_loss(ctx, model, criterion, eval_loader, max_batches=args.val_max_batches)
        LOGGER.logger.info(
            f'val_loss: {val_epoch_loss:.4f}  (train {train_epoch_loss:.4f}, '
            f'train-val gap {train_epoch_loss - val_epoch_loss:+.4f})'
        )
        if run:
            run.log({"val_loss": val_epoch_loss,
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
            torch.save(state, f'{DIR_NAME}/state.pt')
            # periodic model snapshots (kept, not overwritten) for trajectory analysis /
            # collapse recovery. Saves model weights only (not optimizer) as model_ep{N}.pt.
            if args.save_interval > 0 and ((epoch + 1) % args.save_interval == 0 or (epoch + 1) == num_epoch):
                torch.save(distributed.unwrap(model).state_dict(), f'{DIR_NAME}/model_ep{epoch + 1}.pt')
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
            distributed.unwrap(model).load_state_dict(torch.load(early_stopping.path))
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
                    f'Collapse detected (effective_rank < {args.collapse_rank_threshold} for '
                    f'{args.collapse_patience} consecutive checks) at Epoch: {epoch} — aborting to save compute'
                )
                # early_stopping.path (best val_loss checkpoint.pt) still holds the last
                # pre-collapse snapshot; restore it so model_ssl.pt isn't the collapsed weights.
                distributed.unwrap(model).load_state_dict(torch.load(early_stopping.path))
                return model, train_loss, True
    return model, train_loss, True
