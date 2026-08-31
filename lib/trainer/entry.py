# -*- coding: utf-8 -*-
"""
# TG-GATE SSL 学習の CLI 定義と実行

`scripts/train/train_tggate.py` が持っていた引数定義と実行フローをここへ移した
（REFACTOR_PLAN.md §5-0「scripts/ はインフラ専用、研究コードは lib/」）。

移した理由は2つ:

1. `scripts/train/train_tggate.py` は既存27実験の `run_slurm.sh` が直接叩くパスなので
   消せない。実体を lib/ に置き、あちらは薄いシムとして残すことで
   「パス・引数の互換維持」と「研究コードは lib/」を両立させる。
2. 新方式の `experiments/*/experiment.py`（templates/experiment.py 由来）からも
   同じパーサと同じ実行関数を使いたい。パーサを共有しないと、config.yml 側と
   CLI 側で既定値が二重管理になる。

@author: Katsuhisa MORITA
"""
import os
import argparse

import torch
import wandb

import lib.sslmodel as sslmodel
from lib.trainer import distributed
from lib.trainer.context import RunContext, build_context
from lib.trainer.model import prepare_model
from lib.trainer.loop import train, diagnose_gpu_bound


def build_parser() -> argparse.ArgumentParser:
    """学習の CLI パーサを組み立てる。

    引数の名前・既定値は移行前の `scripts/train/train_tggate.py` と完全に同じ。
    既存27実験の `run_slurm.sh` がここに依存しているため、削除・改名はしないこと。
    """
    parser = argparse.ArgumentParser(description='CLI learning')
    # base settings
    parser.add_argument('--note', type=str, help='barlowtwins running')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_path', type=str)
    # data settings
    parser.add_argument('--model_path', type=str, help='dir_model')
    parser.add_argument('--dir_result', type=str, help='result')
    parser.add_argument('--mouse_dataset', action='store_true')
    parser.add_argument('--rat_dataset', action='store_true')
    parser.add_argument('--eisai_dataset', action='store_true')
    parser.add_argument('--shionogi_dataset', action='store_true')
    # model/learning settings
    parser.add_argument('--model_name', type=str, default='ResNet18') # model architecture name
    parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
    parser.add_argument('--num_epoch', type=int, default=50) # epoch
    parser.add_argument('--batch_size', type=int, default=64) # batch size
    parser.add_argument('--lr', type=float, default=0.01) # learning rate
    parser.add_argument('--weight_decay', type=float, default=0.) # weight decay (decoupled for adamw)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd', 'lars']) # AdamW is the standard for ViT SSL; LARS for Barlow Twins/SwAV
    parser.add_argument('--beta2', type=float, default=0.999) # AdamW beta2 (MAE uses 0.95)
    parser.add_argument('--resume', action='store_true') # resume from {DIR_NAME}/state.pt (continue past wall-time)
    parser.add_argument('--early_stop', action='store_true') # if set, stop early on VAL loss; default off = fixed epoch budget (standard SSL)
    parser.add_argument('--val_max_batches', type=int, default=40) # held-out val fold batches for the pretext val loss
    parser.add_argument('--patience', type=int, default=7) # early stopping
    parser.add_argument('--delta', type=float, default=0) # early stopping
    # collapse-triggered abort: opt-in, for diagnostic/isolation runs only (Goal.yaml treats
    # collapse monitoring as a health check, not a stop criterion, for the main paper_* comparisons)
    parser.add_argument('--collapse_early_stop', action='store_true') # if set, abort when effective_rank stays collapsed
    parser.add_argument('--collapse_rank_threshold', type=float, default=5.0) # effective_rank below this counts as collapsed
    parser.add_argument('--collapse_patience', type=int, default=2) # consecutive rank_monitor_interval checks below threshold before aborting
    parser.add_argument('--freeze_backbone', action='store_true') # whether to freeze backbone during training
    # Transform (augmentation) settings
    parser.add_argument('--color_plob', type=float, default=0.8)
    parser.add_argument('--blur_plob', type=float, default=0.4)
    parser.add_argument('--solar_plob', type=float, default=0.)
    # scheduler
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--warmup_t', type=int, default=15)
    parser.add_argument('--warmup_lr_init', type=float, default=1e-5)
    # layer-wise learning rate
    parser.add_argument('--layer_wise_lr', action='store_true')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.1)
    # rank monitor
    parser.add_argument('--rank_monitor_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=0)  # save a kept model_ep{N}.pt snapshot every N epochs (0=off)
    # --- paper-faithful knobs (per-method) ---
    parser.add_argument('--lr_bias', type=float, default=0.0)          # Barlow Twins: separate LR for biases/BN params (0 = same as --lr)
    parser.add_argument('--lars_exclude_bias_bn', action='store_true') # LARS: exclude bias/BN (ndim<=1) from adaptation + weight decay
    # 既定では bias / LayerNorm・BNのゲイン(ndim<=1) を weight decay から除外する
    # (DINO/MAE/BT/SwAV いずれの公式実装もそうしている)。このフラグを付けると
    # 全パラメータへwdを掛ける従来挙動に戻る (SimSiam原論文のResNetレシピ再現用)。
    # 除外しないとLayerNormゲインが指数的に0へ削られ恒久崩壊する (lib/trainer/model.py:_wd_groups)。
    parser.add_argument('--wd_apply_to_bias_norm', action='store_true')
    parser.add_argument('--weight_decay_end', type=float, default=0.0) # DINO: cosine wd schedule end (0 = fixed weight_decay)
    parser.add_argument('--clip_grad', type=float, default=0.0)        # DINO公式: 3.0 (パラメータ毎のL2ノルムでクリップ)。0 = 無効
    # DINO teacher schedules. 既定(None)は 0017 と同じ固定値運用
    # (momentum 0.9995 / teacher_temp 0.04、意図的なanti-collapse設定)。
    # 指定すると論文(Caron et al. 2021)のスケジュールが有効になる。
    # 片方だけ有効にできるので、0021の恒久崩壊がどちらに起因するかを切り分けられる。
    parser.add_argument('--dino_momentum_start', type=float, default=None)        # 論文: 0.996。未指定なら 0.9995 (0017のanti-collapse値)
    parser.add_argument('--dino_momentum_end', type=float, default=None)          # 論文: 1.0 (cosine start->1.0)。未指定なら momentum 固定
    parser.add_argument('--dino_teacher_temp_end', type=float, default=None)      # 論文: 0.07 (linear warmup 0.04->0.07)。未指定なら teacher_temp 固定
    parser.add_argument('--dino_teacher_temp_warmup_epochs', type=int, default=30) # 論文: 30 epoch
    # 公式(main_dino.py, --arch vit_base)の既定は out_dim 65536 / drop_path_rate 0.1。
    # 0024以前は 8192 / 0.0 で走っていたため、既定は据え置きにして明示指定で論文値にする。
    parser.add_argument('--dino_out_dim', type=int, default=8192)      # 公式: 65536 (プロトタイプ数。崩壊時の loss = ln(out_dim))
    parser.add_argument('--dino_drop_path', type=float, default=0.0)   # 公式: 0.1 (stochastic depth。studentのみに適用)
    parser.add_argument('--n_prototypes', type=int, default=512)       # SwAV: number of prototypes (paper: 3000)
    parser.add_argument('--n_global_crops', type=int, default=2)       # multicrop global views
    parser.add_argument('--n_local_crops', type=int, default=0)        # multicrop local views (SwAV 6 / DINO 8); 0 = disabled
    parser.add_argument('--local_crop_size', type=int, default=96)     # multicrop local crop resolution
    parser.add_argument('--proj_dim', type=int, default=0)             # Barlow Twins: projector width (paper: 8192); 0 = class default
    parser.add_argument('--bt_lambda', type=float, default=0.0)        # Barlow Twins: off-diagonal loss weight (paper: 5e-3); 0 = class default
    parser.add_argument('--fix_pred_lr', action='store_true')          # SimSiam: keep predictor LR constant (not decayed)
    parser.add_argument('--diagnose_only', action='store_true')        # run the GPU-bound throughput probe and exit without training
    # --- multi-GPU / DDP knobs (docs/multi_gpu_migration.md, Goal.yaml 2026-08-03) ---
    parser.add_argument('--ddp_linear_scale_lr', action='store_true')  # multiply --lr by world_size (linear scaling rule). Default off: batch/LR redesign per method should be a deliberate per-experiment choice, not automatic.
    return parser


def run_from_args(args, filename: str = 'train_tggate') -> None:
    """引数から RunContext を作って学習を回す。

    新方式の `experiments/*/experiment.py` はこれを呼ぶ。
    """
    main(build_context(args, filename=filename))


def main(ctx: RunContext):
    """エントリポイント。分散のセットアップ/破棄で実処理を挟む。

    `RANK`/`WORLD_SIZE` 等が未設定の単一プロセス実行では distributed.* は
    すべて no-op（従来通りのシングルGPU実行）。`torchrun` 等でこれらの環境変数が
    設定された場合のみ、実際に init_process_group / DDP が有効になる
    （lib/trainer/distributed.py、docs/multi_gpu_migration.md 参照）。
    """
    distributed.setup()
    try:
        _run(ctx)
    finally:
        distributed.teardown()


def _run(ctx: RunContext):
    args, DEVICE, DIR_NAME, LOGGER = ctx.args, ctx.device, ctx.dir_name, ctx.logger
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True   # faster fp32 matmuls (norms/loss); bf16 autocast unaffected
    torch.backends.cudnn.allow_tf32 = True
    # rank0ガード: 複数プロセスから wandb.init すると重複runが作られるため
    # rank0のみ初期化する（Goal.yaml 2026-08-03 / docs/multi_gpu_migration.md §1）。
    # 単一プロセス実行では distributed.is_main_process() は常に True で従来と同じ。
    run = None
    if distributed.is_main_process():
        run = wandb.init(project="tox-patho-tggate",
                         entity="benzelongji-the-university-of-tokyo",
                         name=f"{args.note}_{args.ssl_name}_{args.model_name}_lr{args.lr}_epoch{args.num_epoch}_patience{args.patience}_delta{args.delta}_freeze{args.freeze_backbone}",
                         config=args.__dict__)
    # 1. Self-Supervised Learning
    model, criterion, optimizer, scheduler, early_stopping, collapse_monitor = prepare_model(
        ctx,
        model_name=args.model_name, patience=args.patience, delta=args.delta, lr=args.lr, weight_decay=args.weight_decay, num_epoch=args.num_epoch, freeze_backbone=args.freeze_backbone,
        layer_wise_lr=args.layer_wise_lr, backbone_lr_ratio=args.backbone_lr_ratio
    )
    model = distributed.wrap(model)  # 単一プロセス実行ではno-op。world_size>1ならSyncBN化+DDPラップ
    if args.diagnose_only:
        # スループット診断のみ実行して終了（学習には進まない）
        diagnose_gpu_bound(ctx, model, criterion, optimizer, batch_size=args.batch_size)
        LOGGER.logger.info('--diagnose_only set; skipping training.')
        return
    # resume from a previous run's state.pt (e.g. after hitting the wall-time limit)
    start_epoch, train_loss_init = 0, None
    if args.resume and os.path.exists(f'{DIR_NAME}/model_ssl.pt'):
        LOGGER.logger.info(f'{DIR_NAME}/model_ssl.pt already exists; training complete, nothing to resume.')
        return
    state_path = f'{DIR_NAME}/state.pt'
    if args.resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=DEVICE, weights_only=False)  # state.pt holds criterion/early_stopping objects
        # state.pt は常に unwrap 済み(非DDP)のstate_dictで保存される(loop.py)ため、
        # DDPラップ後のmodelへ読む場合も .module 側にロードする必要がある。
        distributed.unwrap(model).load_state_dict(state['model_state_dict'])
        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        except ValueError as e:
            # 2026-08-31に weight decay の param group を分割した(bias/ndim<=1 を wd=0 の
            # 別グループへ。lib/trainer/model.py:_wd_groups)。それ以前の state.pt は
            # グループ数が違うため復元できない。黙って続けると optimizer 状態が
            # 失われたまま学習が進むので、原因を明示して止める。
            raise RuntimeError(
                f'{state_path} は weight-decay param group 分割より前に保存されたもので、'
                f'現在のoptimizerと構成が一致しないため復元できません ({e})。'
                f'旧ランの続きは不可です。--resume を外して学習し直してください'
                f'（旧設定のまま続けたい場合のみ --wd_apply_to_bias_norm を付けると'
                f'グループ構成が一致しますが、崩壊の原因を残したまま走ることになります）。'
            ) from e
        try:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        except Exception as e:
            LOGGER.logger.warning(f'scheduler state not restored: {e}')
        if state.get('criterion') is not None:
            criterion = state['criterion']
            if hasattr(criterion, 'to'):
                criterion.to(DEVICE)
        if state.get('early_stopping') is not None:
            early_stopping = state['early_stopping']
        train_loss_init = state.get('train_loss')
        start_epoch = int(state['epoch']) + 1
        LOGGER.logger.info(f'Resumed from state.pt (epoch {state["epoch"]}) -> continue at epoch {start_epoch}/{args.num_epoch}')
    elif args.resume:
        LOGGER.logger.info('--resume set but no state.pt found; starting fresh')
    model, train_loss, flag_finish = train(
        ctx, model, criterion, optimizer, scheduler, early_stopping, num_epoch=args.num_epoch, run=run,
        start_epoch=start_epoch, train_loss=train_loss_init, collapse_monitor=collapse_monitor
    )
    distributed.barrier()  # 全rankが学習ループを終えてから rank0 の書き出しへ進む
    if flag_finish:
        # 最終成果物の書き出しはrank0限定（複数プロセスが同じファイルへ競合書き込みするのを防ぐ）。
        if distributed.is_main_process():
            sslmodel.plot.plot_progress_train(train_loss, DIR_NAME)
            sslmodel.utils.summarize_model(
                distributed.unwrap(model),
                None,
                DIR_NAME, lst_name=['summary_ssl.txt', 'model_ssl.pt']
            )
            # 2. save results & config
            LOGGER.to_logger(name='argument', obj=args)
            LOGGER.to_logger(name='loss', obj=criterion)
            LOGGER.to_logger(
                name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
            )
            LOGGER.to_logger(name='scheduler', obj=scheduler)
    else:
        LOGGER.logger.info('reached max epoch / train')
    distributed.barrier()  # rank0の書き出し完了を他rankが待ってからteardown()へ進む
