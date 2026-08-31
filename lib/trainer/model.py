# -*- coding: utf-8 -*-
"""
# モデル・損失・スケジューラの構築

DDP ラップ自体は lib/trainer/entry.py の `distributed.wrap(model)` で行う。
ここでは EarlyStopping の書き込みをrank0限定にする（`save_enabled`）等、
分散実行時にもモデル構築ロジック自体は単一プロセス実行と同じにする。

"""
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler

import lib.sslmodel as sslmodel
import lib.model.zoo as zoo
from lib.trainer import distributed
from lib.trainer.context import RunContext
from lib.trainer.optim import build_optimizer


def _wd_groups(args, groups):
    """各param groupを「weight decayを掛ける側」と「掛けない側(bias・ndim<=1)」に割る。

    bias / LayerNorm・BatchNormのゲインに weight decay を掛けてはいけない
    （DINO `utils.get_params_groups()`, MAE `param_groups_lrd()`, Barlow Twins/SwAV
    のLARS除外フラグ、いずれの公式実装も 1次元パラメータと bias を wd=0 の別グループ
    にしている）。これらのパラメータは wd に対抗する勾配をほとんど持たないため、
    掛けると毎step `γ <- γ(1 - lr*wd)` で単調に削られ、backboneのLayerNormゲインが
    指数的にゼロへ向かう。

    2026-08-31の解析: この実装漏れがDINOの恒久崩壊(loss = ln(8192) = 9.0109 固定)の
    主因だった。0017(num_epoch=100, wd 0.04->0.4)では ep100 で最終LayerNormのゲイン
    平均が 0.0019 (初期値の0.2%)、blocks.11.norm2 は 0.0004 まで削られており、
    backboneが入力に依存しない定数を出す -> DINOLossのcenterがその定数へ収束 ->
    teacher softmaxが厳密に一様 -> 勾配が厳密に0、という吸収状態に落ちていた。
    観測されたゲインの減衰率は純粋な weight decay の予測 prod(1 - lr*wd) と数%以内で
    一致しており、勾配ではなくwdが主因であることを定量的に確認済み。

    `wd_exempt: True` を立てたグループは lib/trainer/loop.py のwdコサイン
    スケジュール(`--weight_decay_end`)でも上書きされない。

    --wd_apply_to_bias_norm を付けると従来通り全パラメータへwdを掛ける
    (SimSiam原論文のResNetレシピはbias/BNを除外しないため、その再現用)。
    """
    if getattr(args, 'wd_apply_to_bias_norm', False):
        return groups
    out = []
    for g in groups:
        decay, no_decay = [], []
        for p in g['params']:
            (no_decay if p.ndim <= 1 else decay).append(p)
        if decay:
            out.append({**g, 'params': decay})
        if no_decay:
            out.append({**g, 'params': no_decay, 'weight_decay': 0.0, 'wd_exempt': True,
                        'weight_decay_filter': True})
    return out


# prepare model
def prepare_model(
        ctx: RunContext,
        model_name:str='ResNet18', patience:int=7, delta:float=0, lr:float=0.003, weight_decay:float=0., num_epoch:int=150, freeze_backbone:bool=False,
        layer_wise_lr:bool=False, backbone_lr_ratio:float=0.1
    ):
    """
    preparation of models
    Parameters
    ----------
        ctx (RunContext)
            args / ssl_class / dir_name を保持する実行コンテキスト

        model_name (str)
            model architecture name

        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    args, ssl_class = ctx.args, ctx.ssl_class
    # model building with indicated name
    # SSL手法比較の前提は「各手法を同一条件(=同じランダム初期化)で事前学習する」こと
    # (Goal.yaml: 「各手法を同一条件で事前学習し...」)。ImageNet事前学習済み重みを
    # 一部の手法・backboneにだけ読み込むと、その手法だけ有利なwarm startになり
    # 比較が成立しない。mae/dinoは元々スクラッチだったが、byol/simclr/wsl や
    # barlowtwins/simsiam/swavを非ViT backboneで使うと weights="DEFAULT" 経由で
    # 事前学習済み重みが混入していたため、常にスクラッチに統一する
    # (2026-08-11実機確認)。
    try:
        encoder = zoo.DICT_MODEL[model_name][0](weights=None)
        size=zoo.DICT_MODEL[model_name][1]
    except:
        print("indicated model name is not implemented")
        ValueError
    if "ViT" in model_name:
        # UNIFIED backbone: BT/SimSiam/SwAV use the SAME timm ViT-B/16 as DINO
        # (dynamic_img_size for pos-embed interpolation on SwAV's 96px local crops).
        # This makes the SSL-method comparison backbone-invariant, and gains
        # SDPA/flash-attention (torchvision ViT uses plain MHA -> no flash, slower).
        # MAE keeps its own (architecturally identical) masking encoder.
        if args.ssl_name in ("barlowtwins", "simsiam", "swav"):
            import timm
            backbone = timm.create_model("vit_base_patch16_224", num_classes=0, dynamic_img_size=True)
            print(f"{args.ssl_name}: unified timm ViT-B/16 (dynamic_img_size) backbone [same as DINO]")
        else:
            encoder.heads = nn.Identity() # ViTはheadがあるため、Identityに置き換える
            backbone = encoder
    elif model_name=="DenseNet121":
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
            )
    else:
        backbone = nn.Sequential(
            *list(encoder.children())[:-1],
            )
    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
    # method-specific paper-faithful model knobs
    model_kwargs = {}
    if args.ssl_name == "swav":
        model_kwargs["n_prototypes"] = args.n_prototypes
    if args.ssl_name == "barlowtwins" and args.proj_dim > 0:
        model_kwargs["projection_dim"] = args.proj_dim
        model_kwargs["pred_dim"] = args.proj_dim
    if args.ssl_name == "dino":
        # None のまま渡せば prepare_model 側で固定運用(0017と同一)になる。
        if args.dino_momentum_start is not None:
            model_kwargs["momentum"] = args.dino_momentum_start
        model_kwargs["momentum_end"] = args.dino_momentum_end
        model_kwargs["teacher_temp_end"] = args.dino_teacher_temp_end
        model_kwargs["teacher_temp_warmup_epochs"] = args.dino_teacher_temp_warmup_epochs
        model_kwargs["out_dim"] = args.dino_out_dim
        model_kwargs["drop_path_rate"] = args.dino_drop_path
    model, criterion = ssl_class.prepare_model(backbone, head_size=size, **model_kwargs)
    if args.model_path:
        # warm start: load weights only (student+teacher, via model.state_dict()) from a
        # finished run's model_ssl.pt. Unlike --resume, optimizer/scheduler/epoch counter
        # are NOT restored, so this run gets a fresh schedule (e.g. continuing 0017 past
        # its original epoch budget with a new cosine cycle instead of a flat lr_min tail).
        model.load_state_dict(torch.load(args.model_path, map_location=ctx.device))
        if args.resume:
            # entry.py's resume path (state.pt found) loads its own model_state_dict right
            # after prepare_model() returns, silently discarding these warm-started weights.
            print(f"warm-started model weights from {args.model_path}, but --resume is also set: "
                  f"if {ctx.dir_name}/state.pt exists, its weights will override this warm start")
        else:
            print(f"warm-started model weights from {args.model_path}")
    if args.optimizer == "lars" and args.lars_exclude_bias_bn:
        # Barlow Twins / SwAV: weights (LARS-adapted, weight-decayed) vs bias & BN
        # (ndim<=1: excluded from LARS adaptation and weight decay, own LR).
        weights, biases = [], []
        for p in model.parameters():
            if p.requires_grad:
                (biases if p.ndim <= 1 else weights).append(p)
        lr_bias = args.lr_bias if args.lr_bias > 0 else lr
        optimizer = build_optimizer(ctx, [
            {'params': weights},
            {'params': biases, 'lr': lr_bias, 'weight_decay': 0.0, 'wd_exempt': True,
             'weight_decay_filter': True, 'lars_adaptation_filter': True},
        ], lr=lr, weight_decay=weight_decay)
        print(f"LARS param-groups: weights(lr={lr:.3g},wd={weight_decay:.1e}) | "
              f"bias/BN(lr={lr_bias:.3g},no-LARS,no-wd)")
    elif args.fix_pred_lr and hasattr(model, "predictor"):
        # SimSiam: predictor kept at a constant LR (not decayed by the scheduler).
        pred_ids = set(id(p) for p in model.predictor.parameters())
        pred_params, base_params = [], []
        for p in model.parameters():
            if p.requires_grad:
                (pred_params if id(p) in pred_ids else base_params).append(p)
        optimizer = build_optimizer(ctx, _wd_groups(args, [
            {'params': base_params},
            {'params': pred_params, 'fix_lr': True},
        ]), lr=lr, weight_decay=weight_decay)
        print(f"SimSiam fix-pred-lr: predictor held at constant lr={lr:.3g}")
    elif layer_wise_lr:
        backbone_param_ids = set(id(p) for p in model.backbone.parameters())
        backbone_params = []
        head_params = []
        for p in model.parameters():
            if p.requires_grad:
                if id(p) in backbone_param_ids:
                    backbone_params.append(p)
                else:
                    head_params.append(p)
        optimizer = build_optimizer(ctx, _wd_groups(args, [
            {'params': backbone_params, 'lr': lr * backbone_lr_ratio},
            {'params': head_params, 'lr': lr}
        ]), lr=lr, weight_decay=weight_decay)
        print(f"Using layer-wise learning rate. Backbone LR: {lr * backbone_lr_ratio:.2e}, Head LR: {lr:.2e}")
    else:
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(ctx, _wd_groups(args, [{'params': parameters}]),
                                    lr=lr, weight_decay=weight_decay)
    if weight_decay > 0:
        n_exempt = sum(len(g['params']) for g in optimizer.param_groups if g.get('wd_exempt'))
        n_decayed = sum(len(g['params']) for g in optimizer.param_groups if not g.get('wd_exempt'))
        note = " [EXEMPTION OFF: --wd_apply_to_bias_norm]" if n_exempt == 0 else ""
        print(f"weight decay {weight_decay:.3g}{'->' + format(args.weight_decay_end, '.3g') if args.weight_decay_end > 0 else ''}: "
              f"decayed {n_decayed} tensors / exempt (bias & ndim<=1) {n_exempt} tensors{note}")

    scheduler = CosineLRScheduler(
        optimizer, t_initial=num_epoch, lr_min=args.lr_min,
        warmup_t=args.warmup_t, warmup_lr_init=args.warmup_lr_init, warmup_prefix=True)
    early_stopping = sslmodel.utils.EarlyStopping(
        patience=patience, delta=delta, path=f'{ctx.dir_name}/checkpoint.pt',
        save_enabled=distributed.is_main_process(),
    )
    collapse_monitor = None
    if args.collapse_early_stop:
        # out_dim を渡すと「train_loss が ln(out_dim) に張り付いた＝一様崩壊」を
        # 判定できる(DINO/SwAV等のprototype系)。持たない手法では None のままで
        # uniformity と effective_rank だけで判定する。
        collapse_monitor = sslmodel.utils.CollapseMonitor(
            rank_threshold=args.collapse_rank_threshold, patience=args.collapse_patience,
            out_dim=getattr(ssl_class, 'out_dim', 0),
        )
    return model, criterion, optimizer, scheduler, early_stopping, collapse_monitor
