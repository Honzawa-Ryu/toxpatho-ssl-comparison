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


def build_scheduler(optimizer, num_epoch, warmup_t, lr_min, warmup_lr_init):
    """epoch 単位の warmup + cosine。loop.py は epoch e の末に step(e + 1) を呼び、epoch e は _get_lr(e) で走る。
    warmup を含めて num_epoch で減衰が終わる(tests/test_scheduler_and_save.py)。"""
    return CosineLRScheduler(
        optimizer, t_initial=max(1, num_epoch - warmup_t), lr_min=lr_min,
        warmup_t=warmup_t, warmup_lr_init=warmup_lr_init, warmup_prefix=True)


def configure_bt_criterion(criterion, args):
    """BT の損失の CLI 設定を criterion に入れる(構築時と、state.pt から復元した後の両方で呼ぶ)。

    - --bt_lambda: 0 ならクラス既定(5e-3 = 論文値)。⚠️ 2026-10-13 まではこの引数がどこにも配線されて
      おらず、常にクラス既定の 5e-3 だった(全 BT 実験が 5e-3 を指定していたので結果は同じ)。
    - --bt_grad_ref_world: 勾配を world_size / ref 倍(barlowtwins._GradScale)。0 なら 1 倍(従来どおり)。
    """
    if getattr(args, 'bt_lambda', 0.0) > 0:
        criterion.lambda_param = args.bt_lambda
    ref = getattr(args, 'bt_grad_ref_world', 0)
    criterion.grad_scale = distributed.world_size() / ref if ref > 0 else 1.0
    print(f"BT criterion: lambda={criterion.lambda_param}, grad_scale={criterion.grad_scale:.4g} "
          f"(world_size={distributed.world_size()}, bt_grad_ref_world={ref or 'off'})")


def _point_triton_to_libcuda():
    """triton に GPU ドライバの libcuda.so の場所を教える(コンパイル時のみ)。

    apptainer --nv はホストのドライバを /.singularity.d/libs に bind して LD_LIBRARY_PATH に載せるが、triton は
    `ldconfig -p` の結果を優先し、コンテナの ld キャッシュにある /usr/local/cuda/compat/lib/libcuda.so.1(実体が無い)
    を見て「libcuda.so cannot found」で落ちた(exp 0042、2026-10-13)。実体のある libcuda.so.1 を探し、リンク用の
    libcuda.so の symlink を置いたディレクトリを TRITON_LIBCUDA_PATH に設定する。
    """
    import os
    import tempfile
    if os.environ.get('TRITON_LIBCUDA_PATH'):
        return
    cands = [d for d in os.environ.get('LD_LIBRARY_PATH', '').split(':') if d]
    cands += ['/.singularity.d/libs', '/usr/lib64', '/usr/lib/aarch64-linux-gnu', '/usr/lib/x86_64-linux-gnu',
              '/usr/local/cuda/compat/lib']
    real = next((os.path.join(d, 'libcuda.so.1') for d in cands if os.path.isfile(os.path.join(d, 'libcuda.so.1'))), None)
    if real is None:
        print('compile_blocks: libcuda.so.1 が見つからない(GPU の無い環境)。triton の既定の探索に任せる')
        return
    link_dir = os.path.join(tempfile.gettempdir(), f'triton_libcuda_{os.getuid()}')
    os.makedirs(link_dir, exist_ok=True)
    for name in ('libcuda.so', 'libcuda.so.1'):
        dst = os.path.join(link_dir, name)
        if os.path.islink(dst) and os.readlink(dst) != real:
            os.remove(dst)
        if not os.path.lexists(dst):
            try:
                os.symlink(real, dst)
            except FileExistsError:     # 同じノードの別プロセスが同時に作った
                pass
    os.environ['TRITON_LIBCUDA_PATH'] = link_dir
    print(f'compile_blocks: TRITON_LIBCUDA_PATH={link_dir} -> {real}')


def compile_blocks(model):
    """backbone の各 Transformer block(`blocks.N`、BT/SimSiam は `backbone.blocks.N`、DINO は `student_backbone.blocks.N`)の
    forward を torch.compile に置き換える。戻り値は置き換えた block 数。

    モジュール全体ではなく「block の forward だけ」をコンパイルする理由:
    - state_dict のキーが変わらない(torch.compile(model) は `_orig_mod.` を挟み、既存のチェックポイントと互換が無くなる)。
    - モジュールの forward フック(step テレメトリの block 出力の測定、bt_telemetry.StepProbe)はコンパイル領域の外で
      eager に動くので、epoch ごとにフックを付け外ししても再コンパイルされない。
    - DDP はモジュールの外側で勾配を集約するので、内側の forward をコンパイルしても干渉しない。
    dynamic=False: 形状(MAE のエンコーダは 50 トークン、評価時は 197 トークン等)ごとに別のグラフを作る(数種類)。
    """
    import os
    import re
    import shutil
    # ログインノードの module 環境から CC/CXX(NVIDIA HPC SDK の nvc / nvc++)が入ってくる。/work が bind されているので
    # コンテナ内からもパスは見えるが動かず、inductor が「C++ コンパイラが無い」で落ちる(2026-10-13 に実際に踏んだ)。
    # コンパイルにはコンテナの gcc / g++ を使う(triton の launcher も CC を見る)。inductor の設定は import 時に
    # CXX を読み込み済みなので、設定値も直接書き換える。
    os.environ['CC'], os.environ['CXX'] = shutil.which('gcc') or 'gcc', shutil.which('g++') or 'g++'
    import torch._inductor.config as inductor_config
    inductor_config.cpp.cxx = (None, os.environ['CXX'])
    _point_triton_to_libcuda()
    pat = re.compile(r"^(module\.)?(backbone\.|student_backbone\.)?blocks\.\d+$")
    n = 0
    for name, m in model.named_modules():
        if pat.match(name):
            m.forward = torch.compile(m.forward, dynamic=False)
            n += 1
    print(f"compile_blocks: torch.compile applied to {n} transformer blocks")
    return n


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
    if args.ssl_name == "barlowtwins" and args.bt_fp32_head:
        model_kwargs["fp32_head"] = True
    if args.ssl_name == "dino":
        # None のまま渡せば prepare_model 側で固定運用(0017と同一)になる。
        if args.dino_momentum_start is not None:
            model_kwargs["momentum"] = args.dino_momentum_start
        model_kwargs["momentum_end"] = args.dino_momentum_end
        model_kwargs["teacher_temp_end"] = args.dino_teacher_temp_end
        model_kwargs["teacher_temp_warmup_epochs"] = args.dino_teacher_temp_warmup_epochs
        model_kwargs["out_dim"] = args.dino_out_dim
        model_kwargs["drop_path_rate"] = args.dino_drop_path
        model_kwargs["freeze_last_layer_epochs"] = args.dino_freeze_last_layer
        model_kwargs["fp32_head"] = args.dino_fp32_head
    model, criterion = ssl_class.prepare_model(backbone, head_size=size, **model_kwargs)
    if args.ssl_name in ("barlowtwins", "simsiam"):
        # 学習には影響しない(no_grad の統計を記録するだけ)。再開時は entry.py が復元後の
        # criterion に同じ値を設定し直す。
        criterion.collect_stats = args.bt_step_log or args.step_log
    if args.ssl_name == "barlowtwins":
        configure_bt_criterion(criterion, args)
    if getattr(args, 'compile_blocks', False):
        compile_blocks(model)
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
             'weight_decay_filter': True, 'lars_adaptation_filter': True, 'lr_role': 'bias'},
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

    # warmup_prefix=True の timm は t_initial を「warmup の後の」cosine の長さとして扱う。t_initial=num_epoch だと
    # 減衰の終わりが num_epoch + warmup_t になり、最終 epoch でも lr がピークの数 % 残っていた(SimSiam 160/16 で 3.1%。
    # レビュー指摘 2026-10-12)。warmup を含めて num_epoch で終わるよう、cosine の長さは num_epoch - warmup_t にする。
    # ⚠️ DINO 0028 / BT 0029〜0034 は旧設定(減衰が warmup 分はみ出す + 1 epoch 遅れ)で走っている。
    scheduler = build_scheduler(optimizer, num_epoch, args.warmup_t, args.lr_min, args.warmup_lr_init)
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
            uniformity_threshold=None if args.collapse_ignore_uniformity else -0.05,
            loss_rebound_ratio=args.collapse_loss_rebound,
            loss_rebound_min_epoch=args.collapse_loss_rebound_min_epoch,
            loss_rebound_patience=args.collapse_loss_rebound_patience,
        )
    return model, criterion, optimizer, scheduler, early_stopping, collapse_monitor
