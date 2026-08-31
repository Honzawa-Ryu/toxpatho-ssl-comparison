# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

utilities

@author: Katsuhisa, tadahaya
"""
import os
import datetime
import random
import logging
from typing import List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import matplotlib.pyplot as plt
from sklearn import metrics
from PIL import ImageOps, Image
import torchvision.transforms as transforms

from lib.sslmodel.utils_lightly import RandomRotate, RandomSolarization

# assist model building
def fix_seed(seed:int=None,fix_gpu:bool=True):
    """ fix seed """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def fix_params(model, forall=False):
    """ freeze model parameters """
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # except last layer
    if forall:
        pass
    else:
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
    return model

def unfix_params(model):
    """ unfreeze model parameters """
    # activate layers 
    for param in model.parameters():
        param.requires_grad = True
    return model

def random_rotation_transform(
    rr_prob: float = 0.5,
    rr_degrees: Union[None, float, Tuple[float, float]] = 90,
    ) -> Union[RandomRotate, transforms.RandomApply]:
    if rr_degrees == 90:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=rr_degrees)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return transforms.RandomApply([transforms.RandomRotation(degrees=rr_degrees)], p=rr_prob)

def ssl_transform(
    split=False, multi=False,
    size=(224,224),
    color_plob=0.8,
    blur_plob=0.2,
    solar_plob=0,
    ):
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # augmentation
    # scale=(0.2, 1.0): 病理パッチはWSIから224pxで切り出し済みのため、
    # デフォルト(0.08,1.0)だと最小63pxになり微細構造が失われる
    # RandomGrayscale: H&E染色の色情報(青紫=核, ピンク=細胞質)は診断的に重要なため低確率に抑える
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob
            ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob
            ),
        RandomSolarization(prob=solar_plob),
        transforms.ToTensor(),
        normalize
    ])

    # set
    if split:
        if multi:
            return MultiCropsTransform(augmentation)
        else:
            return TwoCropsTransform(augmentation)
    else:
        return augmentation


class MultiCropList:
    """Apply a list of per-crop transforms to one image, returning a list of views."""
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, x):
        return [t(x) for t in self.transforms_list]


def multicrop_transform(
    n_global: int = 2, n_local: int = 6,
    global_size: int = 224, local_size: int = 96,
    global_scale=(0.2, 1.0), local_scale=(0.05, 0.2),
    color_plob: float = 0.8, blur_plob: float = 0.5, solar_plob: float = 0.0,
    ):
    """Paper-style multi-crop (SwAV / DINO): n_global full-res global crops +
    n_local low-res local crops. Each crop applies a SINGLE RandomResizedCrop at
    its own resolution (no double-crop), then the shared photometric aug tail.
    Views are ordered [global..., local...] so same-resolution views are contiguous
    (required by DINO._forward_views grouping and default_collate per-view stacking)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    def tail():
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            random_rotation_transform(rr_prob=1., rr_degrees=[0, 180]),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob),
            RandomSolarization(prob=solar_plob),
            transforms.ToTensor(),
            normalize,
        ]

    g = transforms.Compose([transforms.RandomResizedCrop(global_size, scale=global_scale), *tail()])
    l = transforms.Compose([transforms.RandomResizedCrop(local_size, scale=local_scale), *tail()])
    return MultiCropList([g] * n_global + [l] * n_local)

def weak_strong_transform(
    size=(224,224),
    color_plob_w=0.2,
    blur_plob_w=0.1,
    solar_plob_w=0,
    color_plob_s=1,
    blur_plob_s=0.8,
    solar_plob_s=0.2,
    ):
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # augmentation (weak)
    weak_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob_w
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob_w
            ),
        RandomSolarization(prob=solar_plob_w),
        transforms.ToTensor(),
        normalize
    ])

    # augmentation (strong)
    strong_augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=color_plob_s
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=blur_plob_s
            ),
        RandomSolarization(prob=solar_plob_s),
        transforms.ToTensor(),
        normalize
    ])

    return WeakStrongTwoCropsTransform(weak_augmentation, strong_augmentation)


def my_transform():
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # augmentation
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
            ),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    # set
    train_data_transform = augmentation
    other_data_transform = transforms.Compose([
        transforms.CenterCrop((224,224)), #230724 fixed
        transforms.ToTensor(),
        normalize
    ])
    return train_data_transform, other_data_transform

class WeakStrongTwoCropsTransform:
    """Take two crops of one image as the query and key."""

    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        q = self.weak_transform(x)
        k = self.strong_transform(x)
        return [q, k]

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class MultiCropsTransform:
    """ return high and low resolution different crops from one image """
    def __init__(
        self, base_transform, 
        crop_counts=[2,6], 
        crop_sizes=[224,96]
        ):
        self.base_transform = base_transform
        self.crop_counts=crop_counts
        self.crop_sizes=crop_sizes
        
        # list of transforms for crop images
        self.crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = transforms.RandomResizedCrop(
                (crop_sizes[i], crop_sizes[i])
            )
            self.crop_transforms.extend([
                transforms.Compose([
                    random_resized_crop,
                    base_transform
                ])] * crop_counts[i]
            )

    def __call__(self, x):
        views = [crop_transform(x) for crop_transform in self.crop_transforms]
        return views

# logger
class logger_save():
    def __init__(self):
        self.tag=None
        self.level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
        self.logger=None
        self.init_info=None
        self.level_console=None
        self.module_name=None

    def init_logger(self, module_name:str, outdir:str='', tag:str='',
                    level_console:str='warning', level_file:str='info'):
        #setting
        if len(tag)==0:
            tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.init_info={
            'level':self.level_dic[level_file],
            'filename':f'{outdir}/log_{tag}.txt',
            'format':'[%(asctime)s] [%(levelname)s] %(message)s',
            'datefmt':'%Y%m%d-%H%M%S'
            }
        self.level_console=level_console
        self.module_name=module_name
        #init
        logging.basicConfig(**self.init_info)
        logger = logging.getLogger(self.module_name)
        sh = logging.StreamHandler()
        sh.setLevel(self.level_dic[self.level_console])
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
            )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        self.logger=logger

    def load_logger(self, filein:str=''):
        # load
        self.__dict__.update(pd.read_pickle(filein))
        #init
        logging.basicConfig(**self.init_info)
        logger = logging.getLogger(self.module_name)
        sh = logging.StreamHandler()
        sh.setLevel(self.level_dic[self.level_console])
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            "%Y%m%d-%H%M%S"
            )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        self.logger=logger

    def save_logger(self, fileout:str=''):
        pd.to_pickle(self.__dict__, fileout)

    def to_logger(self, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True):
        """ add instance information to logging """
        self.logger.info(name)
        for k,v in vars(obj).items():
            if k not in skip_keys:
                if skip_hidden:
                    if not k.startswith('_'):
                        self.logger.info('  {0}: {1}'.format(k,v))
                else:
                    self.logger.info('  {0}: {1}'.format(k,v))

def init_logger(
    module_name:str, outdir:str='', tag:str='',
    level_console:str='warning', level_file:str='info'
    ):
    """
    initialize logger
    
    """
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag)==0:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(
        level=level_dic[level_file],
        filename=f'{outdir}/log_{tag}.txt',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y%m%d-%H%M%S',
        )
    logger = logging.getLogger(module_name)
    sh = logging.StreamHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
        )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def to_logger(
    logger, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True
    ):
    """ add instance information to logging """
    logger.info(name)
    for k,v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden:
                if not k.startswith('_'):
                    logger.info('  {0}: {1}'.format(k,v))
            else:
                logger.info('  {0}: {1}'.format(k,v))

# learning tools
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    add some changes from from https://github.com/Bjarten/early-stopping-pytorch/pytorchtools.py
    """
    def __init__(self, patience:int=7, delta:float=0, path:str='checkpoint.pt', save_enabled:bool=True):
        """
        Parameters
        ----------
            patience (int)
                How long to wait after last time validation loss improved.

            delta (float)
                Minimum change in the monitored quantity to qualify as an improvement.

            path (str):
                Path for the checkpoint to be saved to.

            save_enabled (bool)
                Trueならcheckpoint.ptへの実書き込みを行う。マルチGPU(DDP)実行時、
                全rankが同一のval_lossを見て内部状態(counter/best_score/early_stop)は
                揃えたいが、ファイル書き込みは rank0 だけにしたい場合に False を渡す
                （lib/trainer/model.py: save_enabled=distributed.is_main_process()）。
                デフォルトTrueなので単一プロセス実行時の挙動は変わらない。

        """
        self.patience = patience
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.save_enabled = save_enabled

    def __call__(self, val_loss, model):
        if val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        if self.save_enabled:
            torch.save(model.state_dict(), self.path)

    def delete_checkpoint(self):
        os.remove(self.path)


class CollapseMonitor:
    """
    Aborts training if the representation stays collapsed for consecutive checks.

    Unlike EarlyStopping (which reacts to val_loss plateauing), this reacts to
    representation collapse so a diagnostic/isolation run doesn't burn GPU-hours
    on an already-dead model. Goal.yaml treats collapse monitoring as a health
    check, not a stop criterion, for the main `paper_*` comparison runs; this
    class is opt-in (see `--collapse_early_stop`) and left unused there.

    判定は3指標のOR（どれか1つでも patience 回連続で該当したら崩壊とみなす）:

    1. `train_loss >= ln(out_dim) * loss_ratio`
       DINOの一様崩壊。student/teacherがともに一様分布になると損失は厳密に
       ln(out_dim)(out_dim=8192 なら 9.0109)で固定され、勾配が厳密に0になる
       吸収状態に入る。**最も鋭敏で誤検知が少ない**ため第一指標。
    2. `uniformity > uniformity_threshold`
       Wang & Isola の uniformity は「全サンプルが1点に潰れると0に漸近」する。
       DINO以外(out_dimを持たない手法)でも効く汎用指標。
    3. `effective_rank < rank_threshold`
       従来の指標。**鈍すぎるので単独では使い物にならない**: 0023 ep5 は
       loss=ln(8192)ちょうど・feat_std 0.055 と壊滅状態なのに eff_rank は 31.99
       あり、閾値5.0では発火しなかった。0024 は36 epoch崩壊し続けたが最後まで
       発火しなかった。後方互換のため残してあるだけ。

    注意: `alignment` は Wang & Isola の alignment **loss**(正例ペア間の正規化後
    2乗距離)であり **小さいほど良い**。崩壊時も0に近づくため単独では判定に使えない
    (2026-08-29のログで「alignment 0.0032は壊滅」と誤読した経緯がある)。
    """
    def __init__(self, rank_threshold: float = 5.0, patience: int = 2,
                 out_dim: int = 0, loss_ratio: float = 0.999,
                 uniformity_threshold: float = -0.05):
        self.rank_threshold = rank_threshold
        self.patience = patience
        self.loss_ceiling = float(np.log(out_dim)) if out_dim and out_dim > 1 else None
        self.loss_ratio = loss_ratio
        self.uniformity_threshold = uniformity_threshold
        self.counter = 0
        self.collapsed = False
        self.reason = ''

    def update(self, effective_rank, uniformity=None, train_loss=None):
        # NaN (e.g. from a diverged/collapsed run whose SVD blows up) must be treated
        # like None here: `NaN < threshold` is always False in Python, so without this
        # check a NaN reading would silently reset the counter instead of being ignored.
        def _valid(x):
            return x is not None and not np.isnan(x)

        reasons = []
        if self.loss_ceiling is not None and _valid(train_loss) \
                and train_loss >= self.loss_ceiling * self.loss_ratio:
            reasons.append(f'train_loss {train_loss:.4f} >= ln(out_dim) x {self.loss_ratio} '
                           f'(= {self.loss_ceiling * self.loss_ratio:.4f}) — uniform collapse')
        if _valid(uniformity) and uniformity > self.uniformity_threshold:
            reasons.append(f'uniformity {uniformity:.4f} > {self.uniformity_threshold} '
                           f'— all samples collapsed onto one point')
        if _valid(effective_rank) and effective_rank < self.rank_threshold:
            reasons.append(f'effective_rank {effective_rank:.2f} < {self.rank_threshold}')

        if reasons:
            self.counter += 1
            self.reason = '; '.join(reasons)
            if self.counter >= self.patience:
                self.collapsed = True
        else:
            self.counter = 0
            self.reason = ''


def set_criterion(criterion_name="BCE"):
    if criterion_name == "BCE":
        criterion = nn.BCEWithLogitsLoss()
        preprocess = lambda x: x.sigmoid()
    elif criterion_name == "WeightedBCE":
        positive_weight = train_info[ft_list].mean().values.astype(np.float32)
        positive_weight = torch.tensor((1 - positive_weight)/(positive_weight + 1e-5))
        criterion = WeightedBCELossWithLogits(positive_weight=positive_weight, negative_weight=torch.tensor(1), device=device)
        preprocess = lambda x: x.sigmoid()
    elif criterion_name == "FocalBCE":
        criterion = FocalBCELossWithLogits(gamma=1)
        preprocess = lambda x: x.sigmoid()
    elif criterion_name == "MSE":
        criterion = nn.MSELoss()
        preprocess = lambda x: x
    return criterion, preproceess

# save & export
def summarize_model(model, summary, outdir, lst_name=['summary.txt', 'model.pt']):
    """
    summarize model using torchinfo

    Parameters
    ----------
    outdir: str
        output directory path

    model:
        pytorch model
    
    size:
        size of input tensor
    
    """
    try:
        with open(f'{outdir}/{lst_name[0]}', 'w') as writer:
            writer.write(repr(summary))
    except ModuleNotFoundError:
        print('!! CAUTION: no torchinfo and model summary was not saved !!')
    torch.save(model.state_dict(), f'{outdir}/{lst_name[1]}')


