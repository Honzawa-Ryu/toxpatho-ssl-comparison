# -*- coding: utf-8 -*-
"""
# byol module

reference: https://github.com/lucidrains/byol-pytorch

@author: Katsuhisa MORITA
"""

import copy
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F

# helper
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

class EMA():
    """ Exponential Moving Average """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)
    
# Projector / Predictor
def MLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=512):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

# Main Wrapper
class NetWrapper(nn.Module):
    def __init__(self, backbone, projection_size, projection_hidden_size, layer = -2, use_simsiam_mlp = False):
        super().__init__()
        self.backbone = backbone
        self.layer = layer
        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.use_simsiam_mlp = use_simsiam_mlp
        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.backbone.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.backbone.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        layer = self._find_layer()
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        dim = hidden.shape[1] # changed from _, dim
        create_mlp_fn = SimSiamMLP if self.use_simsiam_mlp else MLP
        projector = create_mlp_fn(
            dim, 
            self.projection_size, 
            self.projection_hidden_size
            )
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.backbone(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.backbone(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        representation = torch.flatten(representation, start_dim=1)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# Main Class
class BYOL(nn.Module):
    def __init__(
        self,
        backbone,
        image_size,
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        DEVICE=None
    ):
        super().__init__()
        self.backbone = backbone
        self.online_encoder = NetWrapper(backbone, projection_size, projection_hidden_size, layer=hidden_layer, use_simsiam_mlp=False)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(
            projection_size, 
            projection_size, 
            projection_hidden_size)

        self.to(DEVICE)
        # send a mock image tensor to instantiate singleton parameters
        self.forward(
            x1=torch.randn(2, 3, image_size, image_size, device=DEVICE),
            x2=torch.randn(2, 3, image_size, image_size, device=DEVICE)
            )

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        for p in target_encoder.parameters():
            p.requires_grad = False
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        for current_params, ma_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            ma_params.data = self.target_ema_updater.update_average(ma_params.data, current_params.data)

    def forward(self, x1, x2):
        online_proj_one, _ = self.online_encoder(x1)
        online_proj_two, _ = self.online_encoder(x2)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one, _ = target_encoder(x1)
            target_proj_two, _ = target_encoder(x2)
            target_proj_one.detach_()
            target_proj_two.detach_()

        return online_pred_one, online_pred_two, target_proj_one, target_proj_two
