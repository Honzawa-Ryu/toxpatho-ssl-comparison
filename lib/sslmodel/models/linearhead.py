# -*- coding: utf-8 -*-
"""
# leaner head module for classification

@author: Katsuhisa MORITA
"""

import torch
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, backbone, num_classes=9, dim:int=2048):
        super(LinearHead, self).__init__()
        self.backbone=backbone
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out