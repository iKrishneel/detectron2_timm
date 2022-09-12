#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple
from detectron2_timm.layers.base import LayerBase, LAYER_REGISTRY


__all__ = ['ResNetStem']


class ResNetStem(LayerBase):
    def __init__(self, cfg):
        super(ResNetStem, self).__init__()
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),                
            ]
        )

    def forward(self, x):
        return self.stem(x)


@LAYER_REGISTRY.register()
def resnet_stem(cfg):
    return ResNetStem(cfg)
