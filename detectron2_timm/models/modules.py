#!/usr/bin/env python

import math
from typing import List

import torch
import torch.nn as nn


__all__ = [
    'XCiT',
]


def wrap(cfg, model: nn.Module) -> nn.Module:
    model_name = cfg.MODEL.BACKBONE.NAME
    assert model_name and callable(model)

    for name in __all__:
        cls = globals()[name]
        assert cls

        if cls.__name__.lower() in model_name:
            return cls(cfg, model)

    return Base(cfg, model)


class Base(nn.Module):

    def __init__(self, cfg, model):
        super(Base, self).__init__()
        self.cfg = cfg
        self.model = model

    def forward(self, x):
        return self.model(x)

    def feature_res_adj(self, x):
        return x


class XCiT(Base):

    __name__ = 'xcit'
    
    def __init__(self, cfg, model: nn.Module):
        super(XCiT, self).__init__(cfg, model)

        self._in_shape = None
        
    def forward(self, x):
        self._in_shape = x.shape
        b = x.shape[0]
        # x is (b, N, C). (Hp, Hw) is (height in units of patches, width in units of patches)
        x, (Hp, Wp) = self.model.patch_embed(x)

        if self.model.use_pos_embed:
            # `pos_embed` (b, C, Hp, Wp), reshape -> (b, C, N), permute -> (b, N, C)
            pos_encoding = self.model.pos_embed(b, Hp, Wp).reshape(
                b, -1, x.shape[1]
            ).permute(0, 2, 1)
            x = x + pos_encoding

        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x, Hp, Wp)
        return x

    def feature_res_adj(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        assert self._in_shape

        h, w = self._in_shape[2:]
        strides = self.cfg.MODEL.BACKBONE.CONFIG.STRIDES
        remaps = self.cfg.MODEL.BACKBONE.CONFIG.REMAPS
        if len(remaps) != len(strides):
            remaps = self.cfg.MODEL.BACKBONE.OUT_FEATURES

        out_features = {}
        for s, remap in zip(strides, remaps):
            feature = features[remap]
            size = (h // s, w // s)

            if size[0] == feature.shape[2]:
                ops = nn.Identity()
            elif size[0] > feature.shape[2]:
                ops = nn.Upsample(size=size)
            elif size[0] < feature.shape[2]:
                size = (
                    feature.shape[2] // size[0],
                    feature.shape[3] // size[1]
                )
                ops = nn.MaxPool2d(kernel_size=3, stride=size[0], padding=1)
            out_features[remap] = ops(feature)

        return out_features
