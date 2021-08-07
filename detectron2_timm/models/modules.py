#!/usr/bin/env python

import math

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
        self.model.cls_token.requires_grad = False
        delattr(self.model, 'cls_token')

        self.create_pre_fpn()

    def create_upsample_layers(self, factor: int):
        assert factor % 2 == 0
        num_layer = int(math.log2(factor))

        embed_dim = self.model.embed_dim
        stop = 0
        modules = []
        for i in range(num_layer, stop, -1):
            if i == stop + 1:
                modules.append(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2
                    )
                )
            else:
                modules.append(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2
                    )
                )
                modules.append(
                    nn.SyncBatchNorm(embed_dim),
                )
                modules.append(
                    nn.GELU(),
                )
        return nn.Sequential(*modules)

    def create_downsample_layers(self, factor: int):
        factor = int(factor)
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=factor, stride=factor),
        )

    def create_pre_fpn(self):
        strides = self.cfg.MODEL.BACKBONE.CONFIG.STRIDES
        remaps = self.cfg.MODEL.BACKBONE.CONFIG.REMAPS
        if len(remaps) != len(strides):
            remaps = self.cfg.MODEL.BACKBONE.OUT_FEATURES

        assert len(strides) == len(remaps)

        patch_size = self.model.patch_embed.patch_size

        for stride, remap in zip(strides, remaps):
            factor = patch_size / stride
            if factor == 1:
                op = nn.Identity()
            elif factor > 1:
                op = self.create_upsample_layers(factor)
            else:
                op = self.create_downsample_layers(1.0 / factor)

            setattr(self, remap, op)

    def forward(self, x):
        self._in_shape = x.shape
        b = x.shape[0]
        x, (Hp, Wp) = self.model.patch_embed(x)

        if self.model.use_pos_embed:
            x = (
                x
                + self.model.pos_embed(b, Hp, Wp)
                .reshape(b, -1, x.shape[1])
                .permute(0, 2, 1)
                .contiguous()
            )

        x = self.model.pos_drop(x)

        for blk in self.model.blocks:
            x = blk(x, Hp, Wp)
        return x

    def feature_res_adj(self, features: dict) -> dict:
        assert self._in_shape
        out_features = {
            name: getattr(self, name)(features[name]) for name in features
        }
        return out_features
