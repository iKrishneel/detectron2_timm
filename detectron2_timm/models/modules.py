#!/usr/bin/env python

import math

import torch
from torch import Tensor
import torch.nn as nn


__all__ = ['XCiT', 'Swin', 'Cait']


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

    def post_forward(self, input, output):
        return output

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
        if isinstance(patch_size, tuple):
            assert (
                patch_size[0] == patch_size[1]
            ), 'Current support square patch'
            patch_size = patch_size[0]

        for stride, remap in zip(strides, remaps):
            factor = patch_size / stride
            if factor == 1:
                op = nn.Identity()
            elif factor > 1:
                op = self.create_upsample_layers(factor)
            else:
                op = self.create_downsample_layers(1.0 / factor)

            setattr(self, remap, op)

    def forward(self, x: Tensor) -> Tensor:
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


class Swin(Base):

    __name__ = 'swin'

    def __init__(self, cfg, model: nn.Module):
        super(Swin, self).__init__(cfg, model)

        strides = self.cfg.MODEL.BACKBONE.CONFIG.STRIDES
        remaps = self.cfg.MODEL.BACKBONE.CONFIG.REMAPS
        if len(remaps) != len(strides):
            remaps = self.cfg.MODEL.BACKBONE.OUT_FEATURES

        assert len(remaps) <= len(model.layers)

        for i, remap in enumerate(remaps):
            blocks = model.layers[i].blocks
            num_blocks = len(blocks) - 1
            assert num_blocks > 0
            num_features = blocks[num_blocks].mlp.fc2.out_features
            self.add_module(remap, nn.LayerNorm(num_features))

        self._remaps = remaps
        self._block_counter = 0
        self._actual_size = None

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            size = (
                self.cfg.INPUT.MIN_SIZE_TRAIN[0],
                self.cfg.INPUT.MAX_SIZE_TRAIN,
            )
        else:
            size = (self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        _, _, h, w = x.shape
        self._actual_size = x.shape[2:]
        x = nn.functional.interpolate(x, size, mode='nearest')

        self._in_shape = x.shape
        self._block_counter = 0

        x = self.model.patch_embed(x)
        if self.model.absolute_pos_embed is not None:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)
        x = self.model.norm(x)
        # x = self.model.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def post_forward(self, input, output):
        i = self._block_counter
        assert self._block_counter < len(self._remaps)
        layer = getattr(self, self._remaps[i])
        assert layer

        s = self.cfg.MODEL.BACKBONE.CONFIG.STRIDES[i]
        b, _, h, w = self._in_shape
        h, w = (h + 1) // s, (w + 1) // s

        x = layer(output)
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        size = (int(self._actual_size[0] // s), int(self._actual_size[1] // s))
        x = nn.functional.interpolate(x, size)
        self._block_counter += 1
        return x


class Cait(XCiT):

    __name__ = 'cait'

    def __init__(self, cfg, model: nn.Module):
        super(Cait, self).__init__(cfg, model)

    def forward(self, x):
        self._in_shape = x.shape
        x = self.model.patch_embed(x)

        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        for i, blk in enumerate(self.model.blocks):
            x = blk(x)

        return x
