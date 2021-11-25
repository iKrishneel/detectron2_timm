#!/usr/bin/env python

import math
from typing import List

import torch.nn as nn
from einops import rearrange

from detectron2.config import CfgNode
from detectron2.modeling import Backbone as BB, ShapeSpec
from detectron2.utils.logger import logging

from .decorators import remove_layers


__all__ = ['Backbone']

logger = logging.getLogger(__name__)


class Backbone(BB):
    @remove_layers
    def __init__(self, cfg, *, model, **kwargs) -> None:
        super(Backbone, self).__init__()

        self.out_features = []
        self.strides = {}
        self.channels = {}
        self.feature_maps = []

        self._feature_remap = {}
        self._in_shape = None

        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        model_config = cfg.MODEL.BACKBONE.CONFIG
        assert model_config

        if len(model_config.OUT_FEATURES) == 0:
            logger.info('Using default feature info')
            model_config = self.config_from_feature_info(
                model_config, kwargs.get('feature_info')
            )

        self.parse_model_config(model.model, model_config)
        self.model = model
        self.model_config = model_config

        self._freeze_at(at=freeze_at)

    def parse_model_config(self, model, model_config):

        model_config.OUT_FEATURES
        self.break_model(model, model_config)

        # TODO: create new layers

    def forward(self, x):
        self._in_shape = x.shape
        self.feature_maps = []
        x = self.model(x)

        assert len(self.feature_maps) == len(self.channels)

        output = {
            stage: fmap
            for fmap, stage in zip(self.feature_maps, self.out_features)
        }
        return self.model.feature_res_adj(output)

    def forward_hook(self, module, input, output):

        post_output = self.model.post_forward(input, output)

        if len(post_output.shape) == 3:
            h, w = self._in_shape[2:]
            k = int(math.sqrt((h * w) // post_output.shape[1]))
            self.feature_maps.append(
                rearrange(
                    post_output,
                    'b (h1 w1) c -> b c h1 w1',
                    h1=h // k,
                    w1=w // k,
                )
            )
        else:
            self.feature_maps.append(post_output)

    def break_model(self, model, model_cfg: CfgNode):

        layers = model_cfg.OUT_FEATURES
        strides = model_cfg.STRIDES
        remaps = model_cfg.REMAPS
        channels = model_cfg.CHANNELS

        if not isinstance(layers, list):
            layers = [layers]
        if not isinstance(strides, list):
            strides = [strides]

        assert len(layers) > 0 and len(layers) == len(
            strides
        ), 'STRIDES and OUT_FEATURES must be same size and > 0'

        if len(remaps) > 0 and len(layers) != len(remaps):
            raise ValueError(
                'REMAP can either be empty or same size as OUT_FEATURES'
            )

        for i, (layer, stride) in enumerate(zip(layers, strides)):
            splited = layer.split('.')
            module = model
            for s in splited:
                module = getattr(module, s)

            module.register_forward_hook(self.forward_hook)

            try:
                stage_name = remaps[i]
            except IndexError:
                stage_name = layer

            self.out_features.append(stage_name)
            self.strides[stage_name] = stride
            self._feature_remap[stage_name] = layer

            self.channels[stage_name] = (
                channels[i]
                if len(channels) == len(strides)
                else self.get_channels(module)
            )

    def get_strides(self, module) -> int:
        stride = 1
        # TODO: estimate module stride
        return stride

    def get_channels(self, module) -> int:
        out_channels = None
        for m in module.modules():
            if len(list(m.children())) != 0:
                continue

            if isinstance(m, nn.Conv2d):
                out_channels = m.out_channels
            elif isinstance(m, nn.Linear):
                out_channels = m.out_features

        assert out_channels
        return out_channels

    def output_shape(self) -> dict:
        return {
            name: ShapeSpec(
                channels=self.channels[name], stride=self.strides[name]
            )
            for name in self.out_features
        }

    def config_from_feature_info(
        self, cfg, feature_infos: List[dict]
    ) -> CfgNode:
        assert len(feature_infos) > 0

        for feature_info in feature_infos:
            cfg.OUT_FEATURES.append(feature_info['module'])
            cfg.STRIDES.append(feature_info['reduction'])
        return cfg

    def _freeze_at(self, at: str) -> None:
        # TODO: better to forward and freeze
        if at < 1:
            return
        assert (at - 1) <= len(
            self.out_features
        ), f'Freeze at: {at}, is outside the lenght of {self.out_features}'

        freeze_layer = self._feature_remap[self.out_features[at - 1]]
        for name, feature in self.model.named_children():
            for params in feature.parameters():
                params.requires_grad = False
            if name == freeze_layer:
                break
