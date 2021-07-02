#!/usr/bin/env python

from typing import List, Union
import torch.nn as nn

from detectron2.config import CfgNode
from detectron2.modeling import Backbone as BB, ShapeSpec

from .decorators import remove_layers


__all__ = ['Backbone']


class Backbone(BB):

    @remove_layers
    def __init__(self, cfg, *, model, **kwargs) -> None:
        super(Backbone, self).__init__()

        self.out_features = []
        self.strides = {}
        self.channels = {}
        self.feature_maps = []

        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        model_config = cfg.MODEL.BACKBONE.CONFIG
        assert model_config
        
        self.parse_model_config(model, model_config)

        self.model = model
        self.model_config = model_config

    def parse_model_config(self, model, model_config):

        model_config.OUT_FEATURES
        self.break_model(model, model_config)

        # TODO: create new layers
        
    def forward(self, x):
        self.feature_maps = []
        x = self.model(x)
        assert len(self.feature_maps) == len(self.channels)

        output = {
            stage: fmap
            for fmap, stage in zip(self.feature_maps, self.out_features)
        }
        return output

    def forward_hook(self, module, input, output):
        self.feature_maps.append(output)

    def break_model(self, model, model_cfg: CfgNode):

        layers = model_cfg.OUT_FEATURES
        strides = model_cfg.STRIDES
        remaps = model_cfg.REMAPS

        if not isinstance(layers, list):
            layers = [layers]
        if not isinstance(strides, list):
            strides = [strides]
        
        assert len(layers) > 0 and len(layers) == len(strides), \
            'STRIDES and OUT_FEATURES must be same size and > 0'

        if len(remaps) > 0 and len(layers) != len(remaps):
            raise ValueError(
                'REMAP can either be empty or same size as OUT_FEATURES'
            )
        
        for i, layer in enumerate(layers):
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
            self.channels[stage_name] = self.get_channels(module)
            self.strides[stage_name] = strides[i - 1]

    def get_strides(self, module) -> int:
        stride = 1
        # TODO: estimate module stride
        return stride

    def get_channels(self, module) -> int:
        out_channels = 1
        for m in module.modules():
            if len(list(m.children())) != 0:
                continue
            try:
                out_channels = m.out_channels
            except AttributeError:
                pass
        return out_channels

    def output_shape(self) -> dict:
        return {
            name: ShapeSpec(
                channels=self.channels[name], stride=self.strides[name]
            )
            for name in self.out_features
        }

    def _freeze_at(self, at: Union[int, str]) -> None:
        assert len(self.stage_names) > 0

        if at == 0:
            return
        
        if isinstance(at, int):
            stage_name = self.get_stage_name(at)

        assert stage_name in self.stage_names, f'Cannot freeze the {stage_name}'

        # TODO
