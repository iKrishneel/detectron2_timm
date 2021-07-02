#!/usr/bin/env python

from typing import List, Union
import torch.nn as nn

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

        # self.out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

        model_config = kwargs.get('model_config')
        self.parse_model_config(model, model_config)

        self.model = model
        self.model_config = model_config

    def parse_model_config(self, model, model_config):

        key = 'remove'
        if key in model_config:
            # TODO remove these layers
            pass

        key = 'out_features'
        if key in model_config:
            self.break_model(model, model_config[key])
        else:
            print(model_config)
            print(key)

        key = 'add'
        if key in model_config:
            # TODO: create new layers
            pass

    def forward(self, x):
        x = self.model(x)
        assert len(self.feature_maps) == len(self.channels)

        output = {
            stage: fmap
            for fmap, stage in zip(self.feature_maps, self.out_features)
        }
        self.feature_maps = []
        return output

    def forward_hook(self, module, input, output):
        self.feature_maps.append(output)

    def break_model(self, model, break_cfg: dict):

        print(break_cfg)
        
        layers = break_cfg['layers']
        strides = break_cfg['strides']

        if not isinstance(layers, list):
            layers = [layers]
        if not isinstance(strides, list):
            strides = [strides]

        assert len(layers) == len(strides)

        prefix = break_cfg.get('prefix', 'c')

        for i, layer in enumerate(layers, 1):
            splited = layer.split('.')
            module = model
            for s in splited:
                module = getattr(module, s)

            module.register_forward_hook(self.forward_hook)

            stage_name = f'{prefix}{i}'
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

    """
    def get_stage_name(self, i: int) -> str:
        return f'stage{i}'

    def get_stage_index(self, name: str) -> int:
        return int(''.join(filter(str.isdigit, name)))
    """
