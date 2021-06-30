#!/usr/bin/env python

from typing import Union
import torch.nn as nn

from detectron2.modeling import Backbone, ShapeSpec

from .decorators import remove_layers


__all__ = ['Model']


class Model(Backbone):

    @remove_layers
    def __init__(self, cfg, *, model, **kwargs):
        super(Model, self).__init__()
        
        assert getattr(model, 'stages')
        assert getattr(model, 'stem')

        self._config = kwargs.get('model_config')
        
        self.out_features = cfg.MODEL.BACKBONE.OUT_FEATURES
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

        self._divide_stages(model)
        self._freeze_at(freeze_at)

        self.strides = {'stem': getattr(self, 'stem').stride}
        self.channels = {'stem': self._config.stem_chs}

        prev_stride = self.strides['stem']
        for i, block in enumerate(self._config.blocks, 1):
            if isinstance(block, (tuple, list)):
                stride = max([b.s for b in block])
                channels = max([b.c for b in block])
            else:
                stride = block.s
                channels = block.c
            stage_name = self.get_stage_name(i)
            self.channels[stage_name] = channels
            self.strides[stage_name] = prev_stride * stride
            prev_stride *= stride

    def forward(self, x) -> dict:
        x = self.stem(x)
        features = {}
        for stage_name in self.stage_names:
            x = getattr(self, stage_name)(x)
            if stage_name in self.out_features:
                features[stage_name] = x
        if len(features) == 0:
            features[stage_name] = x
        return features

    def output_shape(self) -> dict:
        return {
            name: ShapeSpec(
                channels=self.channels[name], stride=self.strides[name]
            )
            for name in self.out_features
        }

    def _divide_stages(self, original_model) -> None:
        setattr(self, 'stem', original_model.stem)
        self.stage_names = []
        for i, child in enumerate(original_model.stages.children()):
            stage_name = self.get_stage_name(i + 1)
            setattr(self, stage_name, child)
            self.stage_names.append(stage_name)

    def _freeze_at(self, at: Union[int, str]) -> None:
        assert len(self.stage_names) > 0

        if at == 0:
            return
        
        if isinstance(at, int):
            stage_name = self.get_stage_name(at)

        assert stage_name in self.stage_names, f'Cannot freeze the {stage_name}'

        # TODO

    def get_stage_name(self, i: int) -> str:
        return f'stage{i}'

    def get_stage_index(self, name: str) -> int:
        return int(''.join(filter(str.isdigit, name)))
