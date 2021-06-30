#!/usr/bin/env python

import functools
import inspect
from difflib import get_close_matches
from typing import List, Union

import torch
import torch.nn as nn

import timm.models as tmodels
from timm.models.registry import list_models
from timm.models.byoanet import default_cfgs, model_cfgs

from detectron2.modeling import Backbone, BACKBONE_REGISTRY, ShapeSpec

from detectron2_timm.config import get_cfg, cfg_from_defaults

from detectron2.config import CfgNode


__all__ = []


def get_models(name: str = None) -> List[str]:
    if name is not None:
        try:
            assert name in default_cfgs
            return getattr(tmodels, name)
        except:
            raise ValueError(f'Model name {model_name} not found')

    return {
        model_name: getattr(tmodels, model_name)
        for model_name in default_cfgs.keys()
    }


def remove_named_children(cfg, original_model):
    removals = cfg.MODEL.BACKBONE.REMOVE_LAYERS
    model = nn.Sequential()
    for name, module in original_model.named_children():
        if name in removals:
            continue
        model.add_module(name, module)
    return model


def remove_layers(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cfg = args[0]
        model = kwargs.pop('model')

        attrs = {
            'img_size': (cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN),
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES
        }        
        model = model(**attrs)
        model = remove_named_children(cfg, model)
        func(self, cfg, model=model, **kwargs)
    return wrapper


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


_PREFIX = 'build_'
_SUFFIX = '_backbone'


def find_model_config(model_name: str):
    match = get_close_matches(model_name, model_cfgs.keys())[0]
    return model_cfgs[match]


def get_func_name(model_name: str) -> str:
    assert len(model_name) > 0
    return f'{_PREFIX + model_name + _SUFFIX}'


def get_model_name(func_name: str, replace_with: str=''):
    assert len(func_name) > 0
    name = func_name.replace(_PREFIX, replace_with)
    return name.replace(_SUFFIX, replace_with)


def build_detectron2_model(cfg: CfgNode, input_shape: ShapeSpec) -> Model:
    func_name = cfg.MODEL.BACKBONE.NAME
    assert func_name in __all__ , f'{func_name} not found'

    model_name = get_model_name(func_name)
    model = Model(
        cfg, model=get_models()[model_name],
        model_config=find_model_config(model_name)
    )    
    return model
    

def hook(local_s, model_name: str, **kwargs: dict):
    func_name = get_func_name(model_name=model_name)
    local_s.update({func_name: build_detectron2_model})
    BACKBONE_REGISTRY._do_register(func_name, local_s[func_name])
    __all__.append(func_name)


def embed(local_s):
    model_dict = get_models()

    for model_name in model_dict:
        hook(local_s, model_name=model_name)


embed(locals())


if __name__ == '__main__':

    cfg = get_cfg()
    cfg = cfg_from_defaults(cfg, default_cfgs)

    # m = build_botnet26t_256_backbone(cfg, 1)
    cfg.MODEL.BACKBONE.NAME = "build_halonet26t_backbone"

    from detectron2.modeling import build_model
    r = {'image': torch.randn((3, 1024, 1024))}
    t = build_model(cfg)

    # print(">>> ", m.stage4[0].self_attn.pos_embed.height, m.stage4[0].self_attn.pos_embed.width)
    
    t.eval()
    print(t)
    z = t([r])

    import IPython
    IPython.embed()
