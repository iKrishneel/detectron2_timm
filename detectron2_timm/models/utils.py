#!/usr/bin/env python

from typing import List
from difflib import get_close_matches

import torch.nn as nn

import timm.models as tmodels
from detectron2.modeling import ShapeSpec
from detectron2.config import CfgNode


_PREFIX: str = 'build_'
_SUFFIX: str = '_backbone'


def get_models(default_cfgs: dict, name: str = None) -> List[str]:
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


def remove_named_children(cfg, original_model) -> nn.Module:
    removals = cfg.MODEL.BACKBONE.REMOVE_LAYERS
    model = nn.Sequential()
    for name, module in original_model.named_children():
        if name in removals:
            continue
        model.add_module(name, module)
    return model


def find_model_config(model_cfgs: dict, model_name: str) -> dict:
    match = get_close_matches(model_name, model_cfgs.keys())[0]
    return model_cfgs[match]


def get_func_name(model_name: str) -> str:
    assert len(model_name) > 0
    return f'{_PREFIX + model_name + _SUFFIX}'


def get_model_name(func_name: str, replace_with: str='') -> str:
    assert len(func_name) > 0
    name = func_name.replace(_PREFIX, replace_with)
    return name.replace(_SUFFIX, replace_with)
