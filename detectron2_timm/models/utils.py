#!/usr/bin/env python

import os.path as osp
import yaml

from typing import List
from difflib import get_close_matches

import torch.nn as nn

import timm.models as tmodels
from detectron2.config import CfgNode


_PREFIX: str = 'build_'
_DETECTRON2: str = 'detectron2'
_SUFFIX: str = '_backbone'


def get_models(default_cfgs: dict, name: str = None) -> List[str]:
    if name is not None:
        try:
            assert name in default_cfgs
            return getattr(tmodels, name)
        except AttributeError:
            raise ValueError(f'Model name {name} not found')

    return {
        model_name: getattr(tmodels, model_name)
        for model_name in default_cfgs.keys()
    }


def remove_named_children(cfg, original_model) -> nn.Module:
    removals = cfg.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS
    # model = nn.Sequential()
    for name, module in original_model.named_children():
        if name in removals:
            setattr(original_model, name, nn.Identity())
            # continue
        # model.add_module(name, module)
    # return model
    return original_model


def replace_layers(cfg, original_model):

    import importlib

    layers = importlib.import_module('detectron2_timm.models.layers')

    for replace in cfg.MODEL.BACKBONE.CONFIG.REPLACE_LAYERS:
        assert isinstance(replace, (list, tuple))
        if len(replace) == 0:
            continue
        layer = getattr(layers, replace[1])
        original_model.add_module(replace[0], layer())

    return original_model


def find_model_config(model_cfgs: dict, model_name: str) -> dict:
    match = get_close_matches(model_name, model_cfgs.keys())[0]
    return model_cfgs[match]


def get_func_name(func, model_name: str) -> str:
    assert len(model_name) > 0 and callable(func)
    return func.__name__.replace(_DETECTRON2, model_name)


def get_model_name(func_name: str, replace_with: str = '') -> str:
    assert len(func_name) > 0
    name = func_name.replace(_PREFIX, replace_with)
    name = name.replace(_SUFFIX, replace_with)
    return name.replace('_fpn', replace_with)


def load_yaml(path: str) -> dict:
    assert osp.isfile(path)
    with open(path, 'r') as stream:
        data = yaml.safe_load(stream)
    return data


def get_attr(name: str):
    return getattr(tmodels, name)


def patch_size_mod2(cfg: CfgNode) -> List[int]:
    stride = max(cfg.MODEL.BACKBONE.CONFIG.STRIDES)
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    min_size = max(cfg.INPUT.MIN_SIZE_TRAIN)
    assert stride > 0
    return (min_size - (min_size % stride), max_size - (max_size % stride))
