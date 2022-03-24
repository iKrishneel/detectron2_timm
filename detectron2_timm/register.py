#!/usr/bin/env python

import inspect
import functools
from typing import List

from timm.models import list_models, list_modules

from detectron2.modeling import BACKBONE_REGISTRY

from detectron2_timm import models
from .models import utils


__all__ = []


def register_backbone(build_funcs):
    def wrapper(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            for build_func in build_funcs():
                func_name = func(build_func=build_func, **kwargs)
                BACKBONE_REGISTRY._do_register(func_name, build_func)
                __all__.append(func_name)
            return func

        return inner

    return wrapper


def build_backbone_functions() -> List:
    return [
        func
        for name, func in inspect.getmembers(models, inspect.isfunction)
        if 'build' in name.split('_')[0] and 'backbone' in name.split('_')[-1]
    ]


@register_backbone(build_backbone_functions)
def hook(model_name: str, local_s=None, **kwargs: dict) -> None:
    build_func = kwargs.get('build_func', None)
    assert build_func and callable(build_func)
    assert model_name in list_models() or model_name in models.list_models()

    func_name = utils.get_func_name(build_func, model_name)
    if local_s is not None:
        local_s.update({func_name: build_func})
    return func_name


def register(local_s, module) -> None:

    if callable(module):
        model_dict = {module.__name__: module}
    else:
        default_cfgs = getattr(module, 'default_cfgs')
        model_dict = utils.get_models(default_cfgs)

    for model_name in model_dict:
        hook(local_s=local_s, model_name=model_name)


def register_all(local_s) -> None:
    for module_name in list_modules():
        module = utils.get_attr(module_name)
        register(local_s, module)
