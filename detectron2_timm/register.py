#!/usr/bin/env python

import timm.models as tmodels
from timm.models import list_models, list_modules

from detectron2.modeling import BACKBONE_REGISTRY

from .models import (
    build_detectron2_backbone, build_detectron2_fpn_backbone
)
from .models import utils


__all__ = []


def hook(build_func, model_name: str, local_s=None, **kwargs: dict) -> None:
    assert build_func and callable(build_func)
    assert model_name in list_models()

    func_name = utils.get_func_name(build_func, model_name)
    if local_s is not None:
        local_s.update({func_name: build_func})
    BACKBONE_REGISTRY._do_register(func_name, build_func)
    __all__.append(func_name)


def register(build_func, local_s, module) -> None:

    if callable(module):
        model_dict = {module.__name__: module}
    else:
        default_cfgs = getattr(module, 'default_cfgs')
        model_dict = utils.get_models(default_cfgs)

    for model_name in model_dict:
        hook(build_func, local_s=local_s, model_name=model_name)


def register_all(local_s) -> None:
    not_found = []
    for module_name in list_modules():
        module = utils.get_attr(module_name)
        register(build_detectron2_backbone, local_s, module)
        register(build_detectron2_fpn_backbone, local_s, module)


# register_all(locals())
