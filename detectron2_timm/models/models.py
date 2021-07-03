#!/usr/bin/env python

import importlib

import torch
import torch.nn as nn

import timm.models as tmodels
from timm.models import list_models, list_modules
from detectron2.config import CfgNode
from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec

from detectron2_timm.models import utils
from detectron2_timm.models.backbone import Backbone


__all__ = []


def get_attr(name: str):
    return getattr(tmodels, name)


def get_model_attrs(cfg: CfgNode) -> dict:
    attrs = {'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES}
    if cfg.INPUT.FIXED_INPUT_SIZE:
        attrs.update(
            {
                'img_size': (
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                ),
            }
        )

    return attrs


def build_detectron2_backbone(cfg: CfgNode, input_shape: ShapeSpec) -> Backbone:
    func_name = cfg.MODEL.BACKBONE.NAME
    assert func_name in __all__, f'{func_name} not found'
    model_name = utils.get_model_name(func_name)
    assert model_name

    model = get_attr(name=model_name)
    assert model and callable(model)

    backbone = Backbone(
        cfg,
        model=model(**get_model_attrs(cfg)),
    )
    return backbone


def hook(model_name: str, local_s=None, **kwargs: dict) -> None:
    assert model_name in list_models()

    func_name = utils.get_func_name(model_name=model_name)
    if local_s is not None:
        local_s.update({func_name: build_detectron2_backbone})
        func = local_s[func_name]
    else:
        func = build_detectron2_backbone

    BACKBONE_REGISTRY._do_register(func_name, func)
    __all__.append(func_name)


def register(local_s, module) -> None:

    if callable(module):
        model_dict = {module.__name__: module}
    else:
        default_cfgs = getattr(module, 'default_cfgs')
        model_dict = utils.get_models(default_cfgs)

    for model_name in model_dict:
        hook(local_s=local_s, model_name=model_name)


def register_all(local_s) -> None:
    not_found = []
    for module_name in list_modules():
        module = get_attr(module_name)
        register(local_s, module)


# register_all(locals())


if __name__ == '__main__':

    import IPython
    from detectron2_timm.config import get_cfg, cfg_from_defaults

    cfg = get_cfg()
    # cfg.merge_from_file('../config/backbone/botnet_26t_256.yaml')
    cfg.merge_from_file('../config/backbone/resnet50.yaml')

    print(BACKBONE_REGISTRY)

    name = utils.get_model_name(cfg.MODEL.BACKBONE.NAME)
    hook(name, locals())

    use_d = True
    s = [3, 224, 224]
    if use_d:
        from detectron2.modeling import build_model

        m = build_model(cfg)
        r = [{'image': torch.randn(s)}]
    else:
        # m = build_botnet26t_256_backbone(cfg, 1)
        m = locals()[cfg.MODEL.BACKBONE.NAME](cfg, 1)
        r = torch.randn((1, *s))

    m.eval()
    with torch.no_grad():
        z = m(r)

    IPython.embed()
