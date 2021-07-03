#!/usr/bin/env python

import torch

from timm.models.byoanet import default_cfgs, model_cfgs

from detectron2.modeling import BACKBONE_REGISTRY, ShapeSpec
from detectron2.config import CfgNode

from detectron2_timm.config import get_cfg, cfg_from_defaults
from detectron2_timm.models import utils
from detectron2_timm.models.backbone import Backbone


__all__ = []


def build_detectron2_backbone(cfg: CfgNode, input_shape: ShapeSpec) -> Backbone:
    func_name = cfg.MODEL.BACKBONE.NAME
    assert func_name in __all__, f'{func_name} not found'

    model_name = utils.get_model_name(func_name)
    attrs = {
        'img_size': (cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN),
        'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
    }
    model = utils.get_models(default_cfgs)[model_name](**attrs)
    backbone = Backbone(
        cfg,
        model=model,  # utils.get_models(default_cfgs)[model_name],
        model_config=utils.find_model_config(model_cfgs, model_name),
    )
    return backbone


def hook(local_s, model_name: str, **kwargs: dict) -> None:
    func_name = utils.get_func_name(model_name=model_name)
    local_s.update({func_name: build_detectron2_backbone})
    BACKBONE_REGISTRY._do_register(func_name, local_s[func_name])
    __all__.append(func_name)


def register(local_s):
    model_dict = utils.get_models(default_cfgs)
    for model_name in model_dict:
        hook(local_s, model_name=model_name)


register(locals())


if __name__ == '__main__':

    cfg = get_cfg()
    cfg = cfg_from_defaults(cfg, default_cfgs)

    cfg.MODEL.BACKBONE.NAME = "build_swinnet26t_256_backbone"

    from detectron2.modeling import build_model

    r = {'image': torch.randn((3, 1024, 1024))}
    t = build_model(cfg)

    t.eval()
    print(t)
    z = t([r])

    import IPython

    IPython.embed()
