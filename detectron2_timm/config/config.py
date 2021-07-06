#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C as _c


# configurations
_C = _c.clone()
_C.MODEL.BACKBONE.NAME = "build_botnet26t_256_backbone"
_C.MODEL.BACKBONE.FREEZE_AT = 0

_C.MODEL.BACKBONE.CONFIG = CN()
_C.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS = []
_C.MODEL.BACKBONE.CONFIG.REMAPS = []
_C.MODEL.BACKBONE.CONFIG.OUT_FEATURES = []
_C.MODEL.BACKBONE.CONFIG.STRIDES = []
_C.MODEL.BACKBONE.CONFIG.PRETRAINED = False  # init with pretrained model

_C.INPUT.FIXED_INPUT_SIZE = True


def cfg_from_defaults(cfg, default_cfgs: dict):
    for name in default_cfgs:
        for key, item in default_cfgs[name].items():
            setattr(cfg.MODEL.BACKBONE, key.upper(), item)
        return cfg


def get_cfg() -> CN:
    return _C.clone()
