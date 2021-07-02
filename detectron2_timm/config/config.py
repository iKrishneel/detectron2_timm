#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C as _c


# configurations
_C = _c.clone()
_C.MODEL.BACKBONE.NAME = "build_botnet26t_256_backbone"
_C.MODEL.BACKBONE.FREEZE_AT = 0

# backbone params
# _C.MODEL.BACKBONE.POOL_SIZE = [8, 8]
# _C.MODEL.BACKBONE.OUT_FEATURES = ["stage4"]
# _C.MODEL.BACKBONE.REMOVE_LAYERS = ["final_conv", "head"]

_C.MODEL.BACKBONE.CONFIG = CN()
_C.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS = ["head"]
_C.MODEL.BACKBONE.CONFIG.REMAPS = ["stage4"]
_C.MODEL.BACKBONE.CONFIG.OUT_FEATURES = ["final_conv"]
_C.MODEL.BACKBONE.CONFIG.STRIDES = [32]

# rpn
_C.MODEL.RPN.IN_FEATURES = ["stage4"]

# roi head
_C.MODEL.ROI_HEADS.IN_FEATURES = ["stage4"]
_C.MODEL.ROI_HEADS.NAME = "StandardROIHeads"

_C.MODEL.ROI_BOX_HEAD.FC_DIM = 2048
_C.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 3

# input
_C.INPUT.MAX_SIZE_TRAIN = 1024
_C.INPUT.FIXED_INPUT_SIZE = True


def cfg_from_defaults(cfg, default_cfgs: dict):
    for name in default_cfgs:
        attr = name.upper()
        for key, item in default_cfgs[name].items():
            setattr(cfg.MODEL.BACKBONE, key.upper(), item)
        return cfg


def get_cfg() -> CN:
    return _C.clone()

