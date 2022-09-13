#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C as _c


# configurations
_C = _c.clone()
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.FREEZE_AT = 0

_C.MODEL.BACKBONE.CONFIG = CN()
_C.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS = []
_C.MODEL.BACKBONE.CONFIG.REMAPS = []
_C.MODEL.BACKBONE.CONFIG.OUT_FEATURES = []
_C.MODEL.BACKBONE.CONFIG.STRIDES = []
_C.MODEL.BACKBONE.CONFIG.PRETRAINED = False  # init with pretrained model

_C.MODEL.BACKBONE.CONFIG.CHANNELS = []
_C.MODEL.BACKBONE.CONFIG.REPLACE_LAYERS = [[]]

# for freezing certain layers with name
_C.MODEL.BACKBONE.CONFIG.FREEZE_LAYERS = []

_C.INPUT.FIXED_INPUT_SIZE = True
