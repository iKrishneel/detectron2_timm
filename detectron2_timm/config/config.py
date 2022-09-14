#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from .defaults import _C


def cfg_from_defaults(cfg, default_cfgs: dict):
    for name in default_cfgs:
        for key, item in default_cfgs[name].items():
            setattr(cfg.MODEL.BACKBONE, key.upper(), item)
        return cfg


def get_cfg() -> CN:
    return _C.clone()
