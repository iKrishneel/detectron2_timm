#!/usr/bin/env python

import argparse

import numpy as np
import torch

from detectron2.modeling import ShapeSpec
import detectron2_timm
from detectron2_timm.config import get_cfg


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', required=True, type=str)
args = parser.parse_args()


# c = '../config/regnetz/regnetz_e8.yaml'

cfg = get_cfg()
cfg.merge_from_file(args.config_file)

cfg.MODEL.DEVICE = 'cpu'


shape = ShapeSpec(3, 224, 224)
# model = build_regnetz_e8_fpn_backbone(cfg, shape)
model = getattr(detectron2_timm, cfg.MODEL.BACKBONE.NAME)(cfg, shape)


def trainable(model, n=1E6):
    t = 0
    for p in model.parameters():
        if p.requires_grad:
            t += np.prod(p.size())

    print(t / n)


trainable(model.bottom_up)


import IPython
IPython.embed()
