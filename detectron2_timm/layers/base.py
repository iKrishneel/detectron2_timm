#!/usr/bin/env python

import torch.nn as nn

from detectron2.utils.registry import Registry as _Registry

LAYER_REGISTRY = _Registry('LAYERS')
LAYER_REGISTRY.__doc__ = """Registry for custom layers"""


class LayerBase(nn.Module):

    def __init__(self, *args, **kwargs):
        super(LayerBase, self).__init__()
