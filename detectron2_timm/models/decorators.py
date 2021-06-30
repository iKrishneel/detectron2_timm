#!/usr/bin/env python

import functools

from .utils import remove_named_children


__all__ = [
    'remove_layers'
]


def remove_layers(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cfg = args[0]
        model = kwargs.pop('model')

        attrs = {
            'img_size': (cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN),
            'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES
        }        
        model = model(**attrs)
        model = remove_named_children(cfg, model)
        func(self, cfg, model=model, **kwargs)
    return wrapper
