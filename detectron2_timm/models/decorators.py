#!/usr/bin/env python

import functools

from .utils import remove_named_children


__all__ = ['remove_layers']


def remove_layers(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cfg = args[0]
        model = kwargs.pop('model')
        assert model is not None

        assert model.feature_info

        feature_info = [
            info
            for info in model.feature_info
            if info['module'] not in cfg.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS
        ]

        model = remove_named_children(cfg, model)
        func(self, cfg, model=model, feature_info=feature_info, **kwargs)

    return wrapper
