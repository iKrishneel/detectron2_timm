#!/usr/bin/env python

from timm.models import register_model

from .crossformer import CrossFormer  # NOQA: F401


__all__ = [
    'cross_former_s',
    'cross_former_b'
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 800, 1280),
        'fixed_input_size': True,
        'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5],
        **kwargs
    }


default_cfgs = {
    'cross_former_s': _cfg(),
    'cross_former_b': _cfg()
}


def _create_cross_fromer(**kwargs: dict):
    default_args = dict(
        img_size=[1280, 800],
        patch_size=[4, 8, 16, 32],
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[7, 7, 7, 7],
        crs_interval=[8, 4, 2, 1],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2, 4], [2, 4]]
    )
    default_args.update(**kwargs)
    return CrossFormer(**default_args)


@register_model
def cross_former_s(**kwargs: dict):
    model_kwargs = dict(
        img_size=kwargs.get('img_size', [1280, 800]),
        group_size=kwargs.get('group_size', [7, 7, 7, 7]),
        crs_interval=kwargs.get('crs_interval', [8, 4, 2, 1]),
        **kwargs
    )
    return _create_cross_fromer(**model_kwargs)


@register_model
def cross_former_b(**kwargs):
    model_kwargs = dict(
        img_size=kwargs.get('img_size', [1280, 800]),
        group_size=kwargs.get('group_size', [7, 7, 7, 7]),
        crs_interval=kwargs.get('crs_interval', [8, 4, 2, 1]),
        **kwargs
    )
    return _create_cross_fromer(**model_kwargs)
