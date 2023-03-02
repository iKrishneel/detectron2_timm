#!/usr/bin/env python

from .crossformer import CrossFormer  # NOQA: F401


__all__ = ['cross_former_s', 'cross_former_b']


def cross_former_s(**kwargs):
    return CrossFormer(
        img_size=[
            1280,
            800,
        ],  # This is only used to compute the FLOPs under the give image size
        patch_size=[4, 8, 16, 32],
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=kwargs.get('group_size', [7, 7, 7, 7]),
        crs_interval=kwargs.get('crs_interval', [8, 4, 2, 1]),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2, 4], [2, 4]],
    )


def cross_former_b(**kwargs):
    return CrossFormer(
        img_size=[1280, 800],
        patch_size=[4, 8, 16, 32],
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        group_size=kwargs.get('group_size', [7, 7, 7, 7]),
        crs_interval=kwargs.get('crs_interval', [8, 4, 2, 1]),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2, 4], [2, 4]],
    )
