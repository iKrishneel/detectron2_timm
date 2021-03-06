#!/usr/bin/env python

from detectron2.config import CfgNode
from detectron2.modeling import FPN, ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from detectron2_timm.models import utils
from detectron2_timm.models.backbone import Backbone


__all__ = ['build_detectron2_backbone', 'build_detectron2_fpn_backbone']


def get_model_attrs(cfg: CfgNode) -> dict:
    attrs = {
        'num_classes': cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        'pretrained': cfg.MODEL.BACKBONE.CONFIG.PRETRAINED,
    }
    if cfg.INPUT.FIXED_INPUT_SIZE:
        attrs.update(
            {
                'img_size': utils.patch_size_mod2(cfg=cfg),
            }
        )
    return attrs


def build_detectron2_backbone(cfg: CfgNode, input_shape: ShapeSpec) -> Backbone:
    func_name = cfg.MODEL.BACKBONE.NAME
    # assert func_name in __all__, f'{func_name} not found'
    model_name = utils.get_model_name(func_name)
    assert model_name

    model = utils.get_attr(name=model_name)
    assert model and callable(model)

    backbone = Backbone(
        cfg,
        model=model(**get_model_attrs(cfg)),
    )
    return backbone


def build_detectron2_fpn_backbone(cfg: CfgNode, input_shape: ShapeSpec) -> FPN:
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    norm = cfg.MODEL.FPN.NORM
    fuse_type = cfg.MODEL.FPN.FUSE_TYPE

    bottom_up = build_detectron2_backbone(cfg, input_shape)
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=norm,
        top_block=LastLevelMaxPool(),
        fuse_type=fuse_type,
    )
    return backbone
