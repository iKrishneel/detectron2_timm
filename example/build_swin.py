#!/usr/bin/env python

# import torch
from detectron2.modeling import BACKBONE_REGISTRY  # NOQA

from detectron2_timm.config import get_cfg
from detectron2_timm.models import utils

from detectron2_timm.register import register, hook  # NOQA


if __name__ == '__main__':
    cfg = get_cfg()
    cfg.merge_from_file(
        '../detectron2_timm/config/backbone/swin_base_patch4_window12.yaml'
    )

    name = utils.get_model_name(cfg.MODEL.BACKBONE.NAME)
    print("Model Name: ", name)

    m = BACKBONE_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)(cfg, 1)

    """
    use_d = True
    s = [3, 224, 224]
    if use_d:
        from detectron2.modeling import build_model

        m = build_model(cfg)
        r = [{'image': torch.randn(s)}]
    else:
        m = locals()[cfg.MODEL.BACKBONE.NAME](cfg, 1)
        r = torch.randn((1, *s))

    m.eval()
    with torch.no_grad():
        z = m(r)
    """

    import IPython

    IPython.embed()
