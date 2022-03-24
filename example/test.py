# /usr/bin/env python

import numpy as np
import cv2 as cv
from re import search

import torch
from einops import rearrange

from detectron2.modeling import build_model
from detectron2_timm.config import get_cfg
from detectron2_timm import build_resnet50_fpn_backbone

from utils import load_state_dict, visualize_features


features = []


def forward(m, i, o):
    global features
    features.append(o.detach().cpu().numpy())


def test(p, wpath=None):
    from timm.models import resnet50

    m = resnet50(p)

    if wpath is not None:
        pth = torch.load(wpath)
        m.load_state_dict(pth)

    handle = m.layer2.register_forward_hook(forward)

    return m


def main(c, im_path, w_path):

    cfg = get_cfg()
    cfg.merge_from_file(c)

    cfg.MODEL.BACKBONE.CONFIG.PRETRAINED = False

    # model = test(False, w_path)
    model = build_model(cfg)
    # pth = torch.load(w_path)
    # s = model.load_state_dict(pth, strict=False)

    """
    for k in pth:
        if search('fc', k):
            print(k)
            continue
        key = 'backbone.bottom_up.model.' + k
        w = pth[k].to('cpu')
        v = model.state_dict()[key].to('cpu')
        if not torch.all(v == w):
            print(k, key, w.shape, v.shape)
    """

    # load_state_dict(model, p, strict=False, remap=['', 'bottom_up.model'])

    im = cv.imread(im_path)
    # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    # im = cv.resize(im, (640, 640))

    s = np.array([57.375, 57.120, 58.395], dtype=np.float32)
    m = np.array([103.530, 116.280, 123.675], dtype=np.float32)

    im = im.astype(np.float32)
    im = (im - m) / s
    # im = (im - np.min(im) / (np.max(im) - np.min(im)))

    img = rearrange(torch.from_numpy(im), 'h w c -> c h w').to('cuda')

    # z = model.backbone(img.unsqueeze(0))
    # y = model.backbone.bottom_up.feature_maps[0]
    # y = y.detach().cpu().numpy()

    # m = test(False)
    # z = model(img.unsqueeze(0).to('cpu'))
    # y = features[0]
    # visualize_features(y[0])

    IPython.embed()


if __name__ == '__main__':
    import sys, IPython

    cfg = sys.argv[1]
    im_path = sys.argv[2]
    w_path = sys.argv[3]
    main(cfg, im_path, w_path)
