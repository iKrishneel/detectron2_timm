#!/usr/bin/env python

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2_timm.config import get_cfg


def introspect(features, image, save_dir: str = None):
    size = np.array(image.shape[:2][::-1]) // 1
    # image = cv.resize(image, size)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i, feat in enumerate(features):
        print(feat.shape)
        feat = cv.resize(feat, tuple(size))
        feat = (255 * (feat - feat.min()) / (feat.max() - feat.min())).astype(np.uint8)
        feat = cv.applyColorMap(feat, cv.COLORMAP_JET)
        # im = cv.addWeighted(image, 0.5, feat, 0.5, 0.0)
        im = feat

        if save_dir is not None:
            cv.imwrite(osp.join(save_dir, str(i).zfill(6) + '.jpg'), im)
        else:
            cv.namedWindow('feat', cv.WINDOW_NORMAL)
            cv.imshow('feat', im)
            if cv.waitKey(0) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break


def test_one(image, predictor, metadata, threshold=0.5):
    visualizer = Visualizer(image, metadata=metadata, scale=1.0)
    
    r = predictor(image)
    instances = r['instances'].to('cpu').get_fields()
   
    scores = instances['scores'].numpy()
    remove_indices = scores < threshold
    
    scores = np.delete(scores, remove_indices, 0)
    bboxes = np.delete(instances['pred_boxes'].tensor.numpy(), remove_indices, 0)
    labels = np.delete(instances['pred_classes'].numpy(), remove_indices, 0)
    
    try:
        masks = np.delete(instances['pred_masks'].numpy(), remove_indices, 0)
    except KeyError:
        masks = None

    labels = np.array(metadata.get('thing_classes'))[labels]
    viz = visualizer.overlay_instances(labels=labels, boxes=bboxes, masks=masks)
    
    return viz.get_image()


def forward(attn_obj):
    def _forward(x, mask=None):
        B_, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B_, N, 3, attn_obj.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * attn_obj.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + attn_obj._get_rel_pos_bias()

        mask = None
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, attn_obj.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, attn_obj.num_heads, N, N)
            attn = attn_obj.softmax(attn)
        else:
            attn = attn_obj.softmax(attn)

        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return _forward


feature_maps = {}
def forward_hook(m, inp, out):
    name = 'fmap'
    feature_maps[name] = out.cpu().numpy()  # .transpose(1, 0)


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    predictor.model.backbone.bottom_up.model.model.layers[-1].blocks[-1].attn.forward = forward(
        predictor.model.backbone.bottom_up.model.model.layers[-1].blocks[-1].attn,
    )
    
    # handler = predictor.model.backbone.bottom_up.model.model.layers[3].blocks[1].attn.register_forward_hook(forward_hook)

    image = cv.imread(args.image)
    im_viz = test_one(image, predictor, metadata)

    # introspect(feature_maps['fmap'], image)

    attn_map = predictor.model.backbone.bottom_up.model.model.layers[-1].blocks[-1].attn.attn_map.mean(dim=1)
    attn_cls = predictor.model.backbone.bottom_up.model.model.layers[-1].blocks[-1].attn.cls_attn_map.mean(dim=1)
    attn_cls = cv.resize(attn_cls.cpu().numpy(), image.shape[:2][::-1])

    plt.imshow(image)
    plt.imshow(attn_cls, cmap='jet', alpha=0.5)
    plt.show()

    import IPython; IPython.embed()
    
    # cv.imshow('image', image)
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     cv.destroyAllWindows()
    

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)

    args = parser.parse_args()

    main(args)
