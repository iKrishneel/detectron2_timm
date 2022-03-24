#!/usr/bin/env python

import os.path as osp
import argparse
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2_timm.config import get_cfg

plt.axis('off')


def test_one(cfg, image, threshold=0.5):
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    visualizer = Visualizer(image, metadata=metadata, scale=1.0)

    r = predictor(image)
    instances = r['instances'].to('cpu').get_fields()

    print(instances)
    if len(instances) == 0:
        print("Nothing detected!")
        return

    scores = instances['scores'].numpy()
    indices = np.where(scores >= threshold)
    scores = scores[indices]

    bboxes = instances['pred_boxes'].tensor.numpy()[indices]
    labels = instances['pred_classes'].numpy()[indices]
    try:
        masks = instances['pred_masks'].numpy()[indices]
    except KeyError:
        masks = None

    labels = np.array(metadata.get('thing_classes'))[labels]
    viz = visualizer.overlay_instances(labels=labels, boxes=bboxes, masks=masks)

    plt.imshow(viz.get_image())
    plt.show()


def main(args):
    assert osp.isfile(args.config_file)
    assert osp.isfile(args.weights)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = args.output_dir

    image = cv.imread(args.image, cv.IMREAD_ANYCOLOR)
    test_one(cfg, image, args.threshold)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--output_dir', default='../logs/', type=str)

    args = parser.parse_args()

    main(args)
