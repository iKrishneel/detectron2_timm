#!/usr/bin/env python

import os
from dataclasses import dataclass

from detectron2.modeling import BACKBONE_REGISTRY  # NOQA
from detectron2.config import CfgNode
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.data import MetadataCatalog, DatasetMapper
from detectron2.data.build import build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator

from detectron2_timm.config import get_cfg


@dataclass
class Trainer(DefaultTrainer):

    cfg: CfgNode = None

    def __post_init__(self):
        assert self.cfg
        super(Trainer, self).__init__(self.cfg)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode, mapper=None):
        mapper = DatasetMapper(
            cfg=cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                # T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
                T.RandomFlip(),
                T.Resize([cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN]),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name=dataset_name)
        else:
            raise ValueError('Evaluator type is unknown!')
        return evaluator


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.BACKBONE.NAME = 'build_botnet26t_256_fpn_backbone'
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.BACKBONE.CONFIG.PRETRAINED = False
    cfg.MODEL.BACKBONE.CONFIG.OUT_FEATURES = [
        'stages.0',
        'stages.1',
        'stages.2',
        'stages.3',
    ]
    cfg.MODEL.BACKBONE.CONFIG.STRIDES = [4, 8, 16, 32]
    cfg.MODEL.BACKBONE.CONFIG.REMAPS = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.BACKBONE.CONFIG.REMOVE_LAYERS = ['final_conv', 'head']

    cfg.INPUT.FIXED_INPUT_SIZE = True
    cfg.INPUT.MAX_SIZE_TRAIN = 1024

    try:
        cfg.OUTPUT_DIR = args.output_dir
        cfg.MODEL.WEIGHTS = args.weights
    except AttributeError as e:
        print(e)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args=None):

    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=False, type=str, default=None)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    parser.add_argument('--train', action='store_true', default=True)
    args = parser.parse_args()

    if args.train:
        launch(main, args.num_gpus, args=(args,), dist_url='auto')
    else:
        main(args)
