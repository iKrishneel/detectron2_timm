#!/usr/bin/env python

import os
from dataclasses import dataclass

import torch

from detectron2.modeling import BACKBONE_REGISTRY  # NOQA
from detectron2.config import CfgNode
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator

from detectron2_timm.config import get_cfg
from detectron2_timm.models import utils

# from utils import visualize_features


@dataclass
class Trainer(DefaultTrainer):

    cfg: CfgNode = None

    def __post_init__(self):
        assert self.cfg
        super(Trainer, self).__init__(self.cfg)

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


def debug(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    name = utils.get_model_name(cfg.MODEL.BACKBONE.NAME)
    print("Model Name: ", name)

    use_d = True
    s = [3, 224, 224]
    if use_d:
        from detectron2.modeling import build_model

        m = build_model(cfg)
        x = [{'image': torch.randn(s)}]
    else:
        m = locals()[cfg.MODEL.BACKBONE.NAME](cfg, 1)
        x = torch.randn((1, *s))

    m = m.to('cpu')
    m.eval()
    with torch.no_grad():
        z = m(x)

    import IPython

    IPython.embed()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=False, type=str, default=None)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        debug(args)
    else:
        train = True
        if train:
            launch(main, args.num_gpus, args=(args,), dist_url='auto')
        else:
            main(args)
