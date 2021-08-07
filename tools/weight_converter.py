#!/usr/bin/env python

import os
import os.path as osp
from collections import OrderedDict
import argparse

import numpy as np
import torch


def xcit_models(weights, model_name: str) -> OrderedDict:

    state_dict = weights['state_dict']
    new_state_dict = OrderedDict()

    for key in state_dict:
        new_key = None
        if 'backbone' in key:
            new_key = 'backbone.bottom_up.model.model.' + key[9:]
            new_key = new_key.replace('pos_embeder', 'pos_embed')

            blk_name = 'block4' if 'small' in model_name else 'block8'
            new_key = new_key.replace('model.fpn1', blk_name)

        elif 'neck.lateral_convs' in key:
            x, y = 0, 0
            for i, j in enumerate(np.arange(2, 6)):
                if str(i) in key:
                    x, y = i, j
                    break

            t = 'weight' if 'weight' in key else 'bias'
            new_key = key.replace(
                f'neck.lateral_convs.{x}.conv.{t}',
                f'backbone.fpn_lateral{j}.{t}',
            )

        elif 'neck.fpn_convs' in key:
            x, y = 0, 0
            for i, j in enumerate(np.arange(2, 6)):
                if str(i) in key:
                    x, y = i, j
                    break

            t = 'weight' if 'weight' in key else 'bias'
            new_key = key.replace(
                f'neck.fpn_convs.{x}.conv.{t}', f'backbone.fpn_output{j}.{t}'
            )
        elif 'rpn_head' in key:
            t = 'weight' if 'weight' in key else 'bias'
            if 'conv' in key:
                new_key = key.replace(
                    f'rpn_head.rpn_conv.{t}',
                    f'proposal_generator.rpn_head.conv.{t}',
                )
            elif 'rpn_cls' in key:
                new_key = key.replace(
                    f'rpn_head.rpn_cls.{t}',
                    f'proposal_generator.rpn_head.objectness_logits.{t}',
                )
            elif 'rpn_reg' in key:
                new_key = key.replace(
                    f'rpn_head.rpn_reg.{t}',
                    f'proposal_generator.rpn_head.anchor_deltas.{t}',
                )
        elif 'roi_head.bbox_head' in key:
            t = 'weight' if 'weight' in key else 'bias'
            if 'fc_cls' in key:
                new_key = key.replace(
                    f'roi_head.bbox_head.fc_cls.{t}',
                    f'roi_heads.box_predictor.cls_score.{t}',
                )
            elif 'fc_reg' in key:
                new_key = key.replace(
                    f'roi_head.bbox_head.fc_reg.{t}',
                    f'roi_heads.box_predictor.bbox_pred.{t}',
                )
            elif 'shared_fcs.0' in key:
                new_key = key.replace(
                    f'roi_head.bbox_head.shared_fcs.0.{t}',
                    f'roi_heads.box_head.fc1.{t}',
                )
            elif 'shared_fcs.1' in key:
                new_key = key.replace(
                    f'roi_head.bbox_head.shared_fcs.1.{t}',
                    f'roi_heads.box_head.fc2.{t}',
                )
        elif 'roi_head.mask_head' in key:
            t = 'weight' if 'weight' in key else 'bias'
            if 'convs' in key:
                for i, j in enumerate(np.arange(0, 4), 1):
                    if str(j) in key:
                        break
                new_key = key.replace(
                    f'roi_head.mask_head.convs.{j}.conv.{t}',
                    f'roi_heads.mask_head.mask_fcn{i}.{t}',
                )

                print(key, "       ", new_key, i, j)
            elif 'upsample' in key:
                new_key = key.replace(
                    f'roi_head.mask_head.upsample.{t}',
                    f'roi_heads.mask_head.deconv.{t}',
                )
            elif 'conv_logits' in key:
                new_key = key.replace(
                    f'roi_head.mask_head.conv_logits.{t}',
                    f'roi_heads.mask_head.predictor.{t}',
                )

        if not new_key:
            print(f'Skipping... {key}')
            continue

        new_state_dict[new_key] = state_dict[key]
        # print(new_state_dict[new_key].dtype, state_dict[key].dtype, new_key)

    return new_state_dict


def main(args):
    name = args.name.lower()
    weights = args.weights
    log_dir = args.output_dir

    assert osp.isfile(weights)
    if not osp.isdir(log_dir):
        os.mkdir(log_dir)

    weights = torch.load(weights)
    if 'xcit' in name:
        new_state_dict = xcit_models(weights, args.weight)
    else:
        raise ValueError('Unknown model')

    filename = osp.join(log_dir, name + '.pth')
    torch.save(new_state_dict, filename)

    print(f"Done and saved to: {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument(
        '--output_dir', type=str, required=False, default='./logs'
    )

    args = parser.parse_args()
    main(args=args)
