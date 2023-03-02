#!/usr/bin/env python

import os.path as osp
from collections import OrderedDict
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def visualize_features(data: np.array, show: bool = True) -> None:

    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if show:
        plt.imshow(data, cmap='jet')
        plt.axis('off')
        plt.show()


def load_state_dict(
    model: nn.Module,
    path: str,
    key: str = None,
    strict: bool = True,
    device=torch.device('cpu'),
    remap: List[str] = None,
) -> nn.Module:
    assert osp.isfile(path)
    state_dict = torch.load(path, map_location=device)
    if key is not None:
        assert isinstance(key, str)
        state_dict = state_dict[key]
    if remap is not None:
        assert len(remap) == 2
        old = len(remap[0])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = remap[1] + k[old:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=strict)
    return model
