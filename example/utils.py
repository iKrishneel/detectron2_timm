#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def visualize_features(data: np.array, show: bool = True):

    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 1), (0, 1)) + ((0, 0),) * (
        data.ndim - 3
    )
    data = np.pad(data, padding, mode='constant', constant_values=1)
    data = data.reshape((n, n) + data.shape[1:]).transpose(
        (0, 2, 1, 3) + tuple(range(4, data.ndim + 1))
    )
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    if show:
        plt.imshow(data)
        plt.axis('off')
        plt.show()
