#! /usr/bin/env python
from itertools import product
import numpy as np

optim_dict = {
    'fine_tune_layers':
    [
        ['fc6', 'fc7', 'fc8'],
        ['conv5_3', 'fc6', 'fc7', 'fc8'],
        ['conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8'],
        ['conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2',
            'conv5_3', 'fc6', 'fc7', 'fc8'],
    ],
    'new_lr': [[1e-5]],
    'data_augmentations':
    [
        ['random_crop', 'left_right'],
        ['random_crop', 'left_right', 'random_contrast', 'random_brightness'],
    ]

}


def create_combos(random_search=False, keep_prop=.5):
    # Get all combinations of optim_dict
    flat = [[(k, v) for v in vs] for k, vs in optim_dict.items()]
    params = [dict(items) for items in product(*flat)]
    if random_search:
        params = np.random.permutation(np.asarray(params))[int(
            len(params) * keep_prop)]
    return params


if __name__ == '__main__':
    print create_combos()
