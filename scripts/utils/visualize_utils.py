#!/usr/bin/env python3
# coding: utf-8

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchviz import make_dot

from BCNN import BCNN


def viz_out_feature(
        data_path, idx=0, width=640, height=640, grid_range=60.):

    data_name = os.listdir(os.path.join(data_path, 'in_feature'))[idx]
    out_feature = np.load(os.path.join(data_path, 'out_feature/', data_name))

    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    grid_ticks = np.arange(-grid_range, grid_range + grid_length, grid_length)

    grid_centers = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    def fill_grid(i, j, color):
        grid_center = np.array([grid_centers[i], grid_centers[j]])
        fill_area = np.array([[(grid_center[0] - grid_length / 2),
                               (grid_center[0] + grid_length / 2),
                               (grid_center[0] + grid_length / 2),
                               (grid_center[0] - grid_length / 2)],
                              [(grid_center[1] + grid_length / 2),
                               (grid_center[1] + grid_length / 2),
                               (grid_center[1] - grid_length / 2),
                               (grid_center[1] - grid_length / 2)]])

        plt.fill(fill_area[0], fill_area[1], color=color, alpha=0.1)

    for i in range(width):
        for j in range(height):
            if out_feature[i, j, 0] == 1:
                fill_grid(i, j, 'r')
                grid_center = np.array([grid_centers[i], grid_centers[j]])
                plt.arrow(x=grid_center[0],
                          y=grid_center[1],
                          dx=out_feature[i, j, 1],
                          dy=out_feature[i, j, 2],
                          width=0.01,
                          head_width=0.05,
                          head_length=0.05,
                          length_includes_head=True,
                          color='k')
    plt.show()


def visualize_model():
    net = BCNN()
    bcnn_img = torch.rand(1, 8, 640, 640)
    output = net(bcnn_img)

    dot = make_dot(output, params=dict(net.named_parameters()))

    dot.render("bcnn")


if __name__ == '__main__':
    viz_out_feature(
        data_path='/media/kosuke/HD-PNFU3/nusc/feature_instance_one',
        idx=0, width=640, height=640, grid_range=60.)

    visualize_model()
