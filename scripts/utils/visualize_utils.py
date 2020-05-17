#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import os
import os.path as osp
import sys
for path in sys.path:
    if 'opt/ros/' in path:
        print('sys.path.remove({})'.format(path))
        sys.path.remove(path)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchviz import make_dot

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from pytorch.BCNN import BCNN


def viz_in_feature(
        # data_path, idx=0, width=640, height=640, grid_range=60.,
        # draw_instance_pt=False, draw_heading_pt=False):
        data_path, width=672, height=672, grid_range=70.):

    # data_name = os.listdir(os.path.join(data_path, 'in_feature'))[idx]
    # in_feature = np.load(os.path.join(data_path, 'in_feature/', data_name))
    # in_feature = np.load(data_path)

    # non_empty_channel = 5
    # non_empty_channel = in_feature[..., non_empty_channel].astype(np.uint8) * 255

    nonempty = np.load(data_path)
    print(nonempty.shape)
    # cv2.imwrite("non_empty.png", non_empty)
    # return

    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    grid_ticks = np.arange(-grid_range, grid_range + grid_length, grid_length)

    grid_centers = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    # def fill_grid(i, j, color):
    #     grid_center = np.array([grid_centers[i], grid_centers[j]])
    #     fill_area = np.array([[(grid_center[0] - grid_length / 2),
    #                            (grid_center[0] + grid_length / 2),
    #                            (grid_center[0] + grid_length / 2),
    #                            (grid_center[0] - grid_length / 2)],
    #                           [(grid_center[1] + grid_length / 2),
    #                            (grid_center[1] + grid_length / 2),
    #                            (grid_center[1] - grid_length / 2),
    #                            (grid_center[1] - grid_length / 2)]])
    #     plt.fill(fill_area[0], fill_area[1], color=color, alpha=0.1)

    def fill_grid(i, j, color):
        grid_center = np.array([grid_centers[j], -grid_centers[i]])
        # grid_center = np.array([-grid_centers[i], grid_centers[j]])
        # grid_center = np.array([grid_centers[i], grid_centers[j]])
        fill_area = np.array([[(grid_center[0] - grid_length / 2),
                               (grid_center[0] + grid_length / 2),
                               (grid_center[0] + grid_length / 2),
                               (grid_center[0] - grid_length / 2)],
                              [(grid_center[1] + grid_length / 2),
                               (grid_center[1] + grid_length / 2),
                               (grid_center[1] - grid_length / 2),
                               (grid_center[1] - grid_length / 2)]])
        plt.fill(fill_area[0], fill_area[1], color=color, alpha=0.1)

    for i in range(height):
        for j in range(width):
            # if in_feature[i, j, non_empty_channel] == 1:
            # if in_feature[i, j, non_empty_channel] == 1:
            if nonempty[i, j, 0] == 1:
                fill_grid(i, j, 'r')

    # for i in range(width):
    #     for j in range(height):
    #         if in_feature[i, j, non_empty_channle] == 1:
    #             fill_grid(i, j, 'r')

    plt.show()


def viz_out_feature(
        # data_path, width=672, height=672, grid_range=70.,
        # draw_instance_pt=False, draw_heading_pt=False):
        nonempty, category_pt, instance_pt_x, instance_pt_y, width=672, height=672, grid_range=70.,
        draw_instance_pt=False, draw_heading_pt=False):

    # data_name = os.listdir(os.path.join(data_path, 'in_feature'))[idx]
    # in_feature = np.load(os.path.join(data_path, 'in_feature/', data_name))
    # out_feature = np.load(os.path.join(data_path, 'out_feature/', data_name))
    # out_feature = np.load(data_path)

    nonempty = np.load(nonempty)
    category_pt = np.load(category_pt)
    instance_pt_x = np.load(instance_pt_x)
    instance_pt_y = np.load(instance_pt_y)

    # print(out_feature.shape)

    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    grid_ticks = np.arange(-grid_range, grid_range + grid_length, grid_length)

    grid_centers = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    def fill_grid(i, j, color):
        grid_center = np.array([grid_centers[j], -grid_centers[i]])
        fill_area = np.array([[(grid_center[0] - grid_length / 2),
                               (grid_center[0] + grid_length / 2),
                               (grid_center[0] + grid_length / 2),
                               (grid_center[0] - grid_length / 2)],
                              [(grid_center[1] + grid_length / 2),
                               (grid_center[1] + grid_length / 2),
                               (grid_center[1] - grid_length / 2),
                               (grid_center[1] - grid_length / 2)]])

        plt.fill(fill_area[0], fill_area[1], color=color, alpha=0.1)

    instance_norms = []
    if draw_instance_pt:
        print('draw instance pt')
        for i in range(height):
            for j in range(width):
                # if in_feature[i, j, 5] == 1:
                if nonempty[i, j, 0] == 1:
                    fill_grid(i, j, 'r')
                # if out_feature[i, j, 0] == 1:
                if category_pt[i, j, 0] > 0.5:
                    instance_norms.append(
                        np.linalg.norm([instance_pt_y[i, j, 0], instance_pt_x[i, j, 0]]))
                    # print(np.linalg.norm([instance_pt_y[i, j, 0], instance_pt_x[i, j, 0]]))
                    fill_grid(i, j, 'b')
                    grid_center = np.array([grid_centers[j], -grid_centers[i]])
                    plt.arrow(x=grid_center[0],
                              y=grid_center[1],
                    #           # dx=out_feature[i, j, 2],
                    #           # dy=-out_feature[i, j, 1],
                              dx=instance_pt_y[i, j, 0],
                              dy=-instance_pt_x[i, j, 0],
                              width=0.01,
                              head_width=0.05,
                              head_length=0.05,
                              length_includes_head=True,
                              color='k')
        print('instance norm \nmean: {}\nmax: {}\nmin: {}'.format(np.mean(instance_norms),
                                                                  np.max(instance_norms),
                                                                  np.min(instance_norms)))
    # if draw_heading_pt:
    #     for i in range(height):
    #         for j in range(width):
    #             # if in_feature[i, j, 5] == 1:
    #             #     fill_grid(i, j, 'r')
    #             # if out_feature[i, j, 0] == 1:
    #             if category_pt[i, j, 0] == 1:
    #                 fill_grid(i, j, 'b')
    #                 grid_center = np.array([grid_centers[j], -grid_centers[i]])
    #                 heading_pt = out_feature[i, j, 6]
    #                 dx = -1.0 * np.cos(heading_pt)
    #                 dy = 1.0 * np.sin(heading_pt)
    #                 plt.arrow(x=grid_center[0],
    #                           y=grid_center[1],
    #                           dx=dx,
    #                           dy=-dy,
    #                           width=0.01,
    #                           head_width=0.05,
    #                           head_length=0.05,
    #                           length_includes_head=True,
    #                           color='k')

    plt.show()


def visualize_model():
    net = BCNN()
    bcnn_img = torch.rand(1, 8, 640, 640)
    output = net(bcnn_img)

    dot = make_dot(output, params=dict(net.named_parameters()))

    dot.render("bcnn")


if __name__ == '__main__':

    # viz_in_feature(
    #     data_path='data/non_empty.npy',
    #     width=672, height=672, grid_range=70)

    viz_out_feature(
        nonempty='data/non_empty.npy',
        category_pt='data/category_pt.npy',
        instance_pt_x='data/instance_pt_x.npy',
        instance_pt_y='data/instance_pt_y.npy',
        width=672, height=672, grid_range=70.,
        draw_instance_pt=True, draw_heading_pt=False)

    # visualize_model()
