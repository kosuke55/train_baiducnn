#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import
from __future__ import division

import os
import os.path as osp
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchviz import make_dot

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from pytorch.BCNN import BCNN  # noqa

try:
    import cv2
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            import cv2
            sys.path.append(path)
            break


def viz_feature(
        in_data, out_data, width=672, height=672, grid_range=70.,
        draw_target='instance', use_cnpy_feature=True):

    in_feature = np.load(in_data)
    out_feature = np.load(out_data)

    if use_cnpy_feature:
        in_feature = in_feature.transpose(1, 2, 0)
        out_feature = out_feature.transpose(1, 2, 0)

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
    for i in range(height):
        for j in range(width):
            if in_feature[i, j, 5] == 1:
                fill_grid(i, j, 'r')
            if out_feature[i, j, 0] > 0.5:
                instance_norms.append(
                    np.linalg.norm([out_feature[i, j, 2], out_feature[i, j, 1]]))
                fill_grid(i, j, 'b')
                grid_center = np.array([grid_centers[j], -grid_centers[i]])
                if draw_target == 'instance':
                    dx = out_feature[i, j, 2]
                    dy = out_feature[i, j, 1]
                elif draw_target == 'heading':
                    if use_cnpy_feature:
                        yaw = math.atan2(out_feature[i, j, 10],
                                         out_feature[i, j, 9]) * 0.5
                    else:
                        yaw = math.atan2(out_feature[i, j, 6],
                                         out_feature[i, j, 5]) * 0.5
                    dx = math.sin(yaw)
                    dy = math.cos(yaw)

                plt.arrow(x=grid_center[0],
                          y=grid_center[1],
                          dx=dx,
                          dy=-dy,
                          width=0.01,
                          head_width=0.05,
                          head_length=0.05,
                          length_includes_head=True,
                          color='k')
    print('instance norm \nmean: {}\nmax: {}\nmin: {}'.format(np.mean(instance_norms),
                                                              np.max(
                                                                  instance_norms),
                                                              np.min(instance_norms)))

    plt.savefig("viz_feature_low.png", format="png")
    plt.savefig("viz_feature.png", format="png", dpi=3000)
    plt.show()


def viz_inference_feature(
        # data_path, width=672, height=672, grid_range=70.,
        # draw_instance_pt=False, draw_heading_pt=False):
        in_data, inference_data, width=672, height=672, grid_range=70.,
        draw_instance_pt=False, draw_heading_pt=False):

    in_feature = np.load(in_data)
    inference_feature = np.load(inference_data)

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
                if in_feature[i, j, 5] == 1:
                    fill_grid(i, j, 'r')
                # if inference_feature[0, 3, i, j] > 0.1:
                # if inference_feature[0, 0, i, j] > 0.3:
                if inference_feature[0, 0, i, j] > 0.3:
                    instance_norms.append(
                        np.linalg.norm([inference_feature[0, 2, i, j], inference_feature[0, 1, i, j]]))
                    fill_grid(i, j, 'b')
                    grid_center = np.array([grid_centers[j], -grid_centers[i]])

                    yaw = math.atan2(inference_feature[0, 6, i, j],
                                     inference_feature[0, 5, i, j]) * 0.5

                    # instance
                    plt.arrow(x=grid_center[0],
                              y=grid_center[1],
                              #   dx=inference_feature[0, 2, i, j],
                              #   dy=-inference_feature[0, 1, i, j],
                              dx=math.sin(yaw),
                              dy=-math.cos(yaw),
                              width=0.01,
                              head_width=0.05,
                              head_length=0.05,
                              length_includes_head=True,
                              color='k')

        print('instance norm \nmean: {}\nmax: {}\nmin: {}'.format(np.mean(instance_norms),
                                                                  np.max(
                                                                      instance_norms),
                                                                  np.min(instance_norms)))
    plt.show()


def visualize_model():
    net = BCNN()
    bcnn_img = torch.rand(1, 8, 640, 640)
    output = net(bcnn_img)

    dot = make_dot(output, params=dict(net.named_parameters()))

    dot.render("bcnn")


if __name__ == '__main__':

    # viz_inference_feature(
    #     in_data='/media/yukihiro/3594a1e3-a5ed-4fcf-a386-9d98730f5989/v1.0-mini_dataset/mini-6c-672_test/in_feature/00006.npy',
    #     inference_data='/media/yukihiro/3594a1e3-a5ed-4fcf-a386-9d98730f5989/v1.0-mini_dataset/mini-6c-672_test/inference_feature/00006.npy',
    #     width=672, height=672, grid_range=70.,
    #     draw_instance_pt=True, draw_heading_pt=False)

    viz_feature(
        in_data='/home/kosuke/ros/autoware_ws/src/lidar_instance_segmentation/saved_feature/in_feature_0.npy',
        out_data='/home/kosuke/ros/autoware_ws/src/lidar_instance_segmentation/saved_feature/out_feature_0.npy',
        # in_data='/media/kosuke/SANDISK/nusc/yaw_two_infer/in_feature/00000.npy',
        # out_data='/media/kosuke/SANDISK/nusc/yaw_two_infer/inference_feature/00000.npy',
        # out_data='/media/kosuke/SANDISK/nusc/yaw_two_infer/out_feature/00000.npy',
        width=672, height=672, grid_range=70.,
        # draw_target='instance',
        draw_target='heading',
        use_cnpy_feature=True)
