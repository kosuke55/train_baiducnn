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


def get_arrow_image(in_feature, out_feature, width=672, height=672,
                    grid_range=70., draw_target='instance', thresh=0.3,
                    viz_range=1 / 3., viz_all_grid=False):
    """Visualize the direction of instance and heading with arrows.

    Only supported in the following cases.
        use_intensity_feature = True
        use_constant_feature = False

    TO DO
        Supports all intensity and constant conditions.

    Parameters
    ----------
    in_feature : numpy.ndarray
    out_feature : numpy.ndarray
    width : int, optional
        feature map width, by default 672
    height : int, optional
        feature map height, by default 672
    grid_range : floar, optional
        feature map range, by default 70.
    draw_target : str, optional
        whether to visualize instance or heading, by default 'instance'
    thresh : float, optional
        Pixels above the threshold are classified as objects, by default 0.3
    viz_range : float, optional
        visualization range of feature_map, by default 1/3.
    viz_all_grid : bool, optional
        Whether to visualize non-object pixels, by default False

    Returns
    -------
    img: numpy.ndarray
        Image of instance or heading represented by an arrow.

    Raises
    ------
    Exception
        Width and height are not equal.

    """
    print('drawing axis image...')
    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    grid_ticks = np.arange(-grid_range, grid_range + grid_length, grid_length)

    grid_centers = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    fig, ax = plt.subplots()

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

    for i_tmp in range(int(height * viz_range)):
        for j_tmp in range(int(width * viz_range)):
            i = int(height / 2 - height * viz_range / 2 + i_tmp)
            j = int(width / 2 - width * viz_range / 2 + j_tmp)
            if in_feature[i, j, 5] == 1:
                fill_grid(i, j, 'r')
            if out_feature[i, j, 0] > thresh or viz_all_grid:
                if out_feature[i, j, 0] > thresh:
                    fill_grid(i, j, 'b')
                grid_center = np.array([grid_centers[j], -grid_centers[i]])

                if not (np.mod(i, 5) == 0 and np.mod(j, 5) == 0):
                    continue
                if draw_target == 'instance':
                    dx = out_feature[i, j, 2]
                    dy = out_feature[i, j, 1]
                elif draw_target == 'heading':
                    yaw = math.atan2(out_feature[i, j, 10],
                                     out_feature[i, j, 9]) * 0.5
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

    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close()

    return image


def viz_feature(
        in_feature, out_feature, width=672, height=672, grid_range=70.,
        draw_target='instance', viz_all_grid=False,
        save_image=True, use_cnpy=False):

    # in_feature = np.load(in_data)
    # out_feature = np.load(out_data)

    # if use_cnpy_feature:
    #     in_feature = in_feature.transpose(1, 2, 0)
    #     out_feature = out_feature.transpose(1, 2, 0)

    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    grid_ticks = np.arange(-grid_range, grid_range + grid_length, grid_length)

    grid_centers = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    fig, ax = plt.subplots()

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
    # for i in range(height):
    #     for j in range(width):
    viz_range = 1 / 2.
    for i_tmp in range(int(height * viz_range)):
        for j_tmp in range(int(width * viz_range)):
            i = int(height / 2 - height * viz_range / 2 + i_tmp)
            j = int(width / 2 - width * viz_range / 2 + j_tmp)
            if in_feature[i, j, 5] == 1:
                fill_grid(i, j, 'r')
            if out_feature[i, j, 0] > 0.3 or viz_all_grid:
                # if out_feature[i, j, 0] > 0.5:
                if out_feature[i, j, 0] > 0.3:
                    instance_norms.append(
                        np.linalg.norm([out_feature[i, j, 2],
                                        out_feature[i, j, 1]]))
                    fill_grid(i, j, 'b')
                grid_center = np.array([grid_centers[j], -grid_centers[i]])

                if not (np.mod(i, 2) == 0 and np.mod(j, 2) == 0):
                    continue
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
    plt.title(in_data)
    if save_image:
        print('saving image...')
        plt.savefig(draw_target + '.png', format='png', dpi=1000)
        print('saved image')
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

        print('instance norm \nmean: {}\nmax: {}\nmin: {}'.format(
            np.mean(instance_norms),
            np.max(
                instance_norms),
            np.min(instance_norms)))
    plt.show()


def yaw2yaw(yaw):
    normalized_yaw = math.atan(math.sin(yaw) / math.cos(yaw))
    yaw = math.atan2(math.sin(normalized_yaw * 2.0),
                     math.cos(normalized_yaw * 2.0)) * 0.5
    return yaw


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
    in_data = '/media/kosuke/SANDISK/nusc/yaw_two/in_feature/00001.npy'
    out_data = '/media/kosuke/SANDISK/nusc/yaw_two/out_feature/00001.npy'

    use_cnpy_feature = False

    in_feature = np.load(in_data)
    out_feature = np.load(out_data)

    if use_cnpy_feature:
        in_feature = in_feature.transpose(1, 2, 0)
        out_feature = out_feature.transpose(1, 2, 0)

    # viz_feature(
    #     in_feature=in_feature,
    #     out_feature=out_feature,
    #     width=672, height=672, grid_range=70.,
    #     # draw_target='instance',
    #     draw_target='heading',
    #     viz_all_grid=False,
    #     save_image=False)

    get_arrow_image(
        in_feature=in_feature,
        out_feature=out_feature,
        width=672, height=672, grid_range=70.,
        draw_target='heading',
        viz_all_grid=False)
