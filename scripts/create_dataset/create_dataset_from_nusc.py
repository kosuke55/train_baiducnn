#!/usr/bin/env python
# coding: utf-8

"""
python 3.7.3

"""

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import feature_generator as fg
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def view_points(points: np.ndarray,
                view: np.ndarray,
                normalize: bool) -> np.ndarray:

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def points_in_box2d(box2d: np.ndarray, points: np.ndarray):
    p1 = box2d[0]
    p_x = box2d[1]
    p_y = box2d[3]

    i = p_x - p1
    j = p_y - p1
    v = points - p1

    iv = np.dot(i, v)
    jv = np.dot(j, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask = np.logical_and(mask_x, mask_y)

    return mask


# dataroot \
#     = "/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e/nuScenes/trainval"
# SAVE_DIR \
#     = "/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e/baidu_train_data/all/"

# dataroot \
#     = "/media/kosuke/HD-PNFU3/0413/nusc/trainval"
dataroot \
    = "/media/kosuke/HD-PNFU3/0413/nusc/v1.0-mini"
SAVE_DIR \
    = "/media/kosuke/HD-PNFU3/0413/nusc/mini-0427"
# SAVE_DIR \
#     = "/media/kosuke/HD-PNFU3/nusc/feature_instance_one/"

os.makedirs(os.path.join(SAVE_DIR, "in_feature"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "out_feature"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "loss_weight"), exist_ok=True)

nusc_version = "v1.0-mini"
# nusc_version = "v1.0-trainval"

nusc = NuScenes(
    version=nusc_version,
    dataroot=dataroot, verbose=True)
ref_chan = 'LIDAR_TOP'

grid_range = 60
size = 640
width = 640
height = 640
# grid_range = 70
# size = 672
# width = 672
# height = 672
gsize = 2 * grid_range / size
channel = 5

data_id = 0
end_id = None
# end_id = 1

count = 0
for my_scene in nusc.scene:
    first_sample_token = my_scene['first_sample_token']
    token = first_sample_token

    while(token != ''):
        print("--- {} ".format(data_id) + token + " ---")
        out_feature = np.zeros((size, size, 7), dtype=np.float16)
        loss_weight = np.full((size, size, 1), 0.5, dtype=np.float16)
        my_sample = nusc.get('sample', token)
        sd_record = nusc.get('sample_data', my_sample['data'][ref_chan])
        sample_rec = nusc.get('sample', sd_record['sample_token'])
        chan = sd_record['channel']

        pc, times = LidarPointCloud.from_file_multisweep(
            nusc, sample_rec, chan, ref_chan, nsweeps=10)
        _, boxes, _ = nusc.get_sample_data(sd_record['token'], box_vis_level=0)

        # not needed. This is equal to points = pc.points[:3, :]
        # points = pc.points[:3, :]
        points = view_points(pc.points[:3, :], np.eye(4), normalize=False)

        if count == 0:
            header = '''# .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z rgb
            SIZE 4 4 4 4
            TYPE F F F F
            COUNT 1 1 1 1
            WIDTH %d
            HEIGHT %d
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS %d
            DATA ascii'''
            with open(os.path.join(
                    SAVE_DIR, "00000.pcd"), 'w') as f:
                f.write(header % (pc.points.shape[1], 1, pc.points.shape[1]))
                f.write("\n")
                for p in pc.points.T.tolist():
                    f.write('%f %f %f %e' % (p[0], p[1], p[2], p[3]))
                    f.write("\n")
        count += 1

        # print(points)
        # print(points_hoge)
        # print(points.shape)
        # print(points_hoge.shape)
        # if points == points_hoge:
        #     print("equal")
        # else:
        #     print("not equal")
        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))

        ticks = np.arange(-grid_range, grid_range + gsize, gsize)

        fig = plt.figure()
        # ax = fig.add_subplot(111)
        for box_idx, box in enumerate(boxes):
            label = 0
            if box.name.split(".")[0] == "vehicle":
                if box.name.split(".")[1] == "car":
                    label = 1
                elif box.name.split(".")[1] == "bus":
                    label = 2
                elif box.name.split(".")[1] == "truck":
                    label = 2
                elif box.name.split(".")[1] == "bicycle":
                    label = 3
            elif box.name.split(".")[0] == "human":
                label = 4
            else:
                continue
            view = np.eye(4)

            corners3d = view_points(box.corners(), view, normalize=False)
            height_pt = np.linalg.norm(corners3d.T[0] - corners3d.T[3])

            # corners 2d
            corners = corners3d[:2, :]
            box2d = corners.T[[2, 3, 7, 6]]
            # print("box2d: ", box2d)

            # object_center = [np.average(box2d[:, 0]), np.average(box2d[:, 1])]

            # corners_height = corners3d[2, :]
            # height_pt = corners_height[0] - corners_height[2]

            # find search_area
            box2d_left = box2d[:, 0].min()
            box2d_right = box2d[:, 0].max()
            box2d_top = box2d[:, 1].max()
            box2d_bottom = box2d[:, 1].min()

            grid_centers = (ticks + gsize / 2)[:len(ticks) - 1]

            search_area_left_idx = np.abs(
                grid_centers - box2d_left).argmin() - 1
            search_area_right_idx = np.abs(
                grid_centers - box2d_right).argmin() + 1
            search_area_bottom_idx = np.abs(
                grid_centers - box2d_bottom).argmin() - 1
            search_area_top_idx = np.abs(
                grid_centers - box2d_top).argmin() + 1

            box2d_center = box2d.mean(axis=0)
            box_fill_area = np.array([box2d[:, 0], box2d[:, 1]])

            plt.fill(box_fill_area[0], box_fill_area[1], color="b", alpha=0.1)
            # plt.fill(box_fill_area[0], box_fill_area[1], color="b", alpha=0.1)
            c = patches.Circle(xy=(box2d_center[0], box2d_center[1]),
                               radius=0.1, fc='b', ec='b', fill=False)
            # fig, ax = plt.subplots()
            # ax.add_patch(c)
            # print(object_center)
            # print(box2d_center)
            # raise
            # start from lefght bottom, go right.
            for i in range(search_area_left_idx, search_area_right_idx):
                for j in range(search_area_bottom_idx, search_area_top_idx):
                    # grid_center is in meter coords
                    grid_center = np.array([grid_centers[i], grid_centers[j]])
                    # fill_area = np.array([[(grid_center[0] - gsize / 2),
                    #                        (grid_center[0] + gsize / 2),
                    #                        (grid_center[0] + gsize / 2),
                    #                        (grid_center[0] - gsize / 2)],
                    #                       [(grid_center[1] + gsize / 2),
                    #                        (grid_center[1] + gsize / 2),
                    #                        (grid_center[1] - gsize / 2),
                    #                        (grid_center[1] - gsize / 2)]])
                    if points_in_box2d(box2d, grid_center):
                    #     plt.fill(fill_area[0], fill_area[1], color="r", alpha=0.1)
                    #     plt.arrow(x=grid_center[0],
                    #               y=grid_center[1],
                    #               dx=box2d_center[0] - grid_center[0],
                    #               dy=box2d_center[1] - grid_center[1],
                    #               width=0.01,
                    #               head_width=0.01,
                    #               head_length=0.01,
                    #               length_includes_head=True,
                    #               color='k')
                        out_feature[i, j, 0] = 1.  # category_pt
                        # instance_pt = object_center - grid_center
                        instance_pt = box2d_center - grid_center
                        # print("instace_pt  ", instance_pt)
                        # print(i,j)
                        out_feature[i, j, 1] = instance_pt[0]  # instance_pt x
                        out_feature[i, j, 2] = instance_pt[1]  # instance_pt y
                        # out_feature[i, j, 1] = box2d_center[0] - grid_center[0]
                        # out_feature[i, j, 2] = box2d_center[1] - grid_center[1]
                        out_feature[i, j, 3] = 1.  # confidence_pt
                        out_feature[i, j, 4] = label  # classify_pt
                        out_feature[i, j, 5] = 0  # heading_pt (unused)
                        out_feature[i, j, 6] = height_pt  # height_pt

                        loss_weight[i, j, 0] = 1.

        # fill_area = np.array([[(grid_center[0] - gsize / 2),
        #                        (grid_center[0] + gsize / 2),
        #                        (grid_center[0] + gsize / 2),
        #                        (grid_center[0] - gsize / 2)],
        #                       [(grid_center[1] + gsize / 2),
        #                        (grid_center[1] + gsize / 2),
        #                        (grid_center[1] - gsize / 2),
        #                        (grid_center[1] - gsize / 2)]])

        # plt.fill(fill_area[0], fill_area[1], color="b", alpha=0.1)


        # just draw coords arrow
        # fill_area = np.array([[(1 - gsize / 2),
        #                        (1 + gsize / 2),
        #                        (1 + gsize / 2),
        #                        (1 - gsize / 2)],
        #                       [(0 + gsize / 2),
        #                        (0 + gsize / 2),
        #                        (0 - gsize / 2),
        #                        (0 - gsize / 2)]])

        # plt.fill(fill_area[0], fill_area[1], color="r", alpha=1)

        # fill_area = np.array([[(0 - gsize / 2),
        #                        (0 + gsize / 2),
        #                        (0 + gsize / 2),
        #                        (0 - gsize / 2)],
        #                       [(1 + gsize / 2),
        #                        (1 + gsize / 2),
        #                        (1 - gsize / 2),
        #                        (1 - gsize / 2)]])

        # plt.fill(fill_area[0], fill_area[1], color="g", alpha=1)

        # print(grid_centers)
        # gsize *= 10
        # fill_area = np.array([[(grid_centers[10] - gsize / 2),
        #                        (grid_centers[10] + gsize / 2),
        #                        (grid_centers[10] + gsize / 2),
        #                        (grid_centers[10] - gsize / 2)],
        #                       [(grid_centers[10] + gsize / 2),
        #                        (grid_centers[10] + gsize / 2),
        #                        (grid_centers[10] - gsize / 2),
        #                        (grid_centers[10] - gsize / 2)]])
        # plt.fill(fill_area[0], fill_area[1], color="g", alpha=1)

        # plt.show()

        # This is input feature
        feature_generator = fg.Feature_generator(grid_range, width, height)
        feature_generator.generate(pc.points.T)
        in_feature = feature_generator.feature

        # check if input data is correct
        grid_centers = (ticks + gsize / 2)[:len(ticks) - 1]

        # pos_y, pos_x, 8
        in_feature = in_feature.reshape(size, size, 8)
        # in_feature = in_feature.reshape(size, size, 6)
        out_feature = np.flip(np.flip(out_feature, axis=0), axis=1)
        out_feature[:, :, 1:3] *= -1
        # out_feature[:, :, 1:3] = out_feature[:, :, 1:3] * -1  # instance_pt x
        loss_weight = np.flip(np.flip(loss_weight, axis=0), axis=1)

        np.save(os.path.join(
            SAVE_DIR, "in_feature/{:05}".format(data_id)), in_feature)
        np.save(os.path.join(
            SAVE_DIR, "out_feature/{:05}".format(data_id)), out_feature)
        np.save(os.path.join(
            SAVE_DIR, "loss_weight/{:05}".format(data_id)), loss_weight)

        token = my_sample['next']
        data_id += 1
        if data_id == end_id:
            break
    if data_id == end_id:
        break
