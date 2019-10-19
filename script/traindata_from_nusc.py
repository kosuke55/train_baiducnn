#!/usr/bin/env python
# coding: utf-8

"""
under development.
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
from typing import Tuple
import h5py
import feature_generator as fg


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


def get_color(category_name: str) -> Tuple[int, int, int]:
    """ Provides the default colors based on the category names. """
    if category_name in ['vehicle.bicycle', 'vehicle.motorcycle']:
        return 255, 61, 99  # Red
    elif 'vehicle' in category_name:
        return 255, 158, 0  # Orange
    elif 'human.pedestrian' in category_name:
        return 0, 0, 230  # Blue
    elif 'cone' in category_name or 'barrier' in category_name:
        return 0, 0, 0  # Black
    else:
        return 255, 0, 255  # Magenta


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


grid_range = 60
size = 640
# size = 100
rows = 640
cols = 640

gsize = 2 * grid_range / size

# center -> x, y
out_feature = np.zeros((1, size, size, 6))
print(out_feature.shape)

channel = 5
dataroot = '/home/kosuke/dataset/nuScenes/'
nusc = NuScenes(
    version='v1.0-mini',
    dataroot=dataroot, verbose=True)

my_scene = nusc.scene[0]
token = my_scene['first_sample_token']

ref_chan = 'LIDAR_TOP'

my_sample = nusc.get('sample', token)
my_sample = nusc.sample[20]

sd_record = nusc.get('sample_data', my_sample['data'][ref_chan])
sample_rec = nusc.get('sample', sd_record['sample_token'])
chan = sd_record['channel']

pc, times = LidarPointCloud.from_file_multisweep(
    nusc, sample_rec, chan, ref_chan, nsweeps=10)
_, boxes, _ = nusc.get_sample_data(sd_record['token'], box_vis_level=0)

# not needed. This is equal to points = pc.points[:3, :]
points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))

axes_limit = grid_range
colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

_, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
ax.plot(0, 0, 'x', color='black')
ax.grid(which="major", color="silver")
ticks = np.arange(-grid_range, grid_range + gsize, gsize)

grid_x, grid_y = np.meshgrid(ticks, ticks)
grid_x = grid_x.flatten()
grid_y = grid_y.flatten()

plt.tick_params(labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False)

ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlim(-grid_range, grid_range)
ax.set_ylim(-grid_range, grid_range)

for box in boxes:
    c = np.array(get_color(box.name)) / 255.0
    box.render(ax, view=np.eye(4), colors=(c, c, c))

box = boxes[2]
for box_idx, box in enumerate(boxes):
    print("box_idx  {}/{}".format(box_idx, len(boxes)))
    view = np.eye(4)

    corners3d = view_points(box.corners(), view, normalize=False)
    height = np.linalg.norm(corners3d.T[0] - corners3d.T[3])

    # corners 2d
    corners = corners3d[:2, :]
    box2d = corners.T[[2, 3, 7, 6]]
    corners_height = corners3d[2, :]
    height = corners_height[0] - corners_height[2]

    # plt.scatter(box2d[:, 0], box2d[:, 1], marker='^', s=100)

    # find search_area
    box2d_left = box2d[:, 0].min()
    box2d_right = box2d[:, 0].max()
    box2d_top = box2d[:, 1].max()
    box2d_bottom = box2d[:, 1].min()

    grid_centers = (ticks + gsize / 2)[:len(ticks) - 1]

    search_area_left_idx = np.abs(grid_centers - box2d_left).argmin() - 1
    search_area_right_idx = np.abs(grid_centers - box2d_right).argmin() + 1
    search_area_bottom_idx = np.abs(grid_centers - box2d_bottom).argmin() - 1
    search_area_top_idx = np.abs(grid_centers - box2d_top).argmin() + 1

    box2d_center = box2d.mean(axis=0)

    # start from lefght bottom, go right.
    for i in range(search_area_left_idx, search_area_right_idx):
        for j in range(search_area_bottom_idx, search_area_top_idx):
            # grid_center is in meter coords
            grid_center = np.array([grid_centers[i], grid_centers[j]])
            # print(i*len(grid_centers) + j)
            fill_area = np.array([[(grid_center[0] - gsize / 2),
                                   (grid_center[0] + gsize / 2),
                                   (grid_center[0] + gsize / 2),
                                   (grid_center[0] - gsize / 2)],
                                  [(grid_center[1] + gsize / 2),
                                   (grid_center[1] + gsize / 2),
                                   (grid_center[1] - gsize / 2),
                                   (grid_center[1] - gsize / 2)]])
            if(points_in_box2d(box2d, grid_center)):
                plt.fill(fill_area[0], fill_area[1], color="r", alpha=0.1)
                # grid center to object center dx
                out_feature[0, i, j, 0] = box2d_center[0] - grid_center[0]
                # grid center to object center dy
                out_feature[0, i, j, 1] = box2d_center[1] - grid_center[1]
                # objectness
                out_feature[0, i, j, 2] = 1.
                # positiveness
                out_feature[0, i, j, 3] = 1.
                # object_hight
                out_feature[0, i, j, 4] = height
                # class probability
                out_feature[0, i, j, 5] = 1.


# This is input feature
feature_generator = fg.Feature_generator()
feature_generator.generate(pc.points.T)
# print(pc.points.shape)
# for i in range(10):
#     print(pc.points.T[i])

feature = feature_generator.feature
# feature = feature_generator.feature[::-1]


# feature = feature_generator.feature.T.reshape(
#     1, 8, feature_generator.height, feature_generator.width)
print(feature[feature != 0])
for i in range(8):
    print("{}-----{}".format(i, np.count_nonzero(feature[:, i])))

# check if input data is correct
grid_centers = (ticks + gsize / 2)[:len(ticks) - 1]

# pos_y, pos_x, 8

feature = feature.reshape(size, size, 8)
in_feature = feature[np.newaxis, :, :, :]
print("in_feature.shape = " + str(in_feature.shape))

nonzero_idx = np.where(feature[:, :, 7] != 0)
print(nonzero_idx)

grid_center = np.array([grid_centers[size - 1 - nonzero_idx[0]],
                        grid_centers[size - 1 - nonzero_idx[1]]])
# grid_center = np.array([grid_centers[nonzero_idx[0]],
#                         grid_centers[nonzero_idx[1]]])

fill_area = np.array([[(grid_center[0] - gsize / 2),
                       (grid_center[0] + gsize / 2),
                       (grid_center[0] + gsize / 2),
                       (grid_center[0] - gsize / 2)],
                      [(grid_center[1] + gsize / 2),
                       (grid_center[1] + gsize / 2),
                       (grid_center[1] - gsize / 2),
                       (grid_center[1] - gsize / 2)]])

plt.fill(fill_area[0], fill_area[1], color="b", alpha=0.1)


# just draw coords arrow
fill_area = np.array([[(1 - gsize / 2),
                       (1 + gsize / 2),
                       (1 + gsize / 2),
                       (1 - gsize / 2)],
                      [(0 + gsize / 2),
                       (0 + gsize / 2),
                       (0 - gsize / 2),
                       (0 - gsize / 2)]])

plt.fill(fill_area[0], fill_area[1], color="r", alpha=1)

fill_area = np.array([[(0 - gsize / 2),
                       (0 + gsize / 2),
                       (0 + gsize / 2),
                       (0 - gsize / 2)],
                      [(1 + gsize / 2),
                       (1 + gsize / 2),
                       (1 - gsize / 2),
                       (1 - gsize / 2)]])

plt.fill(fill_area[0], fill_area[1], color="g", alpha=1)

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

# To do
# save in_feature and out_feature as h5
# The upper right is the first. Then go left.
# So need to change the order of out_feature.
# print(out_feature)
print("out_featere.shape" + str(out_feature.shape))
out_feature = np.flip(np.flip(out_feature, axis=1), axis=2)

with h5py.File('nusc_baidu.h5', 'w') as f:
    # transform data into caffe format
    out_feature = np.transpose(
        out_feature, (0, 3, 2, 1))  # NxWxHxC -> NxCxHxW
    print(out_feature.shape)
    f.create_dataset('output', dtype=np.float, data=out_feature)
    in_feature = np.transpose(
        in_feature, (0, 3, 2, 1))  # NxWxHxC -> NxCxHxW
    print(in_feature.shape)
    f.create_dataset('input', dtype=np.float, data=in_feature)

# debug now. this may be better
# out_feature = np.transpose(
#     out_feature, (0, 3, 2, 1))  # NxWxHxC -> NxCxHxW
# in_feature = np.transpose(
#     in_feature, (0, 3, 2, 1))  # NxWxHxC -> NxCxHxW
# f = h5py.File("nusc_baidu_2.h5", "w")
# f.create_dataset("data", in_feature.shape, dtype="f8")
# f.create_dataset("label", out_feature.shape, dtype="f8")
# f["data"][:] = in_feature.astype("f8")
# f["label"][:] = out_feature.astype("f8")
# f.close
