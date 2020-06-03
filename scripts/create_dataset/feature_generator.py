#!/usr/bin/env python3
# coding: utf-8

import math

import numpy as np


def F2I(val, orig, scale):
    # return int(np.floor((orig - val) * scale))
    return np.floor((orig - val) * scale).astype(np.uint8)


def Pixel2pc(in_pixel, in_size, out_range):
    res = 2.0 * out_range / in_size
    return out_range - (in_pixel + 0.5) * res


class FeatureGenerator():
    def __init__(self, grid_range, width, height,
                 use_constant_feature, use_intensity_feature):
        self.range = grid_range
        self.width = int(width)
        self.height = int(height)
        self.siz = self.width * self.height
        self.min_height = -5.0
        self.max_height = 5.0

        self.log_table = np.zeros(256)
        for i in range(len(self.log_table)):
            self.log_table[i] = np.log1p(i)

        if use_constant_feature and use_intensity_feature:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.direction_data = 3
            self.top_intensity_data = 4
            self.mean_intensity_data = 5
            self.distance_data = 6
            self.nonempty_data = 7
            self.feature = np.zeros((self.siz, 8), dtype=np.float16)

        elif use_constant_feature:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.direction_data = 3
            self.distance_data = 4
            self.nonempty_data = 5
            self.feature = np.zeros((self.siz, 6), dtype=np.float16)

        elif use_intensity_feature:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.top_intensity_data = 3
            self.mean_intensity_data = 4
            self.nonempty_data = 5
            self.feature = np.zeros((self.siz, 6), dtype=np.float16)

        else:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.nonempty_data = 3
            self.feature = np.zeros((self.siz, 4), dtype=np.float16)

        if use_constant_feature:
            for row in range(self.height):
                for col in range(self.width):
                    idx = row * self.width + col
                    center_x = Pixel2pc(row, self.height, self.range)
                    center_y = Pixel2pc(col, self.width, self.range)
                    self.feature[idx, self.direction_data] \
                        = np.arctan2(center_y, center_x) / (2.0 * np.pi)
                    self.feature[idx, self.distance_data] \
                        = np.hypot(center_x, center_y) / 60. - 0.5

    def logCount(self, count):
        if count < len(self.log_table):
            return self.log_table[count]
        else:
            return np.log(1 + count)

    def load_pc_from_file(self, pc_f):
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])

    def generate(self, points):
        self.map_idx = np.zeros(len(points))
        inv_res_x = 0.5 * self.width / self.range
        inv_res_y = 0.5 * self.height / self.range
        for i in range(len(points)):
            if points[i, 2] <= self.min_height or \
               points[i, 2] >= self.max_height:
                self.map_idx[i] = -1
            pos_x = F2I(points[i, 1], self.range, inv_res_x)
            pos_y = F2I(points[i, 0], self.range, inv_res_y)
            if pos_x >= self.width or pos_x < 0 or \
               pos_y >= self.height or pos_y < 0:
                self.map_idx[i] = -1
                continue
            self.map_idx[i] = pos_y * self.width + pos_x
            idx = int(self.map_idx[i])
            pz = points[i, 2]
            pi = points[i, 3] / 255.0
            if self.feature[idx, self.max_height_data] < pz:
                self.feature[idx, self.max_height_data] = pz
                self.feature[idx, self.top_intensity_data] = pi

            self.feature[idx, self.mean_height_data] += pz
            self.feature[idx, self.mean_intensity_data] += pi
            self.feature[idx, self.count_data] += 1.0

        for i in range(self.siz):
            eps = 1e-6
            if self.feature[i, self.count_data] < eps:
                self.feature[i, self.max_height_data] = 0.0
            else:
                self.feature[i, self.mean_height_data] \
                    /= self.feature[i, self.count_data]
                self.feature[i, self.mean_intensity_data] \
                    /= self.feature[i, self.count_data]
                self.feature[i, self.nonempty_data] = 1.0
            self.feature[i, self.count_data] \
                = self.logCount(int(self.feature[i, self.count_data]))
