#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import sys
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numba
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms

import feature_generator_pb as fgpb
from BCNN import BCNN

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.data_classes import LidarPointCloud
            sys.path.append(path)
            break


def F2I(val, orig, scale):
    return int(math.floor((orig - val) * scale))


def Pixel2pc(in_pixel, in_size, out_range):
    res = 2.0 * out_range / in_size
    return out_range - (in_pixel + 0.5) * res


class Feature_generator():
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


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


def create_dataset(dataroot, save_dir, pretrained_model, width=672, height=672, grid_range=70.,
                   nusc_version='v1.0-mini',
                   use_constant_feature=True, use_intensity_feature=True,
                   end_id=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bcnn_model = BCNN(in_channels=6, n_class=5).to(device)
    # bcnn_model = torch.nn.DataParallel(bcnn_model)  # multi gpu
    if os.path.exists(pretrained_model):
        print('Use pretrained model')
        bcnn_model.load_state_dict(
            fix_model_state_dict(torch.load(pretrained_model)))
        bcnn_model.eval()
    else:
        return

    os.makedirs(os.path.join(save_dir, 'in_feature'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'out_feature'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'inference_feature'), exist_ok=True)

    nusc = NuScenes(
        version=nusc_version,
        dataroot=dataroot, verbose=True)
    ref_chan = 'LIDAR_TOP'

    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    label_half_length = 0

    data_id = 0
    grid_ticks = np.arange(
        -grid_range, grid_range + grid_length, grid_length)
    grid_centers \
        = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    for my_scene in nusc.scene:
        first_sample_token = my_scene['first_sample_token']
        token = first_sample_token

        # try:
        while(token != ''):
            print('--- {} '.format(data_id) + token + ' ---')
            my_sample = nusc.get('sample', token)
            sd_record = nusc.get(
                'sample_data', my_sample['data'][ref_chan])
            sample_rec = nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            pc, times = LidarPointCloud.from_file_multisweep(
                nusc, sample_rec, chan, ref_chan, nsweeps=1)
            _, boxes, _ = nusc.get_sample_data(
                sd_record['token'], box_vis_level=0)
            out_feature = np.zeros((size, size, 8), dtype=np.float32)
            for box_idx, box in enumerate(boxes):
                label = 0
                if box.name.split('.')[0] == 'vehicle':
                    if box.name.split('.')[1] == 'car':
                        label = 1
                    elif box.name.split('.')[1] == 'bus':
                        label = 1
                    elif box.name.split('.')[1] == 'truck':
                        label = 1
                    elif box.name.split('.')[1] == 'construction':
                        label = 1
                    elif box.name.split('.')[1] == 'emergency':
                        label = 1
                    elif box.name.split('.')[1] == 'trailer':
                        label = 1
                    elif box.name.split('.')[1] == 'bicycle':
                        label = 2
                    elif box.name.split('.')[1] == 'motorcycle':
                        label = 2
                elif box.name.split('.')[0] == 'human':
                    label = 3
                # elif box.name.split('.')[0] == 'movable_object':
                #     label = 1
                # elif box.name.split('.')[0] == 'static_object':
                #     label = 1
                else:
                    continue
                height_pt = np.linalg.norm(
                    box.corners().T[0] - box.corners().T[3])
                corners2d = box.corners()[:2, :]
                box2d = corners2d.T[[2, 3, 7, 6]]
                box2d_center = box2d.mean(axis=0)
                yaw, pitch, roll = box.orientation.yaw_pitch_roll
                generate_out_feature(width, height, size, grid_centers,
                                     box2d, box2d_center, height_pt,
                                     label, label_half_length, yaw,
                                     out_feature)

            out_feature = out_feature.astype(np.float16)
            # in_feature_generator = Feature_generator(
            #     grid_range, width, height,
            #     use_constant_feature, use_intensity_feature)
            # in_feature_generator.generate(pc.points.T)
            feature_generator = fgpb.FeatureGenerator(
                grid_range, width, height)
            in_feature = feature_generator.generate(
                pc.points.T, use_constant_feature, use_intensity_feature)

            if use_constant_feature and use_intensity_feature:
                channels = 8
            elif use_constant_feature or use_intensity_feature:
                channels = 6
            else:
                channels = 4

            in_feature = np.array(in_feature).reshape(
                channels, size, size).astype(np.float16)
            in_feature = in_feature.transpose(1, 2, 0)

            np.save(os.path.join(
                save_dir, 'in_feature/{:05}'.format(data_id)),
                in_feature)
            np.save(os.path.join(
                save_dir, 'out_feature/{:05}'.format(data_id)),
                out_feature)

            in_feature = in_feature.astype(np.float32)
            transform = transforms.Compose([transforms.ToTensor()])
            in_feature = transform(in_feature)
            in_feature = in_feature.to(device)
            in_feature = torch.unsqueeze(in_feature, 0)
            print(in_feature.size())
            inference_feature = bcnn_model(in_feature)
            np.save(os.path.join(
                save_dir, 'inference_feature/{:05}'.format(data_id)),
                inference_feature.cpu().detach().numpy()[0, ...].transpose(1, 2, 0))
            token = my_sample['next']
            data_id += 1
            if data_id == end_id:
                return
        # except KeyboardInterrupt:
        #     return
        # except BaseException:
        #     print('skipped')
        #     continue


@numba.jit(nopython=True)
def generate_out_feature(
        width, height, size, grid_centers, box2d, box2d_center,
        height_pt, label, label_half_length, yaw, out_feature):
    box2d_left = box2d[:, 0].min()
    box2d_right = box2d[:, 0].max()
    box2d_top = box2d[:, 1].max()
    box2d_bottom = box2d[:, 1].min()

    def F2I(val, orig, scale):
        return int(np.floor((orig - val) * scale))

    def Pixel2pc(in_pixel, in_size, out_range):
        res = 2.0 * out_range / in_size
        return out_range - (in_pixel + 0.5) * res

    inv_res = 0.5 * width / 70.
    res = 1.0 / inv_res
    max_length = abs(2*res)

    search_area_left_idx = F2I(box2d_left, 70, inv_res)
    search_area_right_idx = F2I(box2d_right, 70, inv_res)
    search_area_top_idx = F2I(box2d_top, 70, inv_res)
    search_area_bottom_idx = F2I(box2d_bottom, 70, inv_res)

    for i in range(
            search_area_right_idx - 1, search_area_left_idx + 1):
        for j in range(
                search_area_top_idx - 1, search_area_bottom_idx + 1):
            if 0 <= i and i < size and 0 <= j and j < size:
                # grid_center = np.array(
                #     [grid_centers[i], grid_centers[j]])
                grid_center_x = Pixel2pc(i, float(height), 70)
                grid_center_y = Pixel2pc(j, float(width), 70)

                p1 = box2d[0]
                p_x = box2d[1]
                p_y = box2d[3]
                pi = p_x - p1
                pj = p_y - p1
                v = np.array([grid_center_x, grid_center_y]) - p1
                iv = np.dot(pi, v)
                jv = np.dot(pj, v)
                mask_x = np.logical_and(0 <= iv, iv <= np.dot(pi, pi))
                mask_y = np.logical_and(0 <= jv, jv <= np.dot(pj, pj))
                mask = np.logical_and(mask_x, mask_y)

                if max_length < abs(box2d_center[0] - grid_center_x):
                    x_scale = max_length / abs(box2d_center[0] - grid_center_x)
                else:
                    x_scale = 1.
                if max_length < abs(box2d_center[1] - grid_center_y):
                    y_scale = max_length / abs(box2d_center[1] - grid_center_y)
                else:
                    y_scale = 1.

                normalized_yaw = math.atan(math.sin(yaw) / math.cos(yaw))

                # normalized_yaw =  math.atan2(math.sin(yaw), math.cos(yaw))
                # while normalized_yaw < -pi/2.0 :
                #     normalized_yaw = normalized_yaw + pi

                # while pi/2.0 < normalized_yaw :
                #     normalized_yaw = normalized_yaw - pi

                if mask:
                    out_feature[i, j, 0] = 1.  # category_pt
                    out_feature[i, j, 1] = (
                        (box2d_center[0] - grid_center_x) * -1) * min(x_scale, y_scale)
                    out_feature[i, j, 2] = (
                        (box2d_center[1] - grid_center_y) * -1) * min(x_scale, y_scale)
                    out_feature[i, j, 3] = 1.  # confidence_pt
                    out_feature[i, j, 4] = label  # classify_pt
                    out_feature[i, j, 5] = math.cos(normalized_yaw * 2.0)
                    out_feature[i, j, 6] = math.sin(normalized_yaw * 2.0)
                    out_feature[i, j, 7] = height_pt  # height_pt

    return out_feature


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataroot', '-dr', type=str,
                        help='Nuscenes dataroot path',
                        default='/media/kosuke/SANDISK/nusc/v1.0-mini')
    parser.add_argument('--save_dir', '-sd', type=str,
                        help='Dataset save directory',
                        default='/media/kosuke/SANDISK/nusc/mini-6c-672')
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=672)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=672)
    parser.add_argument('--range', type=int,
                        help='feature map range',
                        default=70)
    parser.add_argument('--nusc_version', type=str,
                        help='Nuscenes version. v1.0-mini or v1.0-trainval',
                        default='v1.0-mini')
    parser.add_argument('--use_constant_feature', type=str,
                        help='Whether to use constant feature',
                        default=False)
    parser.add_argument('--use_intensity_feature', type=str,
                        help='Whether to use intensity feature',
                        default=True)
    parser.add_argument('--end_id', type=int,
                        help='How many data to generate. If None, all data',
                        default=None)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model',
                        default='checkpoints/mini_672_6c.pt')

    args = parser.parse_args()
    create_dataset(dataroot=args.dataroot,
                   save_dir=args.save_dir,
                   pretrained_model=args.pretrained_model,
                   width=args.width,
                   height=args.height,
                   grid_range=args.range,
                   nusc_version=args.nusc_version,
                   use_constant_feature=args.use_constant_feature,
                   use_intensity_feature=args.use_intensity_feature,
                   end_id=args.end_id)
