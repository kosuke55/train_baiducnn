#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
import sys

import gdown
import numpy as np
import torch
import torch.optim as optim
# import torch_optimizer as optim
import tqdm
import visdom
from datetime import datetime

from BCNN import BCNN
from BcnnLoss import BcnnLoss
from NuscData import load_dataset

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from utils.visualize_utils import get_arrow_image  # noqa


class Trainer(object):

    def __init__(self, data_path, batch_size, max_epoch, pretrained_model,
                 train_data_num, val_data_num,
                 width, height, use_constant_feature, use_intensity_feature):

        self.train_dataloader, self.val_dataloader \
            = load_dataset(data_path, batch_size)
        self.max_epoch = max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.best_loss = 1e10
        self.vis = visdom.Visdom()
        self.vis_interval = 1

        if use_constant_feature and use_intensity_feature:
            self.in_channels = 8
            self.non_empty_channle = 7
        elif use_constant_feature or use_intensity_feature:
            self.in_channels = 6
            self.non_empty_channle = 5
        else:
            self.in_channels = 4
            self.non_empty_channle = 3

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BCNN(
            in_channels=self.in_channels, n_class=5).to(self.device)
        self.model = torch.nn.DataParallel(self.model)  # multi gpu
        self.save_model_interval = 1

        if osp.exists(pretrained_model):
            print('Use pretrained model')
            self.model.load_state_dict(torch.load(pretrained_model))
        else:
            print('Not found ', pretrained_model)
            if pretrained_model == 'checkpoints/bestmodel.pt':
                print('Downloading ', pretrained_model)
                gdown.cached_download(
                    'https://drive.google.com/uc?export=download&id=19IPtsVes3w-qogsiJToHmLrjCAdVEl9K',
                    pretrained_model,
                    md5='b124dab72fd6f2b642c6e46e5b142ebf')
                self.model.load_state_dict(torch.load(pretrained_model))

        self.train_data_num = train_data_num
        self.val_data_num = val_data_num

        self.width = width
        self.height = height
        self.grid_range = 70.

        # self.optimizer = optim.RAdam(
        #     self.model.parameters(),
        #     lr=5e-7,
        #     betas=(0.9, 0.999),
        #     eps=1e-9,
        #     weight_decay=1e-5,
        # )
        # self.optimizer = optim.AdaBound(
        #     self.model.parameters(),
        #     lr=1e-4,
        #     betas=(0.9, 0.999),
        #     final_lr=0.1,
        #     gamma=1e-3,
        #     eps=1e-8,
        #     weight_decay=0,
        #     amsbound=False,
        # )
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=2e-6, momentum=0.5, weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epo: 0.9 ** epo)

    def get_arrow_image(self, in_feature, out_feature):
        img = get_arrow_image(in_feature, out_feature,
                              self.width, self.height, self.grid_range,
                              'heading')
        img = img.transpose(2, 0, 1)
        return img

    def get_category_or_confidence_image(self, feature, thresh=0.3):
        feature_np = feature.cpu().detach().numpy().copy()
        feature_np = feature_np.transpose(1, 2, 0)
        img = np.zeros(
            (self.height, self.width, 1), dtype=np.uint8)
        idx = np.where(feature_np[..., 0] > thresh)
        img[idx] = 1.0
        img = img.transpose(2, 0, 1)
        return img

    def get_class_image(self, feature):
        feature_np = feature.cpu().detach().numpy().copy()
        feature_np = feature_np.transpose(1, 2, 0)
        feature_np = np.argmax(feature_np, axis=2)[..., None]
        car_idx = np.where(feature_np[:, :, 0] == 1)
        bus_idx = np.where(feature_np[:, :, 0] == 2)
        bike_idx = np.where(feature_np[:, :, 0] == 3)
        human_idx = np.where(feature_np[:, :, 0] == 4)
        img = np.zeros((self.height, self.width, 3))
        img[car_idx] = [255, 0, 0]
        img[bus_idx] = [0, 255, 0]
        img[bike_idx] = [0, 0, 255]
        img[human_idx] = [0, 255, 255]
        img = img.transpose(2, 0, 1)
        return img

    def step(self, mode):
        print('Start {}'.format(mode))

        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
        elif mode == 'val':
            self.model.eval()
            dataloader = self.val_dataloader

        loss_sum = 0
        category_loss_sum = 0
        confidence_loss_sum = 0
        class_loss_sum = 0
        instance_x_loss_sum = 0
        instance_y_loss_sum = 0
        heading_x_loss_sum = 0
        heading_y_loss_sum = 0
        height_loss_sum = 0

        for index, (in_feature, out_feature_gt) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader),
                desc='{} epoch={}'.format(mode, self.epo), leave=True):
            category_weight = out_feature_gt.detach().numpy().copy()
            category_weight = category_weight[:, 3, ...]
            object_idx = np.where(category_weight == 0)
            nonobject_idx = np.where(category_weight != 0)
            category_weight[object_idx] = 2.0
            category_weight[nonobject_idx] = 1.0
            category_weight = torch.from_numpy(category_weight)
            category_weight = category_weight.to(self.device)

            confidence_weight = out_feature_gt.detach().numpy().copy()
            confidence_weight = confidence_weight[:, 3, ...]
            object_idx = np.where(confidence_weight == 0)
            nonobject_idx = np.where(confidence_weight != 0)
            confidence_weight[object_idx] = 1.0
            confidence_weight[nonobject_idx] = 10.0
            confidence_weight = torch.from_numpy(confidence_weight)
            confidence_weight = confidence_weight.to(self.device)

            class_weight = out_feature_gt.detach().numpy().copy()
            class_weight = class_weight[:, 4:5, ...]
            object_idx = np.where(class_weight != 0)
            nonobject_idx = np.where(class_weight == 0)
            class_weight[object_idx] = 1.0
            class_weight[nonobject_idx] = 1.0
            class_weight = np.concatenate(
                [class_weight,
                 class_weight,
                 class_weight * 15.0,  # bike
                 class_weight * 15.0,  # pedestrian
                 class_weight], axis=1)
            class_weight = torch.from_numpy(class_weight)
            class_weight = class_weight.to(self.device)

            criterion = BcnnLoss().to(self.device)
            in_feature = in_feature.to(self.device)
            out_feature_gt = out_feature_gt.to(self.device)
            output = self.model(in_feature)

            (category_loss, confidence_loss, class_loss, instance_x_loss,
             instance_y_loss, heading_x_loss, heading_y_loss, height_loss) \
                = criterion(output, in_feature, out_feature_gt,
                            category_weight, confidence_weight, class_weight)
            # if class_loss > 1000 :
            #     print('loss function1')
            #     loss = category_loss + confidence_loss + class_loss + height_loss
            # elif (instance_x_loss + instance_y_loss + heading_x_loss + heading_y_loss) /4.0 > 2000 :
            #     print('loss function2')
            #     loss = category_loss + confidence_loss + class_loss + (instance_x_loss + instance_y_loss + heading_x_loss + heading_y_loss) * 0.01 + height_loss
            # elif (instance_x_loss + instance_y_loss + heading_x_loss + heading_y_loss) /4.0 > 1000 :
            #     print('loss function3')
            #     loss = category_loss + confidence_loss + class_loss + (instance_x_loss + instance_y_loss + heading_x_loss + heading_y_loss) * 0.1 + height_loss
            # else :
            #     print('loss function4')
            loss = category_loss + confidence_loss + class_loss \
                + (instance_x_loss + instance_y_loss
                   + heading_x_loss + heading_y_loss) * 1.0 + height_loss
            # category_loss, confidence_loss, class_loss, instance_loss, heading_loss, height_loss\
            #     = criterion(output, in_feature, out_feature_gt, category_weight, confidence_weight, class_weight)
            # loss = category_loss + confidence_loss + class_loss + instance_loss + heading_loss + height_loss
            # loss = class_loss + instance_loss + heading_loss + height_loss
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_for_record = category_loss + confidence_loss \
                + class_loss + instance_x_loss + instance_y_loss \
                + heading_x_loss + heading_y_loss + height_loss
            iter_loss = loss_for_record.item()
            loss_sum += iter_loss
            category_loss_sum += category_loss.item()
            confidence_loss_sum += confidence_loss.item()
            class_loss_sum += class_loss.item()
            instance_x_loss_sum += instance_x_loss.item()
            instance_y_loss_sum += instance_y_loss.item()
            heading_x_loss_sum += heading_x_loss.item()
            heading_y_loss_sum += heading_y_loss.item()
            height_loss_sum += height_loss.item()

            # visuzlize
            category_gt_img \
                = out_feature_gt[0, 0:1, ...].cpu().detach().numpy().copy()
            confidence_gt_img \
                = out_feature_gt[0, 3:4, ...].cpu().detach().numpy().copy()
            category_img = self.get_category_or_confidence_image(
                output[0, 0:1, :, :], thresh=0.3)
            confidence_img = self.get_category_or_confidence_image(
                output[0, 3:4, :, :], thresh=0.3)
            label_img = self.get_class_image(out_feature_gt[0, 4:10, ...])
            pred_class_img = self.get_class_image(output[0, 4:10, ...])
            in_feature_img \
                = in_feature[0,
                             self.non_empty_channle:self.non_empty_channle + 1,
                             ...].cpu().detach().numpy().copy()
            in_feature_img[in_feature_img > 0] = 255

            # if np.mod(self.epo, 1) == 0:
            if np.mod(index, 1000) == 0:
                arrow_gt_image = self.get_arrow_image(
                    in_feature[0, ...].cpu().detach(
                    ).numpy().transpose(1, 2, 0),
                    out_feature_gt[0, ...].cpu().detach().numpy().transpose(1, 2, 0))

                arrow_image = self.get_arrow_image(
                    in_feature[0, ...].cpu().detach(
                    ).numpy().transpose(1, 2, 0),
                    output[0, ...].cpu().detach().numpy().transpose(1, 2, 0))

            if np.mod(index, self.vis_interval) == 0:
                print('epoch {}, {}/{}, {}_loss is {}'.format(
                    self.epo,
                    index,
                    len(dataloader),
                    mode,
                    iter_loss))

                self.vis.images(in_feature_img,
                                win='{} in_feature'.format(mode),
                                opts=dict(
                                    title='train in_feature'))
                self.vis.images([category_gt_img, category_img],
                                win='{}_category'.format(mode),
                                opts=dict(
                                    title='{} category(GT, Pred)'.format(mode)))
                self.vis.images([confidence_gt_img, confidence_img],
                                win='{}_confidence'.format(mode),
                                opts=dict(
                                    title='{} confidence(GT, Pred)'.format(mode)))
                self.vis.images([label_img, pred_class_img],
                                win='{}_class'.format(mode),
                                opts=dict(
                                    title='{} class pred(GT, Pred)'.format(mode)))
                if np.mod(index, 1000) == 0:
                    self.vis.images([arrow_gt_image, arrow_image],
                                    win='{} arrow'.format(mode),
                                    opts=dict(
                                        title='arrow'))

            if mode == 'train':
                if index == self.train_data_num - 1:
                    print("Finish training {} data.".format(index))
                    break
            elif mode == 'val':
                if index == self.val_data_num - 1:
                    print("Finish validating {} data".format(index))
                    break

        if len(dataloader) > 0:
            avg_loss = loss_sum / len(dataloader)
            avg_confidence_loss = confidence_loss_sum / len(dataloader)
            avg_category_loss = category_loss_sum / len(dataloader)
            avg_class_loss = class_loss_sum / len(dataloader)
            avg_instance_x_loss = instance_x_loss_sum / len(dataloader)
            avg_instance_y_loss = instance_y_loss_sum / len(dataloader)
            avg_heading_x_loss = heading_x_loss_sum / len(dataloader)
            avg_heading_y_loss = heading_y_loss_sum / len(dataloader)
            avg_height_loss = height_loss_sum / len(dataloader)
        else:
            avg_loss = loss_sum
            avg_confidence_loss = confidence_loss_sum
            avg_category_loss = category_loss_sum
            avg_class_loss = class_loss_sum
            avg_instance_x_loss = instance_x_loss_sum
            avg_instance_y_loss = instance_y_loss_sum
            avg_heading_x_loss = heading_x_loss_sum
            avg_heading_y_loss = heading_y_loss_sum
            avg_height_loss = height_loss_sum

        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_loss]),
                      win='loss', name='{}_loss'.format(mode), update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_category_loss]),
                      win='loss', name='category_{}_loss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_confidence_loss]),
                      win='loss', name='confidence_{}_loss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_class_loss]),
                      win='loss', name='class_{}_loss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_instance_x_loss]),
                      win='loss', name='instance_x_{}_loss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_instance_y_loss]),
                      win='loss', name='instance_y_{}_loss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_heading_x_loss]),
                      win='loss', name='heading_x_{}_foss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_heading_y_loss]),
                      win='loss', name='heading_y_{}_loss'.format(mode),
                      update='append')
        self.vis.line(X=np.array([self.epo]),
                      Y=np.array([avg_height_loss]),
                      win='loss', name='height_{}_loss'.format(mode),
                      update='append')

        if mode == 'val':
            if np.mod(self.epo, self.save_model_interval) == 0:
                torch.save(
                    self.model.state_dict(),
                    'checkpoints/bcnn_latestmodel_' + self.time_now + '.pt')
            if self.best_loss > loss_sum / len(dataloader):
                print('update best model {} -> {}'.format(
                    self.best_loss, loss_sum / len(dataloader)))
                self.best_loss = loss_sum / len(dataloader)
                torch.save(
                    self.model.state_dict(),
                    'checkpoints/bcnn_bestmodel_' + self.time_now + '.pt')

    def train(self):
        for self.epo in range(self.max_epoch):
            self.step('train')
            self.step('val')
            self.scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        default='/media/kosuke/SANDISK/nusc/mini-6c-672')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='max epoch',
                        default=1)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',
                        default=1000000)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model',
                        default='checkpoints/bcnn_latestmodel_20200619_1526.pt')
    parser.add_argument('--train_data_num', '-tn', type=int,
                        help='How much data to use for training',
                        default=1000000)
    parser.add_argument('--val_data_num', '-vn', type=int,
                        help='How much data to use for testing',
                        default=1000000)
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=672)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=672)
    parser.add_argument('--use_constant_feature', type=int,
                        help='Whether to use constant feature',
                        default=0)
    parser.add_argument('--use_intensity_feature', type=int,
                        help='Whether to use intensity feature',
                        default=1)
    args = parser.parse_args()

    trainer = Trainer(data_path=args.data_path,
                      batch_size=args.batch_size,
                      max_epoch=args.max_epoch,
                      pretrained_model=args.pretrained_model,
                      train_data_num=args.train_data_num,
                      val_data_num=args.val_data_num,
                      width=args.width,
                      height=args.height,
                      use_constant_feature=args.use_constant_feature,
                      use_intensity_feature=args.use_intensity_feature)
    trainer.train()
