#!/usr/bin/env python
# coding: utf-8

import argparse

import numpy as np
import torch
import torch.optim as optim
import visdom
from datetime import datetime

from BCNN import BCNN
from BcnnLoss import BcnnLoss
from NuscData import load_dataset


def train(data_path, max_epoch, pretrained_model,
          train_data_num, test_data_num,
          width=640, height=640):
    train_dataloader, test_dataloader = load_dataset(data_path)
    now = datetime.now().strftime('%Y%m%d_%H%M')
    best_loss = 1e10
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bcnn_model = BCNN(in_channels=8, n_class=6).to(device)
    bcnn_model.load_state_dict(torch.load(pretrained_model))
    bcnn_model.eval()

    transfer_learning = False
    if transfer_learning:
        params_to_update = []
        update_param_names = ["deconv0.weight", "deconv0.bias"]
        for name, param in bcnn_model.named_parameters():
            if name in update_param_names:
                param.requires_grad = True
                params_to_update.append(param)
                print(name)
            else:
                param.requires_grad = False
        print("-----------")
        print(params_to_update)
        optimizer = optim.SGD(params=params_to_update, lr=1e-5, momentum=0.9)
    else:
        optimizer = optim.SGD(bcnn_model.parameters(), lr=1e-8, momentum=0.9)

    prev_time = datetime.now()
    for epo in range(max_epoch):
        train_loss = 0
        bcnn_model.train()
        for index, (nusc, nusc_msk) in enumerate(train_dataloader):
            nusc_msk_np = nusc_msk.detach().numpy().copy()
            pos_weight = nusc_msk.detach().numpy().copy()
            pos_weight = pos_weight[0, :, :, 3]

            zeroidx = np.where(pos_weight == 0)
            nonzeroidx = np.where(pos_weight != 0)
            pos_weight[zeroidx] = 0.1
            pos_weight[nonzeroidx] = 1.
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device)
            criterion = BcnnLoss().to(device)
            nusc = nusc.to(device)
            nusc_msk = nusc_msk.to(device)

            optimizer.zero_grad()
            output = bcnn_model(nusc)

            confidence = output[:, 3, :, :]
            pred_class = output[:, 4:10, :, :]

            print(nusc_msk.shape)
            print(output.shape)
            loss = criterion(
                output, nusc_msk.transpose(1, 3).transpose(2, 3), pos_weight)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            confidence_np = confidence.cpu().detach().numpy().copy()
            confidence_np = confidence_np.transpose(1, 2, 0)
            confidence_img = np.zeros((width, height, 1), dtype=np.uint8)

            # conf_idx = np.where(confidence_np[..., 0] > 0.5)
            conf_idx = np.where(
                confidence_np[..., 0] > confidence_np[..., 0].mean())
            confidence_img[conf_idx] = 1.0
            confidence_img = confidence_img.transpose(2, 0, 1)

            # draw pred class
            pred_class_np = pred_class.cpu().detach().numpy().copy()
            pred_class_np = np.argmax(pred_class_np, axis=1)
            pred_class_np = pred_class_np.transpose(1, 2, 0)
            car_idx = np.where(pred_class_np[:, :, 0] == 1)
            bus_idx = np.where(pred_class_np[:, :, 0] == 2)
            bike_idx = np.where(pred_class_np[:, :, 0] == 3)
            human_idx = np.where(pred_class_np[:, :, 0] == 4)
            pred_class_img = np.zeros((width, height, 3))
            pred_class_img[car_idx] = [255, 0, 0]
            pred_class_img[bus_idx] = [0, 255, 0]
            pred_class_img[bike_idx] = [0, 0, 255]
            pred_class_img[human_idx] = [0, 255, 255]
            pred_class_img = pred_class_img.transpose(2, 0, 1)

            # draw label image
            true_label_np = nusc_msk_np[..., 4:10]
            true_label_np = np.argmax(true_label_np, axis=3)
            true_label_np = true_label_np.transpose(1, 2, 0)
            car_idx = np.where(true_label_np[:, :, 0] == 1)
            bus_idx = np.where(true_label_np[:, :, 0] == 2)
            bike_idx = np.where(true_label_np[:, :, 0] == 3)
            human_idx = np.where(true_label_np[:, :, 0] == 4)
            label_img = np.zeros((width, height, 3))
            label_img[car_idx] = [255, 0, 0]
            label_img[bus_idx] = [0, 255, 0]
            label_img[bike_idx] = [0, 0, 255]
            label_img[human_idx] = [0, 255, 255]
            label_img = label_img.transpose(2, 0, 1)

            nusc_msk_img = nusc_msk[..., 0].cpu().detach().numpy().copy()
            nusc_img = nusc[:, 7, ...].cpu().detach().numpy().copy()

            if np.mod(index, 1) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(
                    epo,
                    index,
                    len(train_dataloader),
                    iter_loss))

                vis.images(nusc_img,
                           win='nusc_input',
                           opts=dict(
                               title='nusc input'))
                vis.images([nusc_msk_img, confidence_img],
                           win='train_confidencena',
                           opts=dict(
                               title='train confidence(GT, Pred)'))
                vis.images([label_img, pred_class_img],
                           win='train_class',
                           opts=dict(
                               title='train class pred(GT, Pred)'))

            if index == train_data_num - 1:
                print("Finish train {} data. So start test.".format(index))
                break

        if len(train_dataloader) > 0:
            avg_train_loss = train_loss / len(train_dataloader)
        else:
            avg_train_loss = train_loss

        test_loss = 0
        bcnn_model.eval()
        with torch.no_grad():
            for index, (nusc, nusc_msk) in enumerate(test_dataloader):

                nusc_msk_np = nusc_msk.detach().numpy().copy()  # HWC
                nusc = nusc.to(device)
                nusc_msk = nusc_msk.to(device)

                optimizer.zero_grad()
                output = bcnn_model(nusc)

                confidence = output[:, 3, :, :]
                pred_class = output[:, 4:10, :, :]

                loss = criterion(
                    output, nusc_msk.transpose(1, 3).transpose(2, 3),
                    pos_weight)

                iter_loss = loss.item()
                test_loss += iter_loss

                confidence_np = confidence.cpu().detach().numpy().copy()
                confidence_np = confidence_np.transpose(1, 2, 0)

                confidence_img = np.zeros((width, height, 1), dtype=np.uint8)
                # conf_idx = np.where(confidence_np[..., 0] > 0.5)
                conf_idx = np.where(
                    confidence_np[..., 0] > confidence_np[..., 0].mean())

                confidence_img[conf_idx] = 1.
                confidence_img = confidence_img.transpose(2, 0, 1)

                # draw pred class
                pred_class_np = pred_class.cpu().detach().numpy().copy()
                pred_class_np = np.argmax(pred_class_np, axis=1)
                pred_class_np = pred_class_np.transpose(1, 2, 0)
                car_idx = np.where(pred_class_np[:, :, 0] == 1)
                bus_idx = np.where(pred_class_np[:, :, 0] == 2)
                bike_idx = np.where(pred_class_np[:, :, 0] == 3)
                human_idx = np.where(pred_class_np[:, :, 0] == 4)
                pred_class_img = np.zeros((width, height, 3))
                pred_class_img[car_idx] = [255, 0, 0]
                pred_class_img[bus_idx] = [0, 255, 0]
                pred_class_img[bike_idx] = [0, 0, 255]
                pred_class_img[human_idx] = [0, 255, 255]
                pred_class_img = pred_class_img.transpose(2, 0, 1)

                # draw label image
                true_label_np = nusc_msk_np[..., 4:10]
                true_label_np = np.argmax(true_label_np, axis=3)
                true_label_np = true_label_np.transpose(1, 2, 0)
                car_idx = np.where(true_label_np[:, :, 0] == 1)
                bus_idx = np.where(true_label_np[:, :, 0] == 2)
                bike_idx = np.where(true_label_np[:, :, 0] == 3)
                human_idx = np.where(true_label_np[:, :, 0] == 4)
                label_img = np.zeros((width, height, 3))
                label_img[car_idx] = [255, 0, 0]
                label_img[bus_idx] = [0, 255, 0]
                label_img[bike_idx] = [0, 0, 255]
                label_img[human_idx] = [0, 255, 255]
                label_img = label_img.transpose(2, 0, 1)

                nusc_msk_img = nusc_msk[..., 0].cpu().detach().numpy().copy()

                if np.mod(index, 25) == 0:
                    vis.images([nusc_msk_img, confidence_img],
                               win='test_confidencena',
                               opts=dict(
                                   title='test confidence(GT, Pred)'))
                    vis.images([label_img, pred_class_img],
                               win='test_class',
                               opts=dict(
                                   title='test class pred(GT, Pred)'))
                if index == test_data_num - 1:
                    print("Finish test {} data".format(index))
                    break

            avg_test_loss = test_loss / len(test_dataloader)

        vis.line(X=np.array([epo]), Y=np.array([avg_train_loss]), win='loss',
                 name='avg_train_loss', update='append')
        vis.line(X=np.array([epo]), Y=np.array([avg_test_loss]), win='loss',
                 name='avg_test_loss', update='append')

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        if np.mod(epo, 1) == 0:
            torch.save(bcnn_model.state_dict(),
                       'checkpoints/bcnn_latestmodel_' + now + '.pt')
        print('epoch train loss = %f, epoch test loss = %f, best_loss = %f, %s'
              % (train_loss / len(train_dataloader),
                 test_loss / len(test_dataloader),
                 best_loss,
                 time_str))
        if best_loss > test_loss / len(test_dataloader):
            print('update best model {} -> {}'.format(
                best_loss, test_loss / len(test_dataloader)))
            best_loss = test_loss / len(test_dataloader)
            torch.save(bcnn_model.state_dict(),
                       'checkpoints/bcnn_bestmodel_' + now + '.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_path', '-dp', type=str,
                        help='Training data path',
                        default='/media/kosuke/HD-PNFU3/0413/nusc/mini-0421/')
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',
                        default=1000000)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model',
                        default='checkpoints/mini_instance.pt')
    parser.add_argument('--train_data_num', '-tr', type=int,
                        help='How much data to use for training',
                        default=1000000)
    parser.add_argument('--test_data_num', '-te', type=int,
                        help='How much data to use for testing',
                        default=1000000)

    args = parser.parse_args()
    train(data_path=args.data_path,
          max_epoch=args.max_epoch,
          pretrained_model=args.pretrained_model,
          train_data_num=args.train_data_num,
          test_data_num=args.test_data_num)
