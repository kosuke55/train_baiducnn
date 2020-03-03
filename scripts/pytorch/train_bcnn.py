from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import visdom
from NuscData import test_dataloader, train_dataloader
from BcnnLoss import bcnn_loss
from BCNN import BCNN


def train(epo_num, pretrained_model):
    best_loss = 1e10
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bcnn_model = BCNN().to(device)
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
        optimizer = optim.SGD(bcnn_model.parameters(), lr=1e-6, momentum=0.9)

    # optimizer = torch.optim.Adam(bcnn_model.parameters(), lr=1e-6)
    # optimizer = optim.SGD(bcnn_model.parameters(), lr=1e-4)

    prev_time = datetime.now()
    for epo in range(epo_num):
        train_loss = 0
        bcnn_model.train()
        for index, (nusc, nusc_msk) in enumerate(train_dataloader):
            nusc_msk_np = nusc_msk.detach().numpy().copy()  # HWC
            pos_weight = nusc_msk.detach().numpy().copy()  # NHWC
            pos_weight = pos_weight[0, :, :, 0]

            zeroidx = np.where(pos_weight == 0)
            nonzeroidx = np.where(pos_weight != 0)
            pos_weight[zeroidx] = 0.5
            pos_weight[nonzeroidx] = 1.
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device)  # 640 640
            criterion = bcnn_loss().to(device)
            nusc = nusc.to(device)
            nusc_msk = nusc_msk.to(device)  # 1 640 640 6

            optimizer.zero_grad()
            output = bcnn_model(nusc)  # 1 6 640 640

            confidence = output[:, 0, :, :]
            pred_class = output[:, 1:6, :, :]

            # 6 640 640, 1 640 640 6, 640 640
            loss = criterion(
                output, nusc_msk.transpose(1, 3).transpose(2, 3), pos_weight)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()

            confidence_np = confidence.cpu().detach().numpy().copy()
            confidence_np = confidence_np.transpose(1, 2, 0)  # 640 640 1
            confidence_img = np.zeros((640, 640, 1), dtype=np.uint8)
            # conf_idx = np.where(
            #     confidence_np[..., 0] > confidence_np[..., 0].mean())
            conf_idx = np.where(confidence_np[..., 0] > 0.5)
            confidence_img[conf_idx] = 255
            confidence_img = confidence_img.transpose(2, 0, 1)  # 1 640 640

            # draw pred class
            pred_class_np = pred_class.cpu().detach().numpy().copy()
            pred_class_np = np.argmax(pred_class_np, axis=1)
            pred_class_np = pred_class_np.transpose(1, 2, 0)
            car_idx = np.where(pred_class_np[:, :, 0] == 1)
            bus_idx = np.where(pred_class_np[:, :, 0] == 2)
            bike_idx = np.where(pred_class_np[:, :, 0] == 3)
            human_idx = np.where(pred_class_np[:, :, 0] == 4)
            pred_class_img = np.zeros((640, 640, 3))
            pred_class_img[car_idx] = [255, 0, 0]
            pred_class_img[bus_idx] = [0, 255, 0]
            pred_class_img[bike_idx] = [0, 0, 255]
            pred_class_img[human_idx] = [0, 255, 255]
            pred_class_img = pred_class_img.transpose(2, 0, 1)

            # draw label image
            true_label_np = nusc_msk_np[..., 1:6]
            true_label_np = np.argmax(true_label_np, axis=3)
            true_label_np = true_label_np.transpose(1, 2, 0)
            car_idx = np.where(true_label_np[:, :, 0] == 1)
            bus_idx = np.where(true_label_np[:, :, 0] == 2)
            bike_idx = np.where(true_label_np[:, :, 0] == 3)
            human_idx = np.where(true_label_np[:, :, 0] == 4)
            label_img = np.zeros((640, 640, 3))
            label_img[car_idx] = [255, 0, 0]
            label_img[bus_idx] = [0, 255, 0]
            label_img[bike_idx] = [0, 0, 255]
            label_img[human_idx] = [0, 255, 255]
            label_img = label_img.transpose(2, 0, 1)

            nusc_msk_img = nusc_msk[..., 0].cpu().detach().numpy().copy()
            nusc_img = nusc[:, 7, ...].cpu().detach().numpy().copy()
            if np.mod(index, 25) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(
                    epo,
                    index,
                    len(train_dataloader),
                    iter_loss))
                vis.images(nusc_img,
                           win='nusc_img',
                           opts=dict(
                               title='nusc input'))
                vis.images(confidence_img,
                           win='train_pred',
                           opts=dict(
                               title='train prediction'))
                vis.images(nusc_msk_img,
                           win='train_label',
                           opts=dict(
                               title='train_label'))
                vis.images(pred_class_img,
                           win='train_class_pred',
                           opts=dict(
                               title='train class prediction'))
                vis.images(label_img,
                           win='train_true_label',
                           opts=dict(
                               title='true label'))

        avg_train_loss = train_loss / len(train_dataloader)

        test_loss = 0
        bcnn_model.eval()
        with torch.no_grad():
            for index, (nusc, nusc_msk) in enumerate(test_dataloader):

                nusc = nusc.to(device)
                nusc_msk = nusc_msk.to(device)

                optimizer.zero_grad()
                output = bcnn_model(nusc)

                confidence = output[:, 0, :, :]
                pred_class = output[:, 1:6, :, :]

                loss = criterion(
                    output, nusc_msk.transpose(1, 3).transpose(2, 3),
                    pos_weight)  # 1 6 640 640, 1 640 640 6, 640 640
                iter_loss = loss.item()
                test_loss += iter_loss

                confidence_np = confidence.cpu().detach().numpy().copy()
                confidence_np = confidence_np.transpose(1, 2, 0)  # 640 640 1
                confidence_img = np.zeros((640, 640, 1), dtype=np.uint8)
                # conf_idx = np.where(
                #     confidence_np[..., 0] > confidence_np[..., 0].mean())
                conf_idx = np.where(confidence_np[..., 0] > 0.5)
                confidence_img[conf_idx] = 255
                confidence_img = confidence_img.transpose(2, 0, 1)  # 1 640 640

                # draw pred class
                pred_class_np = pred_class.cpu().detach().numpy().copy()
                pred_class_np = np.argmax(pred_class_np, axis=1)
                pred_class_np = pred_class_np.transpose(1, 2, 0)
                car_idx = np.where(pred_class_np[:, :, 0] == 1)
                bus_idx = np.where(pred_class_np[:, :, 0] == 2)
                bike_idx = np.where(pred_class_np[:, :, 0] == 3)
                human_idx = np.where(pred_class_np[:, :, 0] == 4)
                pred_class_img = np.zeros((640, 640, 3))
                pred_class_img[car_idx] = [255, 0, 0]
                pred_class_img[bus_idx] = [0, 255, 0]
                pred_class_img[bike_idx] = [0, 0, 255]
                pred_class_img[human_idx] = [0, 255, 255]
                pred_class_img = pred_class_img.transpose(2, 0, 1)

                # draw label image
                true_label_np = nusc_msk_np[..., 1:6]
                true_label_np = np.argmax(true_label_np, axis=3)
                true_label_np = true_label_np.transpose(1, 2, 0)
                car_idx = np.where(true_label_np[:, :, 0] == 1)
                bus_idx = np.where(true_label_np[:, :, 0] == 2)
                bike_idx = np.where(true_label_np[:, :, 0] == 3)
                human_idx = np.where(true_label_np[:, :, 0] == 4)
                label_img = np.zeros((640, 640, 3))
                label_img[car_idx] = [255, 0, 0]
                label_img[bus_idx] = [0, 255, 0]
                label_img[bike_idx] = [0, 0, 255]
                label_img[human_idx] = [0, 255, 255]
                label_img = label_img.transpose(2, 0, 1)

                nusc_msk_img = nusc_msk[..., 0].cpu().detach().numpy().copy()
                nusc_img = nusc[:, 7, ...].cpu().detach().numpy().copy()
                if np.mod(index, 25) == 0:
                    vis.images(confidence_img, win='test_pred', opts=dict(
                        title='test prediction'))
                    vis.images(nusc_msk_img,
                               win='test_label',
                               opts=dict(
                                   title='test_label'))
                    vis.images(pred_class_img,
                               win='train_class_pred',
                               opts=dict(
                                   title='train class prediction'))
                    vis.images(label_img,
                               win='train_true_label',
                               opts=dict(
                                   title='true label'))

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

        torch.save(bcnn_model.state_dict(),
                   'checkpoints/bcnn_latestmodel_all_0201.pt')
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
                       'checkpoints/bcnn_bestmodel_all_0201.pt')


if __name__ == "__main__":
    pretrained_model = "checkpoints/bcnn_bestmodel_0125.pt"
    train(epo_num=100000, pretrained_model=pretrained_model)
