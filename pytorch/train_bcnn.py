from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import visdom
from NuscData import test_dataloader, train_dataloader
from weighted_mse import wmse
from BCNN import BCNN


def train(epo_num, pretrained_model):
    best_loss = 1e10
    vis = visdom.Visdom()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bcnn_model = BCNN().to(device)
    bcnn_model.load_state_dict(torch.load(pretrained_model))
    # bcnn_model = torch.load(pretrained_model).to(device)
    bcnn_model.eval()

    # criterion = nn.BCELoss().to(device)
    # optimizer = optim.SGD(bcnn_model.parameters(), lr=1e-3, momentum=0.7)
    optimizer = optim.SGD(bcnn_model.parameters(), lr=1e-4)

    all_train_iter_loss = []
    all_test_iter_loss = []

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        train_loss = 0
        bcnn_model.train()
        for index, (nusc, nusc_msk) in enumerate(train_dataloader):
            pos_weight = nusc_msk.detach().numpy().copy()
            pos_weight = pos_weight[0]

            zeroidx = np.where(pos_weight == 0)
            nonzeroidx = np.where(pos_weight != 0)
            pos_weight[zeroidx] = 0.5
            pos_weight[nonzeroidx] = 1.
            pos_weight = torch.from_numpy(pos_weight)
            pos_weight = pos_weight.to(device)
            # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
            # criterion = nn.MSELoss()
            criterion = wmse().to(device)
            # criterion = WMSELoss().to(device)
            nusc = nusc.to(device)
            nusc_msk = nusc_msk.to(device)
            optimizer.zero_grad()
            # output = model(nusc)
            output = bcnn_model(nusc)  # output is confidence
            output = output[2]
            output = torch.sigmoid(output)

            loss = criterion(output, nusc_msk, pos_weight)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy()
            output_np = output_np.transpose(1, 2, 0)
            output_img = np.zeros((640, 640, 1), dtype=np.uint8)
            # conf_idx = np.where(output_np[..., 0] > output_np[..., 0].mean())
            conf_idx = np.where(output_np[..., 0] > 0.5)
            output_img[conf_idx] = 255
            output_img = output_img.transpose(2, 0, 1)
            nusc_msk_img = nusc_msk.cpu().detach().numpy().copy()
            # print(nusc.shape)
            nusc_img = nusc[:, 7, ...].cpu().detach().numpy().copy()

            if np.mod(index, 15) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo,
                                                                index,
                                                                len(train_dataloader),
                                                                iter_loss))
                # vis.close()
                vis.images(nusc_img,
                           win='nusc_img',
                           opts=dict(title='nusc input'))
                vis.images(output_img,
                           win='train_pred',
                           opts=dict(title='train prediction'))
                vis.images(nusc_msk_img,
                           win='train_label',
                           opts=dict(title='train_label'))
                vis.line(all_train_iter_loss,
                         win='train_iter_loss',
                         opts=dict(title='train iter loss'))

        test_loss = 0
        bcnn_model.eval()
        with torch.no_grad():
            for index, (nusc, nusc_msk) in enumerate(test_dataloader):

                nusc = nusc.to(device)
                nusc_msk = nusc_msk.to(device)

                optimizer.zero_grad()
                output = bcnn_model(nusc)[2]
                output = torch.sigmoid(output)
                # loss = criterion(pos_weight, output, nusc_msk)
                loss = criterion(output, nusc_msk, pos_weight)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy()
                output_np = output_np.transpose(1, 2, 0)
                output_img = np.zeros((640, 640, 1), dtype=np.uint8)
                # conf_idx = np.where(output_np[..., 0] > output_np[..., 0].mean())
                conf_idx = np.where(output_np[..., 0] > 0.5)
                output_img[conf_idx] = 255
                output_img = output_img.transpose(2, 0, 1)

                nusc_msk_img = nusc_msk.cpu().detach().numpy().copy()

                if np.mod(index, 15) == 0:
                    # print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    vis.images(output_img, win='test_pred', opts=dict(
                        title='test prediction'))
                    vis.images(nusc_msk_img,
                               win='test_label', opts=dict(title='test_label'))

                    vis.line(all_test_iter_loss, win='test_iter_loss',
                             opts=dict(title='test iter loss'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch train loss = %f, epoch test loss = %f, best_loss = %f, %s'
              % (train_loss/len(train_dataloader),
                 test_loss/len(test_dataloader),
                 best_loss,
                 time_str))
        if(best_loss > test_loss/len(test_dataloader)):
            print('update best model {} -> {}'.format(
                best_loss, test_loss/len(test_dataloader)))
            best_loss = test_loss/len(test_dataloader)
            torch.save(bcnn_model.state_dict(), 'checkpoints/bcnn_bestmodel.pt')


if __name__ == "__main__":
    # pretrained_model = "/home/kosuke/develop/pytorch-FCN-easiest-demo/checkpoints/bcnn_bestmodel_mini_844.pt"
    pretrained_model = "/home/kosuke/develop/pytorch-FCN-easiest-demo/checkpoints/bcnn_bestmodel.pt"
    train(epo_num=100000, pretrained_model=pretrained_model)

