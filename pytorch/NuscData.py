import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import sys
OPENCV_PATH \
    = '/home/kosuke/.pyenv/versions/anaconda3-2019.03/lib/python3.7/site-packages'
sys.path = [OPENCV_PATH] + sys.path
print(sys.path)

transform = transforms.Compose([
    transforms.ToTensor()])


DATA_PATH \
    = "/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e/baidu_train_data/all/"


class NuscDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir(DATA_PATH + 'in_feature'))

    def __getitem__(self, idx):
        data_name = os.listdir(DATA_PATH + 'in_feature')[idx]

        in_feature = np.load(DATA_PATH + "in_feature/" + data_name)
        # print(in_feature[..., c][in_feature[..., c] > 0])
        in_feature = in_feature.astype(np.float32)

        out_feature = np.load(DATA_PATH + "out_feature/" + data_name)
        # print(out_feature.shape)
        out_feature = out_feature[..., 0]   # use only confidence
        # out_feature = out_feature.transpose(2, 0, 1)
        out_feature = torch.FloatTensor(out_feature)
        # print(out_feature[out_feature>0])
        # print(out_feature.shape)
        # import pdb
        # pdb.set_trace()

        # c = 1
        # print(in_feature[..., c][in_feature[..., c] > 0])
        # print(in_feature.shape)
        if self.transform:
            in_feature = self.transform(in_feature)
            # print(in_feature.shape)
            # print(in_feature[c][in_feature[c] > 0])

        return in_feature, out_feature


nusc = NuscDataset(transform)

train_size = int(0.9 * len(nusc))
test_size = len(nusc) - train_size
train_dataset, test_dataset = random_split(nusc, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=1)


if __name__ == '__main__':

    for train_batch in train_dataloader:
        pass
        # print(train_batch)

    for test_batch in test_dataloader:
        pass
        # print(test_batch)
