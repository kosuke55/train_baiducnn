#!/usr/bin/env python3
# coding: utf-8

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def load_dataset(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()])
    nusc = NuscDataset(data_path, transform)

    train_size = int(0.9 * len(nusc))
    test_size = len(nusc) - train_size
    train_dataset, test_dataset = random_split(nusc, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader


def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


class NuscDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path, 'in_feature')))

    def __getitem__(self, idx):
        data_name = os.listdir(os.path.join(self.data_path, 'in_feature'))[idx]

        in_feature = np.load(
            os.path.join(self.data_path, "in_feature/", data_name))
        in_feature = in_feature.astype(np.float32)

        out_feature = np.load(
            os.path.join(self.data_path, "out_feature/", data_name))
        one_hot_class = onehot(out_feature[..., 4].astype(np.int8), 5)

        out_feature = np.concatenate(
            [out_feature[..., 0:4],
             one_hot_class, out_feature[..., 5:]], 2)

        out_feature = out_feature.astype(np.float32)

        if self.transform:
            in_feature = self.transform(in_feature)
            out_feature = self.transform(out_feature)

        return in_feature, out_feature
