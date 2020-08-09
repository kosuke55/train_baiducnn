#!/usr/bin/env python3
# coding: utf-8

import os

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def load_dataset(data_path, batch_size):
    """Load training and validation dataset.

    Parameters
    ----------
    data_path : str
    batch_size : int

    Returns
    -------
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    """
    transform = transforms.Compose([
        transforms.ToTensor()])
    nusc = NuscDataset(data_path, transform)

    train_size = int(0.9 * len(nusc))
    val_size = len(nusc) - train_size
    train_dataset, val_dataset = random_split(nusc, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_dataloader, val_dataloader


def onehot(data, n=5):
    """convert label to onehot_vector

    Originally implemented in
    https://github.com/yunlongdong/FCN-pytorch/blob/master/onehot.py

    Parameters
    ----------
    data : numpy.ndarray
        np.ndarray with int stored in label
    n : int, optional
        [description], by default 5

    Returns
    -------
    buf numpy.ndarray
        onehot vector of class
    """
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


class NuscDataset(Dataset):
    """Nuscenes dataset

    Parameters
    ----------
    data_path : str
        Path of generated dataset.
    transform : torchvision.transforms.Compose, optional
        Currently it only converts a numpy.ndarray to a torch tensor,
        not really needed., by default None

    """
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
