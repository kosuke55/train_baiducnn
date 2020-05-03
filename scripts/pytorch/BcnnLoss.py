#!/usr/bin/env python3
# coding: utf-8

import torch
from torch.nn import Module


class BcnnLoss(Module):
    def __init__(self):
        super(BcnnLoss, self).__init__()

    def forward(self, output, target, weight):
        category_diff = output[:, 0, ...] - target[:, 0, ...]
        category_loss = torch.sum((weight * category_diff) ** 2)

        confidence_diff = output[:, 3, ...] - target[:, 3, ...]
        confidence_loss = torch.sum((weight * confidence_diff) ** 2)
        class_loss \
            = -torch.sum(
                target[:, 4:10, ...] * torch.log(output[:, 4:10, ...] + 1e-7))

        instance_x_diff = output[:, 1, ...] - target[:, 1, ...]
        instance_y_diff = output[:, 2, ...] - target[:, 2, ...]

        instance_loss = torch.sum(
            (weight * instance_x_diff) ** 2 + (weight * instance_y_diff) ** 2)

        height_diff = output[:, 11, ...] - target[:, 11, ...]
        height_loss = torch.sum((weight * height_diff) ** 2)

        loss = category_loss + confidence_loss + \
            class_loss + instance_loss + height_loss
        print("category_loss", category_loss)
        print("confidence_loss", confidence_loss)
        print("class_loss", class_loss)
        print("instace_loss ", instance_loss)
        print("height_loss ", height_loss)

        return loss
