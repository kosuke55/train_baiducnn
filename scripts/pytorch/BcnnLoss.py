#!/usr/bin/env python3
# coding: utf-8

import torch
from torch.nn import Module

import math


class BcnnLoss(Module):
    def __init__(self):
        super(BcnnLoss, self).__init__()

    def forward(self, output, input, target, category_weight, confidence_weight, class_weight):
        gamma = 2.0  # focal
        alpha = 0.7  # class weight
        sigma = 1.0  # huber
        beta = 1.0
        objects_class_weight = target[:, 4:9, ...] * class_weight
        object_weight = objects_class_weight[:, 0, ...]+objects_class_weight[:, 1, ...] + \
            objects_class_weight[:, 2, ...]+objects_class_weight[:,
                                                                 3, ...]+objects_class_weight[:, 4, ...]
        category_diff = output[:, 0, ...] - target[:, 0, ...]
        category_loss = torch.sum(category_weight * object_weight * torch.where(category_diff <= sigma, 0.5 * (
            category_diff**2), sigma * torch.abs(category_diff) - 0.5 * (sigma ** 2)) *
            (1.0 + beta * input[:, 2, ...])) * 0.05

        confidence_diff = output[:, 3, ...] - target[:, 3, ...]
        confidence_loss = torch.sum(confidence_weight * object_weight*torch.where(confidence_diff <= sigma, 0.5 * (
            confidence_diff**2), sigma * torch.abs(confidence_diff) - 0.5 * (sigma ** 2)) *
            (1.0 + beta * input[:, 2, ...])) * 0.005

        # class_loss = -torch.sum(class_weight * ((1.0 - output[:, 4:9, ...]) ** gamma) * (target[:, 4:9, ...] * torch.log(
        #     output[:, 4:9, ...] + 1e-7)) * (1.0 + beta * input[:, 2:3, ...])) * 0.01
        class_loss = -torch.sum(class_weight *
                                (((1.0 - output[:, 4:9, ...]) ** gamma) *
                                 (target[:, 4:9, ...] * torch.log(output[:, 4:9, ...] + 1e-7)) * alpha +
                                 ((output[:, 4:9, ...]) ** gamma) *
                                 ((1.0 - target[:, 4:9, ...]) * torch.log(1.0 - output[:, 4:9, ...] + 1e-7)) * (1.0 - alpha))
                                * (1.0 + beta * input[:, 2:3, ...])) * 0.015

        instance_x_diff = output[:, 1, ...] - target[:, 1, ...]
        instance_y_diff = output[:, 2, ...] - target[:, 2, ...]
        instance_x_loss = torch.sum(
            torch.where(instance_x_diff <= sigma, 0.5 * (instance_x_diff**2), 
            sigma * torch.abs(instance_x_diff) - 0.5 * (sigma ** 2)) * target[:, 0, ...]) * 0.6
        instance_y_loss = torch.sum(
            torch.where(instance_y_diff <= sigma, 0.5 * (instance_y_diff**2), 
            sigma * torch.abs(instance_y_diff) - 0.5 * (sigma ** 2)) * target[:, 0, ...]) * 0.6
        # instance_x_loss = torch.sum(
        #     instance_x_diff**2 * (1.0 + beta * input[:, 2:3, ...])) * 0.0045
        # instance_y_loss = torch.sum(
        #     instance_y_diff**2 * (1.0 + beta * input[:, 2:3, ...])) * 0.0045

        heading_x_diff = output[:, 9, ...] - target[:, 9, ...]
        heading_y_diff = output[:, 10, ...] - target[:, 10, ...]
        heading_x_loss = torch.sum(
            torch.where(heading_x_diff <= sigma, 0.5 * (heading_x_diff**2), 
            sigma * torch.abs(heading_x_diff) - 0.5 * (sigma ** 2)) * target[:, 0, ...]) * 0.6
        heading_y_loss = torch.sum(
            torch.where(heading_y_diff <= sigma, 0.5 * (heading_y_diff**2), 
            sigma * torch.abs(heading_y_diff) - 0.5 * (sigma ** 2)) * target[:, 0, ...]) * 0.6

        # heading_orig_diff = torch.abs(torch.acos((torch.cos(output[:, 10, ...]) * torch.cos(target[:, 10, ...]) + torch.sin(
        #     output[:, 10, ...]) * torch.sin(target[:, 5, ...])).clamp(min=-1 + 1e-7, max=1 - 1e-7)))
        # heading_inv_diff = torch.abs(torch.acos((torch.cos(output[:, 10, ...]) * torch.cos(target[:, 10, ...] + math.pi) + torch.sin(output[:, 10, ...]) * torch.sin(target[:, 10, ...] + math.pi)).clamp(min=-1 + 1e-7, max=1 - 1e-7)))
        # heading_diff = torch.min(heading_orig_diff, heading_inv_diff)
        # heading_loss = torch.sum(
        #     torch.abs(heading_diff) * (1.0 + beta * input[:, 2:3, ...])) * 0.001

        height_diff = output[:, 11, ...] - target[:, 11, ...]
        height_loss = torch.sum(
            torch.where(height_diff <= sigma, 0.5 * (height_diff**2),
                        sigma * torch.abs(height_diff) - 0.5 * (sigma ** 2)) * target[:, 0, ...]) * 0.02

        print("category_loss", float(category_loss))
        print("confidence_loss", float(confidence_loss))
        print("class_loss", float(class_loss))
        # print("instace_loss ", float(instance_loss))
        print("instace_x_loss ", float(instance_x_loss))
        print("instace_y_loss ", float(instance_y_loss))
        print("heading_x_loss ", float(heading_x_loss))
        print("heading_y_loss ", float(heading_y_loss))
        print("height_loss ", float(height_loss))

        # return category_loss, confidence_loss, class_loss, instance_loss, heading_loss, height_loss
        return category_loss, confidence_loss, class_loss, instance_x_loss, instance_y_loss, heading_x_loss, heading_y_loss, height_loss
