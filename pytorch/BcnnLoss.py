import torch
import numpy as np
import torch.nn as nn
import torch.nn._reduction as _Reduction
from torch.nn import Module
import warnings


class bcnn_loss(nn.Module):
    def __init__(self):
        super(bcnn_loss, self).__init__()

    def forward(self, output, target, weight):

        confidence_diff = output[:, 0, ...] - target[:, 0, ...]
        confidence_loss = torch.sum((weight * confidence_diff) ** 2)

        class_loss \
            = -torch.sum(
                target[:, 1:6, ...] * torch.log(output[:, 1:6, ...] + 1e-7))

        loss = confidence_loss + 0.01 * class_loss

        return loss


class wmse(nn.Module):
    def __init__(self):
        super(wmse, self).__init__()

    def forward(self, output, target, weight):
        diff = output - target
        loss = torch.sum((weight * diff) ** 2)
        return loss


def wmse_loss(input, target, weight, size_average=None,
              reduce=None, reduction='mean'):
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({})."
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()), stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        diff = input - target
        diff[np.where(diff > 0)] /= 4.
        ret = diff ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target \
            = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(
            expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class WMSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, weight):
        return wmse_loss(input, target, weight, reduction=self.reduction)
