import torch
from BCNN import BCNN
from torchviz import make_dot

net = BCNN()
bcnn_img = torch.rand(1, 7, 640, 640)
category, instance, confidence, classify, heading, height = net(bcnn_img)
print(category.shape)
print(instance.shape)
print(confidence.shape)
print(classify.shape)
print(heading.shape)
print(height.shape)

dot = make_dot(category, params=dict(net.named_parameters()))

dot.render("bcnn")
