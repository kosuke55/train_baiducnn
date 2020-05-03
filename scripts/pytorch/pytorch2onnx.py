#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path

from torch.autograd import Variable
import torch.onnx

from BCNN import BCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trained_model', '-tm', type=str,
                        help='trained model',
                        default='checkpoints/mini_instance.pt')
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=672)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=672)
    parser.add_argument('--channel', type=int,
                        help='feature map channel',
                        default=6)
    args = parser.parse_args()

    bcnn_model = BCNN(in_channels=args.channel, n_class=6)
    bcnn_model.load_state_dict(torch.load(args.trained_model))
    x = Variable(torch.randn(1, args.channel, args.width, args.height))

    torch.onnx.export(bcnn_model, x, os.path.splitext(
        args.trained_model)[0] + '.onnx', verbose=True)
