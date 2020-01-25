from torch.autograd import Variable
import torch.onnx
from BCNN import BCNN
import os.path


input_names = ["actual_input_1"]
output_names = ["output1"]
dynamic_axes = dict(
    zip(input_names, [{0: 'batch_size'} for i in range(len(input_names))]))

pretrained_model \
    = "/home/kosuke/catkin_ws_autoware/src/autoware_perception/lidar_apollo_cnn_seg_detect/train_baiducnn/pytorch/checkpoints/bcnn_bestmodel_0122.pt"
bcnn_model = BCNN()
bcnn_model.load_state_dict(torch.load(pretrained_model))
x = Variable(torch.randn(1, 8, 640, 640))

torch.onnx.export(bcnn_model, x,
                  os.path.splitext(pretrained_model)[0] + '.onnx', verbose=True)

# torch.onnx.export(
#     bcnn_model, x, os.path.splitext(pretrained_model)[0] + '.onnx',
#     verbose=True,
#     input_names=input_names, output_names=output_names,
#     dynamic_axes=dynamic_axes)
