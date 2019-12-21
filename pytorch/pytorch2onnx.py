from torch.autograd import Variable
import torch.onnx
from BCNN import BCNN

pretrained_model = "checkpoints/bcnn_bestmodel_mini.pt"
bcnn_model = BCNN()
bcnn_model.load_state_dict(torch.load(pretrained_model))
x = Variable(torch.randn(1, 8, 640, 640))
torch.onnx.export(bcnn_model, x, 'bcnn_model_mini.onnx', verbose=True)
