import numpy as np
import h5py
import caffe
import cv2
import feature_generator as fg

caffe.set_device(0)
caffe.set_mode_gpu()
infh = h5py.File('oneframe_nusc_baidu_confidence.h5', 'r')
in_feature = infh['data'].value


model = 'logs/allconf/nusc_baidu_confidence_4_iter_32000.caffemodel'


net = caffe.Net('data/pred_confidence.prototxt',
                model,
                caffe.TEST)

net.blobs['data'].data[...] = in_feature
out = net.forward()
conf = out['confidence_score']
print(conf.shape)
conf = np.transpose(
    conf, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
print(conf.shape)
conf = conf.reshape(640, 640)
print(conf.shape)

mean = np.mean(conf)
print(mean)
print(np.max(conf))
print(np.min(conf))
conf[np.where(conf > 0.6)] = 255

cv2.imwrite("confidence_pred.png", conf)
