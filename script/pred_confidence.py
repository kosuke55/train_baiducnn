import numpy as np
import h5py
import caffe
import cv2

caffe.set_device(0)
caffe.set_mode_gpu()
infh = h5py.File('oneframe_nusc_baidu_confidence.h5', 'r')
in_feature = infh['data'].value
print(in_feature.shape)
# model = 'logs/oneframe_nusc_baidu_confidence_iter_7001.caffemodel'
# model = 'logs/nusc_baidu_confidence_w01_iter10000.caffemodel'
# model = 'logs/mini_nusc_baidu_confidence_iter_100000.caffemodel'
# model = 'logs/nusc_baidu_confidence_iter_1000000.caffemodel'
# model = 'logs/allconf/nusc_baidu_confidence_iter_1145.caffemodel'
# model = 'logs/oneframe/nusc_baidu_confidence_iter_3534.caffemodel'
# model = 'logs/oneframe/nusc_baidu_confidence_iter_2192.caffemodel'
model = 'logs/allconf/nusc_baidu_confidence_4_iter_32000.caffemodel'


net = caffe.Net('data/pred_confidence.prototxt',
                model,
                caffe.TEST)

# in_feature = infh['data'].value


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
