import numpy as np
import h5py
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
infh = h5py.File('nusc_baidu_confidence.h5', 'r')
in_feature = infh['data'].value

net = caffe.Net('data/pred_confidence.prototxt',
                'data/nusc_baidu_confidence_iter_2.caffemodel',
                caffe.TEST)

in_feature = infh['data'].value
print(in_feature.shape)

net.blobs['data'].data[...] = in_feature
out = net.forward()
conf = out['confidence_score']
print(conf)
print(conf[conf > 0])
