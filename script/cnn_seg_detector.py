import numpy as np
import caffe
import cv2
import feature_generator as fg


class CNNSegDetector(object):
    def __init__(self, prototxt, model):
        self.size = 640
        self.model = model
        self.prototxt = prototxt
        self.fg = fg.Feature_generator()
        self.net = caffe.Net(self.prototxt,
                             self.model,
                             caffe.TEST)

    @staticmethod
    def load_pc_from_file(pc_f):
        # return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 5])
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])

    def predict(self, v_p):
        pc = self.load_pc_from_file(v_p)[:, :4]
        # pc[:, 3] = pc[:, 3] / 255
        print(pc.shape)
        self.fg.generate(pc)
        feature = self.fg.feature
        feature = feature.reshape(self.size, self.size, 8)
        in_feature = feature[np.newaxis, :, :, :]
        in_feature = np.transpose(
            in_feature, (0, 3, 2, 1))  # NxWxHxC -> NxCxHxW
        print(in_feature.shape)
        self.net.blobs['data'].data[...] = in_feature
        out = self.net.forward()
        conf = out['confidence_score']
        conf = np.transpose(
            conf, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
        conf = conf.reshape(640, 640)
        conf[np.where(conf > 0.6)] = 255
        mean = np.mean(conf)
        print(mean)
        print(np.max(conf))
        print(np.min(conf))
        print(conf.shape)
        cv2.imwrite("confidence_pred_.png", conf)


if __name__ == "__main__":
    prototxt = "data/pred_confidence.prototxt"
    model = 'logs/allconf/nusc_baidu_confidence_4_iter_83000.caffemodel'
    detector = CNNSegDetector(prototxt, model)

    pcd = '/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e/nuScenes/trainval/samples/LIDAR_TOP/n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801733448313.pcd.bin'
    # pcd = "pcd/merged.pcd.bin"
    detector.predict(pcd)

