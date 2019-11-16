import numpy as np
import caffe
import cv2
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import feature_generator as fg
import ros_numpy


class CNNSegDetector(object):
    def __init__(self, prototxt, model):
        self.size = 640
        self.model = model
        self.prototxt = prototxt
        self.fg = fg.Feature_generator()
        self.net = caffe.Net(self.prototxt,
                             self.model,
                             caffe.TEST)
        self.inpc = rospy.get_param(
            '~input_point', "points_raw")
        self.bridge = CvBridge()
        self.pub_conf = rospy.Publisher("/conf", Image, queue_size=10)
        self.subscribe()

    def subscribe(self):
        self.sub = rospy.Subscriber(self.inpc, PointCloud2, self.callback)

    def callback(self, data):
        pc = ros_numpy.numpify(data)
        points = np.zeros((pc.shape[0], 4))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        points[:, 3] = pc['intensity']
        print(points.shape)
        # pc = np.array(list(pc2.read_points(
        #     data, skip_nans=True,
        #     field_names=("x", "y", "z", "intensity"))), dtype=np.float32)
        # self.fg.generate(pc)
        self.fg.generate(points)
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
        conf[np.where(conf > 0.8)] = 255
        conf[np.where(conf <= 0.8)] = 0
        conf = conf.astype(np.uint8)
        mean = np.mean(conf)
        print(mean)
        print(np.max(conf))
        print(np.min(conf))
        print(conf.shape)
        conf_msg = self.bridge.cv2_to_imgmsg(conf, "mono8")
        self.pub_conf.publish(conf_msg)
        cv2.imwrite("confidence_pred_tmp.png", conf)

    @staticmethod
    def load_pc_from_file(pc_f):
        # return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 5])
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])


if __name__ == "__main__":
    rospy.init_node("cnn_seg_detector", anonymous=False)
    prototxt = "data/pred_confidence.prototxt"
    model = 'logs/allconf/nusc_baidu_confidence_4_iter_78000.caffemodel'
    detector = CNNSegDetector(prototxt, model)

    pcd = '/media/kosuke/f798886c-8a70-48a4-9b66-8c9102072e3e/nuScenes/trainval/samples/LIDAR_TOP/n015-2018-11-21-19-58-31+0800__LIDAR_TOP__1542801733448313.pcd.bin'
    rospy.spin()

