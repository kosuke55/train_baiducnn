import h5py
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# infh = h5py.File('nusc_baidu.h5', 'r')
infh = h5py.File('nusc_baidu_confidence.h5', 'r')
infh.keys()
in_feature = infh['data'].value
in_feature = np.transpose(
    in_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
print(in_feature.shape)
# print(np.count_nonzero(in_feature))
in_image = in_feature.reshape(640, 640, 8)
print(in_image.shape)
print(np.count_nonzero(in_image[:, :, 7]))

out_feature = infh['output'].value
print(out_feature.shape)  # NxCxHxW
print(out_feature[0, :, 0, 0])

out_feature = np.transpose(
    out_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
print(out_feature.shape)
# print(np.count_nonzero(out_feature))
out_image = out_feature.reshape(640, 640) * 255
print(np.count_nonzero(out_image))
print(out_image.shape)


cv2.imwrite("confidence_out.png", out_image)
in_image = in_image[:, :, 7] * 255
cv2.imwrite("confidence_in.png", in_image)


# cv2.namedWindow('window')
# cv2.imshow('window', out_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

infh.close()
