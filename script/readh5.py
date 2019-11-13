import h5py
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# infh = h5py.File('nusc_baidu.h5', 'r')
infh = h5py.File('oneframe_nusc_baidu_confidence.h5', 'r')
infh.keys()
in_feature = infh['data'].value
print(in_feature.shape)
in_feature = np.transpose(
    in_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC

# print(np.count_nonzero(in_feature))
in_image = in_feature.reshape(640, 640, 8)
# print(in_image.shape)
# print(in_image[:, :, 0][in_image[:, :, 0] > 0.5])
# print(np.count_nonzero(in_image[:, :, 7]))

out_feature = infh['output'].value
print(out_feature.shape)  # NxCxHxW

loss_weight = infh['loss_weight'].value
print(loss_weight.shape)  # NxCxHxW
# print(out_feature[0, :, 0, 0])

out_feature = np.transpose(
    out_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
# print(out_feature.shape)
# print(np.count_nonzero(out_feature))
out_image = out_feature.reshape(640, 640) * 255
# print(np.count_nonzero(out_image))
# print(out_image.shape)
lw_image = loss_weight.reshape(640, 640)
lw_image[np.where(lw_image == 1.)] = 0
lw_image *= 255
print(lw_image.shape)

cv2.imwrite("loss_weight.png", lw_image)
cv2.imwrite("confidence_out.png", out_image)
# in_image = in_image[:, :, 7] * 255
for i in range(8):
    in_image = in_feature.reshape(640, 640, 8)
    in_image = in_image[:, :, i]
    in_image[np.where(in_image != 0)] = 255
    cv2.imwrite("confidence_in_{}.png".format(i), in_image)


# cv2.namedWindow('window')
# cv2.imshow('window', out_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

infh.close()
