import h5py
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# infh = h5py.File('nusc_baidu.h5', 'r')
# infh = h5py.File('oneframe_nusc_baidu_confidence.h5', 'r')
infh = h5py.File('nusc_baidu_confidence.h5', 'r')
infh.keys()
idx = 20

in_feature = infh['data'].value
print(in_feature.shape)
in_feature = np.transpose(
    in_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
print(in_feature.shape)
in_feature = in_feature[idx]


# print(np.count_nonzero(in_feature))
# in_image = in_feature.reshape(640, 640, 8)
# print(in_image.shape)
# print(in_image[:, :, 0][in_image[:, :, 0] > 0.5])
# print(np.count_nonzero(in_image[:, :, 7]))

out_feature = infh['output'].value

print(out_feature.shape)  # NxCxHxW

loss_weight = infh['loss_weight'].value
# print(loss_weight.shape)  # NxCxHxW
# print(out_feature[0, :, 0, 0])

# out_feature = out_feature[20]
print(out_feature.shape)  # NxCxHxW

# out_feature = np.transpose(
#     out_feature, (2, 1, 0))  # NxCxHxW -> NxWxHxC
######
out_feature = np.transpose(
    out_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
out_feature = out_feature[idx]

loss_weight = np.transpose(
    loss_weight, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
loss_weight = loss_weight[idx]

# # print(out_feature.shape)
# # print(np.count_nonzero(out_feature))
out_image = out_feature.reshape(640, 640) * 255
# # print(np.count_nonzero(out_image))
# # print(out_image.shape)
loss_weight[np.where(loss_weight != 1)] = 0
loss_weight = loss_weight.reshape(640, 640) * 255

# lw_image *= 255
# print(lw_image.shape)

cv2.imwrite("loss_weight_all.png", loss_weight)
# cv2.imwrite("confidence_out.png", out_image)
# # in_image = in_image[:, :, 7] * 255
for i in range(8):
    in_image = in_feature.reshape(640, 640, 8)
    in_image = in_image[:, :, i]
    in_image[np.where(in_image != 0)] = 255
    cv2.imwrite("confidence_in_all_{}.png".format(i), in_image)


# cv2.namedWindow('window')
# cv2.imshow('window', out_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("confidence_out_all.png", out_image)
infh.close()
