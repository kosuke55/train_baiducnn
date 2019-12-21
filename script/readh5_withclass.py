import h5py
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

infh = h5py.File('conf_class_10.h5', 'r')
infh.keys()

frame_id = 0

loss_weight = infh['loss_weight'].value
print("loss_weight" + str(loss_weight.shape))  # NxCxHxW
loss_weight = np.transpose(
    loss_weight, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
loss_weight = loss_weight[frame_id]
loss_weight[np.where(loss_weight != 1)] = 0
loss_weight = loss_weight.reshape(640, 640) * 255
loss_weight = loss_weight.astype(np.uint8)
cv2.imwrite("image/val_loss_weight.png", loss_weight)

out_feature = infh['output'].value
print("out_feature" + str(out_feature.shape))  # NxCxHxW
out_feature = np.transpose(
    out_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
out_feature = out_feature[frame_id]
print("out_feature" + str(out_feature.shape))  # NxCxHxW
out_feature_ = out_feature.reshape(640, 640, 2)
out_image = np.zeros((640, 640, 3), dtype=np.uint8)

print(out_feature[..., 1][out_feature[..., 1] > 0])
car_idx = np.where(out_feature[..., 1] == 1)
bus_idx = np.where(out_feature[..., 1] == 2)
bike_idx = np.where(out_feature[..., 1] == 3)
human_idx = np.where(out_feature[..., 1] == 4)

print(car_idx)
out_image[car_idx] = [255, 0, 0]
out_image[bus_idx] = [0, 255, 0]
out_image[bike_idx] = [0, 0, 255]
out_image[human_idx] = [0, 255, 255]

# out_image = out_feature.reshape(640, 640) * 255
out_image = out_image.astype(np.uint8)
cv2.imwrite("image/conf_class_out.png", out_image)

in_feature = infh['data'].value
print("in_feature" + str(in_feature.shape))  # NxCxHxW
in_feature = np.transpose(
    in_feature, (0, 3, 2, 1))  # NxCxHxW -> NxWxHxC
in_feature = in_feature[frame_id]

vis_all_in_feature = False
if(vis_all_in_feature):
    for i in range(8):
        in_image = in_feature.reshape(640, 640, 8)
        in_image = in_image[:, :, i]
        in_image[np.where(in_image != 0)] = 255
        in_image = in_image.astype(np.uint8)
        cv2.imwrite("image/conf_class_in_{}.png".format(i), in_image)
else:
    in_image = in_feature.reshape(640, 640, 8)
    in_image = in_image[:, :, 7]
    in_image[np.where(in_image != 0)] = 255
    in_image = in_image.astype(np.uint8)
    cv2.imwrite("image/conf_class_in_{}.png".format(0), in_image)


infh.close()
