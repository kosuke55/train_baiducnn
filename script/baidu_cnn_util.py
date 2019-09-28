import math


# project point cloud to 2d map. clac in which grid point is.
# pointcloud to pixel
def F2I(val, orig, scale):
    return int(math.floor((orig - val) * scale))


def Pc2Pixel(in_pc, in_range, out_size):
    inv_res = 0.5 * out_size / in_range
    return int(math.floor((in_range - in_pc) * inv_res))


# retutn the distance from my car to center of the grid.
# Pc means point cloud = real world scale. so transform pixel scale to real world scale
def Pixel2pc(in_pixel, in_size, out_range):
    res = 2.0 * out_range / in_size
    return out_range - (in_pixel + 0.5) * res
