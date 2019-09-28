import h5py

infh = h5py.File('nusc_baidu.h5', 'r')
infh.keys()
data = infh['data'].value
print(data.shape)
infh.close()
