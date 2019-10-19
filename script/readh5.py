import h5py

infh = h5py.File('nusc_baidu.h5', 'r')
infh.keys()
input = infh['input'].value
print(input.shape)
print(input[0, :, 0, 0])
output = infh['output'].value
print(output.shape)
print(output[0, :, 0, 0])
infh.close()
