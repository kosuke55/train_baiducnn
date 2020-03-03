### Under Development

1. Download [nuscenes](https://www.nuscenes.org/) and set path to [nuscenes API](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes).  

2. create\_dataset\_from\_nusc.py generates a dataset to train bcnn. Set dataroot and SAVE_DIR.  
3. Execute start_server.sh and access from a web browser.  
4. Then you can train with train_bcnn.py.  
5. After training, it can be converted to onnx by pytorch2onnx.py and converted to engine by [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt).  


![result](https://github.com/kosuke55/train_baiducnn/blob/media/bcnn_trt_class_all_nusc_0201.gif)  

#### reference
[apollo 3D Obstacle Percption description][1]  

[1]:https://github.com/ApolloAuto/apollo/blob/master/docs/specs/3d_obstacle_perception.md

[autoware_perception description][2]  

[2]:https://github.com/k0suke-murakami/autoware_perception/tree/feature/integration_baidu_seg/lidar_apollo_cnn_seg_detect

[bat67/pytorch-FCN-easiest-demo][3]  

[3]:https://github.com/bat67/pytorch-FCN-easiest-demo
