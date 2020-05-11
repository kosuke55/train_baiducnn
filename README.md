# Train lidar\_apollo\_instance\_segmentation CNN  

Train lidar\_apollo\_instance\_segmentation CNN with Nuscenes.  

## Getting Started  
Only supports python3.  

1) Clone original_model branch and install some packages.  

```
git clone -b original_model https://github.com/kosuke55/train_baiducnn.git  
pip install -r requirements.txt  
```
2) [create_dataset_from_nusc.py](scripts/create_dataset/create_dataset_from_nusc.py) is for creating a dataset to train apollo cnn.  Set dataroot and save_dir.  

```
cd ~/train_baiducnn/scripts/create_dataset  
python create_dataset_from_nusc.py  --dataroot <downloaded nuscenes path> --save_dir <dir to save created dataset> --nusc_version <v1.0-mini or v1.0-trainval>  
```

3) Execute start\_server.sh and access from a web browser. Then you can train with [train_bcnn.py](scripts/pytorch/train_bcnn.py).  

```
cd ~/train_baiducnn/scripts/pytorch  
./start_server.sh  
python train_bcnn.py --data_path <dir to save created dataset>  
```

4) Trained model can be converted to onnx by [pytorch2onnx.py](scripts/pytorch/pytorch2onnx.py) and converted to engine by [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt).  

```
cd ~/train_baiducnn/scripts/pytorch  
python pytorch2onnx --trained_model <your_trained_model.pt>  
# after installing onnx-tensorrt  
onnx2trt <your_trained_model.onnx> -o <your_trained_model.engine>  
```

5) Run [lidar_apollo_instance_segmentation](https://github.com/tier4/AutowareArchitectureProposal/tree/master/src/perception/object_recognition/detection/lidar_apollo_instance_segmentation) with <your_trained_model.engine>  

<img src="https://user-images.githubusercontent.com/39142679/81552155-ed721480-93bd-11ea-9b9d-88e88dab2ecf.gif" width="680">  

## reference
[apollo 3D Obstacle Percption description][1]  

[1]:https://github.com/ApolloAuto/apollo/blob/master/docs/specs/3d_obstacle_perception.md

[autoware_perception description][2]  

[2]:https://github.com/k0suke-murakami/autoware_perception/tree/feature/integration_baidu_seg/lidar_apollo_cnn_seg_detect

[bat67/pytorch-FCN-easiest-demo][3]  

[3]:https://github.com/bat67/pytorch-FCN-easiest-demo
