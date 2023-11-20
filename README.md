# LightGlue-OnnxRunner
## Introduction
LightGlue-OnnxRunner is a repository hosts the C++ inference code of LightGlue in ONNX format. LightGlue is a lightweight feature matcher with high accuracy and blazing fast inference. It takes as input a set of keypoints and descriptors for each image and returns the indices of corresponding points.  
* Offical Paper : *[LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/pdf/2306.13643.pdf)*  
* Official Repository ：*[cvg/LightGlue](https://github.com/cvg/LightGlue)*  
* Open Neural Network Exchange (ONNX) Repository : *[fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)*  

![superpoint_lightglue_end2end效果图](assets/superpoint_lightglue_end2end.png)  
<p align="center">
<em>superpoint_lightglue_end2end.onnx renderings</em>
</p>  

![disk_lightglue_end2end效果图](assets/disk_lightglue_end2end.png)
<p align="center">
<em>disk_lightglue_end2end.onnx renderings</em>
</p>

## Attention⚠️  
Currently, the interface only supports CPU execution.The specific experimental data and equipment used are shown below. And the inferface is only supported on Windows and may encounter issues when running on Linux.

## Updates📰
- **[2023.09.08]** : LightGlueOnnxRunner supporting end-to-end model inference of SuperPoint and DISK  
- **[2023.09.11]** : LightGlueDecoupleOnnxRunner supporting decouple model inference of SuperPoint/DISK + LightGlue   
- **[2023.09.12]** : Support calling GPU inference and complete README.md experimental data


## Development Enviroments🖥️
>  - Windows 11 Professional 
>  - CUDA v11.7
>  - cmake version 3.26.2

## Quick Start
### Installation
Install this repo in the following ways :  
```bash
git clone https://github.com/OroChippw/LightGlue-OnnxRunner.git
cd LightGlue-OnnxRunner
```
### Requirements⚒️
``` 
# onnxruntime-cpu 3rdparty
This repository use onnxruntime-win-x64-1.14.1
# onnxruntime-gpu 3rdparty
This repository use onnxruntime-win-x64-gpu-1.15.0 # for CUDA 11.7
# opencv 3rdparty
This repository use opencv4.8.0
# CXX_STANDARD 17
```
### Build and Run
```
# Enter the source code directory where CMakeLists.txt is located, and create a new build folder
mkdir build
# Enter the build folder and run CMake to configure the project
cd build
cmake ..
# Use the build system to compile/link this project
cmake --build .
# If the specified compilation mode is debug or release, it is as follows
# cmake --build . --config Debug
# cmkae --build . --config Release
```
### Model Checkpoints
### Experiment Record
Environment Device : i5-13500H + NVIDIA GeForce RTX 4060 Laptop GPU（8GB）.  
All models are available in repository *[fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)*  
The inference speed of onnxruntime-GPU will be slower when the first image is loaded, and will return to normal speed later.  


#### Decouple
| Extractor Type | Extractor Model Name | CPU speed(ms) | GPU speed(ms) | Matcher Model Name | CPU speed(ms) | GPU speed(ms) |
| --------------- | -------------------- | ------------- | ------------- | ------------------ | ------------- | ------------- |
| SuperPoint      | superpoint.onnx       | Debug:123ms Release: 73ms | Debug:15ms Release:13ms | superpoint_lightglue.onnx | Debug:2384ms Release: 2112ms | Debug:155ms Release:230ms |
| DISK      | disk.onnx       | Debug:341ms Release: 336ms | Debug:28ms Release:25ms | disk_lightglue.onnx | Debug:3347ms Release: 3257ms | Debug: 230ms Release:245ms |

#### End-to-End🌟🌟🌟
| Extractor Type | Model Name | Model Size(MB/GB) | CPU speed(ms) | GPU speed(ms) |
| :------------------:| :---------------: | :---------------: | :---------------: | :---------------: |
| SuperPoint | superpoint_lightglue_end2end.onnx | 50.1MB | Debug:2181ms Release: 1829ms |  Debug: 170ms Release:166ms  |
| DISK | disk_lightglue_end2end.onnx | 48.9MB | Debug:3312ms Release: 3287ms | Debug: 285ms Release:285ms |

### CHANGELOG
#### 20231120
- 适配v1.0.0: Fused LightGlue-ONNX接口改动

- 适配linux平台，完成CPU下测试，移除部分Linux下用不到的代码

- 修复可视化时，两张输入图片大小不一致时匹配特征计算错误导致的特征显示错误

- 非end-to-end模式增加sp和lg的独立接口

### License
This project is licensed under the MIT License.
