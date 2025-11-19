# ORIS_AI
ORIS_AI is a deep learning framework specialized for on-device in the ORIS (On-device Robot Intelligence SW-System) project.

## 1. Supported Layers
 - Input: OpenCV Mat
 - Common: Concat, Convolution, Depthwise Convolution, ElementWise (SUM), MatMul, MaxPooling, Softmax. Split, Transpose
 - Activation: SiLU
 - Custom (for YOLO v8): C2f,BottleNeck (for C2f/C3k), DFL, SPPF, Proto
 - Custom (for YOLO v11): C3k, C3k2, C2PSA, PSABlock, Attention
 - Output (for YOLO v8/v11): Detect, Segment

## 2. Supported DNNs
 - Detection: YOLO v8/v11
 - Segmentation: YOLO v8/v11

## 3. Requirements

Component | Recommended version | Command to check the version
--------- | --------------- | ----------------------------
JetPack | 6.2.1 | `dpkg -l \| grep nvidia-jetpack`
gcc/g++ | 11.4.0 | gcc --version
cmake | ≥ 3.22.1 | cmake --version
cudatoolkit | 12.6 | nvcc --version
cudnn | 9.3 | cat {Your CUDNN Install Path}/cudnn_version.h | grep CUDNN_MAJOR -A 2
opencv | 4.8 | python3 -c "import cv2; print(cv2.__version__)"
protobuf | 3.12.4 | protoc --version

## 4. Prerequisites
- cmake [Required]
- eigen [Required]
- protobuf [Required]
- glog [Required]
- OpenCV [Required] : https://github.com/opencv/opencv
- OpenBLAS [Required] : https://github.com/xianyi/OpenBLAS.git

### 4-1. cmake
We recommend using the latest version of cmake. The latest version of cmake can be found at: [https://cmake.org/download/]

If needed, cmake can also be installed using the following commands. However, please note that installing via `apt-get` may not provide the latest version of cmake:

```
$ sudo apt-get install cmake
$ sudo apt-get install cmake-curses-gui # Only needed if you plan to use ccmake
```

### 4-2. eigen
```
$ sudo apt-get install libeigen3-dev
```

### 4-3. protobuf
```
$ sudo apt-get install libprotobuf-dev protobuf-compiler
```

### 4-4. glog
```
$ sudo apt-get install libgoogle-glog-dev
```

### 4-5. OpenBLAS
#### 4-5-1. Build OpenBLAS
```
$ sudo apt-get install build-essential gfortran
$ git clone https://github.com/xianyi/OpenBLAS.git {OpenBLAS_Path}
$ cd {OpenBLAS_Path}
$ make FC=gfortran -j$(nproc)
$ make install PREFIX=/usr/local
$ ldconfig
```

#### 4-5-2. Configure OPENBLAS_NUM_THREADS 
```
$ cd ~
$ vi .bashrc
```
Then add OPENBLAS_NUM_THREADS=n to the last line.
n is the maximum number of threads that your CPU supports.
If you want to check the number of CPU cores in the current system, enter the command below.
```
$ grep -c processor /proc/cpuinfo
```
After editing the shell environment, reflect the edited shell environment and check the modified contents.
```
$ source .bashrc
$ echo $OPENBLAS_NUM_THREADS
```

### 4-6. OpenCV

#### 📢 Notice on GPL License
OpenCV itself is released under the **BSD 3-Clause License**.  
However, when building or installing OpenCV, additional external libraries may introduce **GPL obligations**.  

- If GPL-related libraries such as `libx264-dev` or `libxvidcore-dev` are included, the resulting OpenCV build may be subject to the **GPL license**, requiring source code disclosure when redistributed.  
- When enabling `WITH_FFMPEG=ON`, please check whether your FFmpeg installation is built in **LGPL-only mode** or whether it links to **GPL codecs** (e.g., x264).  
- The option `OPENCV_ENABLE_NONFREE=ON` enables patented algorithms and should remain **OFF** unless you have verified that your use case allows it.  

📌 **Important**  
- For **personal or research use**, GPL obligations are generally not an issue.  
- For **distribution, commercial use, or providing binaries externally**, you must **verify whether GPL libraries are linked** and, if so, ensure **full compliance with the GPL license**.  
- If you want to avoid GPL obligations, build OpenCV **without GPL-related libraries** and rely only on **BSD/LGPL components**.  

---

#### 4-6-1. Install prerequisites
Install only the required dependencies for your environment. Avoid GPL-related packages if you want to distribute without GPL obligations.

#### 4-6-2. Build OpenCV
Use the appropriate build options for your environment. By default, it is recommended to set `WITH_FFMPEG=OFF` and `OPENCV_ENABLE_NONFREE=OFF`.

## 5. How to compile ORIS_AI
The installation path of ORIS_AI (currently `/ORIS_AI`) needs to be modified to suit your environment.

### 5-1. Native compile
```
$ cd /ORIS_AI
$ mkdir build
$ cd build
$ cmake ..
$ make -j$(nproc)
```

### 5-2. Cmake configuration
Use `ccmake`
```
$ cd /ORIS_AI/build
$ ccmake ..
```

### 5.3. Configure the library path
Create symbolic links for the shared libraries to `/usr/local/lib`.
```
$ sudo ln -s /ORIS_AI/lib/aarch_64/libORISCore.so /usr/local/lib/libORISCore.so
$ sudo ln -s /ORIS_AI/lib/aarch_64/libORISCUDA.so /usr/local/lib/libORISCUDA.so
```

## 6. How to run ORIS_AI
The source codes of example are located in the following path.
```
/ORIS_AI/src/oris_ai/examples
```

The binaries of example are located in the following path.
```
/ORIS_AI/build/bin
```

## 7. Example

When using GPU, the ORIS AI inference engine is executed as an asynchronous stream, and when using CUDA, if you want to measure the actual execution time of each step, you must use the sync option.

#### 7-1. YOLOv8 Official Model

##### 7-1-1. YOLOv8n Detection
- /ORIS_AI/src/oris_ai/examples/test_yolo_v8.cc
```
Usage: ./test_yolo_v8 [OPTIONS]
Options:
  -c      Use CPU for inference. Default is GPU.
  -sync   Synchronize CUDA before timing. (For accurate GPU timing)
Example:
  ./test_yolo_v8 -c
  ./test_yolo_v8 -sync
```

##### 7-1-2. YOLOv8n Segmentation
- /ORIS_AI/src/oris_ai/examples/test_yolo_v8_seg.cc
```
Usage: ./test_yolo_v8_seg [OPTIONS]
Options:
  -c      Use CPU for inference. Default is GPU.
  -sync   Synchronize CUDA before timing. (For accurate GPU timing)
Example:
  ./test_yolo_v8_seg -c
  ./test_yolo_v8_seg -sync
```

#### 7-2. YOLOv11 Official Model

##### 7-2-1. YOLOv11n Detection
- /ORIS_AI/src/oris_ai/examples/test_yolo_v11.cc
```
Usage: ./test_yolo_v11 [OPTIONS]
Options:
  -c      Use CPU for inference. Default is GPU.
  -sync   Synchronize CUDA before timing. (For accurate GPU timing)
Example:
  ./test_yolo_v11 -c
  ./test_yolo_v11 -sync
```

##### 7-2-2. YOLOv11n Segmentation
- /ORIS_AI/src/oris_ai/examples/test_yolo_v11_seg.cc
```
Usage: ./test_yolo_v11_seg [OPTIONS]
Options:
  -c      Use CPU for inference. Default is GPU.
  -sync   Synchronize CUDA before timing. (For accurate GPU timing)
Example:
  ./test_yolo_v11_seg -c
  ./test_yolo_v11_seg -sync
```

## 8. Benchmark Results (Jetson ORIN AGX)

Performance benchmarks measured after 5 warmup iterations for GPU inference.

### YOLOv8n

| Model | Task | CPU | GPU |
|-------|------|-----|-----|
| YOLOv8n | Object Detection | 248 ms | 8.5 ~ 9.0 ms |
| YOLOv8n | Instance Segmentation | 538.7 ms | 10.5 ~ 11.0 ms |

### YOLOv11n

| Model | Task | CPU | GPU |
|-------|------|-----|-----|
| YOLOv11n | Object Detection | 234 ms | 9.4 ~ 9.5 ms |
| YOLOv11n | Instance Segmentation | 548 ms | 12.0 ~ 12.8 ms |


---

## 9. Validation 

This demonstration showcases real-time inference results from our robot intelligence neural network across four critical industrial safety scenarios. The system provides comprehensive hazard detection and monitoring capabilities for industrial facilities.

*Real-time inference results demonstrating multi-scenario hazard detection capabilities*


---

### Detection

#### 🦺 Hazard Situation Detection
Analysis and notification of worker PPE (Personal Protective Equipment) compliance status

- **Neural Network**: YOLOv11 (Object Detection)
- **Model Size**: 9.98 MB
- **Inference Time**: 10.38 ms

<div align="center">
  <img src="./demo/2_Hazard_Situation_Detection_logo.gif" width="50%" alt="Hazard Situation Detection">
  <p><em>Real-time PPE compliance monitoring detecting safety vest and helmet usage</em></p>
</div>


#### 👤 Hazard Personnel Detection
Tracking and display of workers deviating from designated pathways

- **Neural Network**: YOLOv11 (Object Detection) + ByteTrack
- **Model Size**: 13 MB
- **Inference Time**: 31 ms
- **Hardware Acceleration**: Mobilint NPU Accelerated LLM

<div align="center">
  <img src="./demo/3_Hazard_Personnel_Detection_logo.gif" width="50%" alt="Hazard Personnel Detection">
  <p><em>Multi-object tracking system monitoring worker movement and pathway compliance</em></p>
</div>


#### ⚠️ Hazard Factor Detection
Identification and marking of facility safety risk factors

- **Neural Network**: YOLOv11 (Object Detection)
- **Model Size**: 9.98 MB
- **Inference Time**: 10.38 ms

<div align="center">
  <img src="./demo/4_Hazard_Factor_Detection_logo.gif" width="50%" alt="Hazard Factor Detection">
  <p><em>Automated detection of facility risk factors including hazardous materials and unsafe conditions</em></p>
</div>

---

### Detection & Segmentation

#### 🔥 Hazard Event Detection
Real-time detection and warning display for fire and smoke incidents

- **Neural Network**: YOLOv11 (Detection & Segmentation)
- **Model Size**: 10.9 MB
- **Inference Time**: 12.73 ms

<div align="center">
  <img src="./demo/1_Hazard_Event_Detection_logo.gif" width="50%" alt="Hazard Event Detection">
  <p><em>Instance segmentation for precise fire and smoke detection with real-time alerting</em></p>
</div>

## Acknowledgement

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2024-00339187, Core Technology Development of On-device Robot Intelligence SW Platform).
