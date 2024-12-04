# ORIS_AI
ORIS_AI is a deep learning framework specialized for on-device in the ORIS (On-device Robot Intelligence SW-System) project.

## 1. Supported Layer
 - Input: OpenCV Mat
 - CNN: Concat, Convolution, MaxPooling, Split, Transpose
 - Activation: SiLU
 - Custom (for Yolo v8): C2f, BottleNeck (for C2f), DFL, SPPF
 - Output (for Yolo v8): Detect 

## 2. Supported DNN
 - Detection: Yolo v8

## 3. Requirements

Component | Recommended version | Command to check the version
--------- | --------------- | ----------------------------
Ubuntu | 22.04 | lsb_release -a
gcc/g++ | 11.4.0 | gcc --version
cmake | 3.30.2 | cmake --version
cudatoolkit | 12.1 | nvcc --version
cudnn | 9.1 | cat {Your CUDNN Install Path}/cudnn_version.h | grep CUDNN_MAJOR -A 2
opencv | 4.8
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
$ sudo apt-get install libprotobuf-dev
```

### 4-4. glog
```
$ sudo apt-get install libgoogle-glog-dev
```

### 4-5. OpenBLAS
#### 4-5-1. Build OpenBLAS
```
$ sudo apt-get build-essential gfortran
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
#### 4-6-1. Install prerequisites for openCV
```
$ sudo apt-get build-essential
$ sudo apt-get libjpeg-dev libtiff5-dev libpng-dev
$ sudo apt-get ffmpeg libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev libxine2-dev
$ sudo apt-get libv4l-dev v4l-utils
$ sudo apt-get libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
$ sudo apt-get libgtk-3-dev
$ sudo apt-get libatlas-base-dev gfortran libeigen3-dev
$ sudo apt-get python3-dev python3-numpy
```

#### 4-6-2. Build OpenCV
```
$ git clone https://github.com/opencv/opencv.git {OpenCV_Path}
$ cd {OpenCV_Path}
$ mkdir build && cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=OFF -D WITH_CUBLAS=OFF -D WITH_CUFFT=OFF -D WITH_MATLAB=OFF \
-D WITH_IPP=OFF -D WITH_1394=OFF -D WITH_OPENCLAMDBLAS=OFF -D WITH_OPENCLAMDFFT=OFF \
-D WITH_TBB=OFF -D WITH_XINE=OFF \
-D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_WITH_DEBUG_INFO=OFF -D BUILD_DOCS=OFF ..
$ make -j$(nproc)
$ make install
$ sudo ldconfig
```

## 5. How to compile ORIS_AI
The installation path of ORIS_AI (currently {ORIS_AI_OSS_PATH}) needs to be modified to suit your environment.

### 5-1. Native compile
```
$ cd {ORIS_AI_OSS_PATH}
$ mkdir build
$ cd build
$ cmake ..
$ make -j$(nproc)
```

### 5-2. Cmake configuration
Use `ccmake`
```
$ cd {ORIS_AI_OSS_PATH}/build
$ ccmake ..
```

## 6. How to run ORIS_AI
The source codes of example are located in the following path.
```
{ORIS_AI_OSS_PATH}/src/oris_ai/examples
```

The binaries of example are located in the following path.
```
{ORIS_AI_OSS_PATH}/build/bin
```

## 7. Example

### 7-1. Basic Tensor Operation
{ORIS_AI_OSS_PATH}/src/oris_ai/examples/test_tensor_basic.cc
```
int main(int argc, char** argv) {
  // Creates a 3D tensor with dimensions 4x4x4.
  // If the second argument to Tensor() is not set to cpu_only=true,
  // both CPU and GPU can be used.
  std::vector<size_t> shape = {4, 4, 4};
  oris_ai::Tensor<float> tensor(shape);

  // Checks and prints the shape of the tensor.
  std::cout << "Tensor shape: ";
  const std::vector<size_t>& tensor_shape = tensor.Shape();
  for (size_t dim : tensor_shape) {
      std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Initializes all values in the tensor to 1.0 (in CPU memory).
  std::cout << "Set CPU Data" << std::endl;
  tensor.SetCPUData(1.0f);

  // Copies data from CPU to GPU.
  std::cout << "Tensor To GPU" << std::endl;
  tensor.To(oris_ai::Device::GPU);

  // Retrieves data from the GPU and checks the value at position (2, 3, 1).
  std::vector<size_t> indices = {2, 3, 1};
  std::cout << "Get GPU Data" << std::endl;
  float& value = tensor.GetCPUData(indices); // Copies the data from GPU to CPU and references it.

  // Prints the current value at position (2, 3, 1).
  std::cout << "The value at (2, 3, 1) is: " << value << std::endl;

  // Modifies the value at position (2, 3, 1) to 5.0.
  std::cout << "New value = 5" << std::endl;
  value = 5.0f;

  // Reflects the modified value back to the GPU (copies data from CPU to GPU).
  std::cout << "Tensor To GPU" << std::endl;
  tensor.To(oris_ai::Device::GPU);

  // Checks the value again at position (2, 3, 1) from the GPU and copies it back to the CPU.
  std::cout << "Tensor To CPU" << std::endl;
  tensor.To(oris_ai::Device::CPU);  // Copies data back to CPU
  std::cout << "The modified value at (2, 3, 1) is: " << tensor.GetCPUData(indices) << std::endl;

  return 0;
}
```

### 7-2. Premute Tensor
{ORIS_AI_OSS_PATH}/src/oris_ai/examples/test_tensor_permute.cc
```
int main(int argc, char** argv) {
    // Creates a 3D tensor with dimensions 2x3x4.
    // Set the second argument of Tensor() to cpu_only=true to use CPU only.
    std::vector<size_t> shape = {2, 3, 4};
    oris_ai::Tensor<float> tensor(shape, true); // CPU only

    // Checks and prints the shape of the tensor.
    std::cout << "Tensor shape: ";
    const std::vector<size_t>& tensor_shape = tensor.Shape();
    for (size_t dim : tensor_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Initializes all values in the tensor to 1.0 (in CPU memory).
    std::cout << "Set CPU Data" << std::endl;
    tensor.SetCPUData(1.0f);

    // Prints all initialized values.
    std::cout << "Initial tensor values:" << std::endl;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            for (size_t k = 0; k < shape[2]; ++k) {
                std::vector<size_t> idx = {i, j, k};
                std::cout << "tensor[" << i << "][" << j << "][" << k << "] = " 
                          << tensor.GetCPUData(idx) << std::endl;
            }
        }
    }

    // Permutes the dimensions of the tensor (e.g., in the order of (0, 2, 1)).
    std::cout << "Permuting tensor dimensions..." << std::endl;
    std::vector<size_t> new_order = {0, 2, 1};
    tensor.Permute(new_order);

    // Checks and prints the shape of the permuted tensor.
    std::cout << "Permuted tensor shape: ";
    const std::vector<size_t>& permuted_shape = tensor.Shape();
    for (size_t dim : permuted_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Prints the data of the permuted tensor.
    std::cout << "Permuted tensor values:" << std::endl;
    for (size_t i = 0; i < permuted_shape[0]; ++i) {
        for (size_t j = 0; j < permuted_shape[1]; ++j) {
            for (size_t k = 0; k < permuted_shape[2]; ++k) {
                std::vector<size_t> idx = {i, j, k};
                std::cout << "permuted_tensor[" << i << "][" << j << "][" << k << "] = " 
                          << tensor.GetCPUData(idx) << std::endl;
            }
        }
    }

    return 0;
}
```

## 8. Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## 9. Authors
Seungtae Hong - sthong@etri.re.kr

## 10. Acknowledgment
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. RS-2024-00339187, Core Technology Development of On-device Robot Intelligence SW Platform)

## 11.License
Distributed under the MIT License.