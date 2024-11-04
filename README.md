# Oris_AI
Oris_AI is a deep learning framework especially for on-devices.

## 1. Supported Layer
 - Input: Image Files, OpenCV Mat
 - CNN: Convolution, MaxPooling, Batch Normalization
 - Activation: SiLU
 - Output: Detect(Yolo v8)

## 3. Requirements

Component | Recommended version | Command to check the version
--------- | --------------- | ----------------------------
Ubuntu | 22.04 | lsb_release -a
gcc/g++ | 11.4.0 | gcc --version
cmake | 3.30.2 | cmake --version
cudatoolkit | 12.1 | nvcc --version
cudnn | 9.1 | cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

## 4. Prerequisites
- cmake [Required]
- protobuf [Required]
- glog [Required]
- OpenCV [Required] : https://github.com/opencv/opencv
- OpenBLAS [Required] : https://github.com/xianyi/OpenBLAS.git
- doxygen [Optional - for documentation]

### 4-1. cmake
We recommend using the latest version of cmake. The latest version of cmake can be found at: [https://cmake.org/download/]

If needed, cmake can also be installed using the following commands. However, please note that installing via `apt-get` may not provide the latest version of cmake:

```
$ sudo apt-get install cmake
$ sudo apt-get install cmake-curses-gui # Only needed if you plan to use ccmake
```

### 4-2. protobuf
```
$ sudo apt-get install libprotobuf-dev protobuf-compiler
```

### 4-3. glog
```
$ sudo apt-get install libgoogle-glog-dev
```

### 4-4. OpenCV
To do

### 4-5. OpenBLAS
To do

### 4-6. doxygen
```
$ sudo apt-get install doxygen
```

## 5. How to compile Oris_AI
The installation path of Oris_AI (currently `/Oris_AI`) needs to be modified to suit your environment.

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
$ cd /Oris_AI/build
$ ccmake ..
```

## 6. How to run Oris_AI
The source codes of example are located in the following path.
```
{ORIS_AI_OSS_PATH}/src/oris_ai/examples
```

The binaries of example are located in the following path.
```
{ORIS_AI_OSS_PATH}/build/bin
```

## 7. Example

### 7-1. Tensor
/Oris_AI/src/oris_ai/examples/test_tensor.cc
```
int main() {
  // Creates a 3D tensor with dimensions 4x4x4 and allocates it on the GPU.
  std::vector<size_t> shape = {4, 4, 4};
  oris_ai::Tensor<float> tensor(shape, oris_ai::Device::GPU);

  // Checks and prints the shape of the tensor.
  std::cout << "Tensor shape: ";
  const std::vector<size_t>& tensor_shape = tensor.Shape();
  for (size_t dim : tensor_shape) {
      std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Initializes all values in the tensor to 1.0 (in CPU memory).
  tensor.SetCPUData(1.0f);

  // Copies the data from the CPU to the GPU.
  tensor.To(oris_ai::Device::GPU);

  // Retrieves the data from the GPU and checks the value at position (2, 3, 1).
  std::vector<size_t> indices = {2, 3, 1};
  float& value = tensor.GetCPUData(indices); // Copies the data from GPU to CPU and references it.

  // Prints the current value at position (2, 3, 1).
  std::cout << "The value at (2, 3, 1) is: " << value << std::endl;

  // Modifies the value at position (2, 3, 1) to 5.0.
  value = 5.0f;

  // Reflects the modified value back to the GPU (copies data from CPU to GPU).
  tensor.To(oris_ai::Device::GPU);

  // Checks the value again at position (2, 3, 1) from the GPU and copies it back to the CPU.
  tensor.To(oris_ai::Device::CPU);  // Copies data back to CPU
  std::cout << "The modified value at (2, 3, 1) is: " << tensor.GetCPUData(indices) << std::endl;

  return 0;
}
```