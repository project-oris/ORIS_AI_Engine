# Oris_AI
Oris_AI is a deep learning framework especially for on-devices.

## 1. Supported Layer
 - Input: Image Files, OpenCV Mat
 - CNN: Convolution, MaxPooling
 - Activation: SiLU
 - Output: Detect(Yolo v8)

## 2. Supported DNN
 - Detection: Yolo v8

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

### 4-1. cmake
We recommend using the latest version of cmake. The latest version of cmake can be found at: [https://cmake.org/download/]

If needed, cmake can also be installed using the following commands. However, please note that installing via `apt-get` may not provide the latest version of cmake:

```
$ sudo apt-get install cmake
$ sudo apt-get install cmake-curses-gui # Only needed if you plan to use ccmake
```

### 4-2. protobuf
```
$ sudo apt-get install libprotobuf-dev
```

### 4-3. glog
```
$ sudo apt-get install libgoogle-glog-dev
```

### 4-4. OpenCV
To do

### 4-5. OpenBLAS
To do

## 5. How to compile Oris_AI
The installation path of Oris_AI (currently {ORIS_AI_OSS_PATH}) needs to be modified to suit your environment.

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