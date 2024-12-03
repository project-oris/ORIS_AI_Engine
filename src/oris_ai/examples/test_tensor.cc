#include <iostream>
#include <vector>

#include "oris_ai/tensor/tensor.h"

int main(int argc, char** argv) {
  // Create a 3D tensor with a size of 4x4x4.
  // If the second argument in Tensor() is not set to cpu_only=true,
  // both CPU and GPU can be used.
  std::vector<size_t> shape = {4, 4, 4};
  oris_ai::Tensor<float> tensor(shape);

  // Check and print the shape of the tensor.
  std::cout << "Tensor shape: ";
  const std::vector<size_t>& tensor_shape = tensor.GetShape();
  for (size_t dim : tensor_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Initialize all values of the tensor to 1.0 (in CPU memory).
  std:: cout << "Set CPU Data" << std::endl;
  tensor.SetCPUData(1.0f);

  // // Print all the initialized values.
  // std::cout << "Initial tensor values:" << std::endl;
  // for (size_t i = 0; i < shape[0]; ++i) {
  //   for (size_t j = 0; j < shape[1]; ++j) {
  //     for (size_t k = 0; k < shape[2]; ++k) {
  //       std::vector<size_t> idx = {i, j, k};
  //       std::cout << "tensor[" << i << "][" << j << "][" << k << "] = " 
  //                 << tensor.GetCPUData(idx) << std::endl;
  //     }
  //   }
  // }

  // Copy data from CPU to GPU.
  std:: cout << "Tensor To GPU" << std::endl;
  tensor.To(oris_ai::Device::GPU);

  // Retrieve the value at position (2, 3, 1) from the GPU.
  std::vector<size_t> indices = {2, 3, 1};
  std:: cout << "Get GPU Data" << std::endl;
  float& value = tensor.GetCPUData(indices); // Copy data from GPU to CPU for reference.

  // Print the current value at position (2, 3, 1).
  std::cout << "The value at (2, 3, 1) is: " << value << std::endl;

  // Modify the value at position (2, 3, 1) to 5.0.
  std::cout << "New value = 5" << std::endl;
  value = 5.0f;

  // Reflect the modified value to the GPU (copy data from CPU to GPU).
  std:: cout << "Tensor To GPU" << std::endl;
  tensor.To(oris_ai::Device::GPU);

  // Verify the value at position (2, 3, 1) again from the GPU and copy it back to the CPU.
  std:: cout << "Tensor To CPU" << std::endl;
  tensor.To(oris_ai::Device::CPU);  // Copy data back to CPU
  std::cout << "The modified value at (2, 3, 1) is: " << tensor.GetCPUData(indices) << std::endl;

  return 0;
}
