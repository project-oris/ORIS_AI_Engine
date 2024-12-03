#include <iostream>
#include <vector>

#include "oris_ai/tensor/tensor.h"

int main(int argc, char** argv) {
  // Create a 3D tensor with a size of 2x3x4.
  // Set the second argument in Tensor() to cpu_only=true to use only the CPU.
  std::vector<size_t> shape = {2, 3, 4};
  oris_ai::Tensor<float> tensor(shape, true); // CPU only

  // Check and print the shape of the tensor.
  std::cout << "Tensor shape: ";
  const std::vector<size_t>& tensor_shape = tensor.GetShape();
  for (size_t dim : tensor_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Initialize all values of the tensor to 1.0 (in CPU memory).
  std::cout << "Set CPU Data" << std::endl;
  tensor.SetCPUData(1.0f);

  // Print all the initialized values.
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

  // Permute the tensor dimensions (e.g., in the order (0, 2, 1)).
  std::cout << "Permuting tensor dimensions..." << std::endl;
  std::vector<size_t> new_order = {0, 2, 1};
  tensor.Permute(new_order);

  // Check and print the shape of the permuted tensor.
  std::cout << "Permuted tensor shape: ";
  const std::vector<size_t>& permuted_shape = tensor.GetShape();
  for (size_t dim : permuted_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Print the permuted tensor data.
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
