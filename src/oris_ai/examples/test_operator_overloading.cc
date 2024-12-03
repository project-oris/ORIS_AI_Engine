#include <iostream>
#include "oris_ai/tensor/tensor.h"  // Tensor class should be included here

using namespace oris_ai;

int main() {
  // Define two tensors with the same shape
  std::vector<size_t> shape = {2, 3};  // Shape is 2x3
  Tensor<float> tensorA(shape, true);  // Initialize tensor A
  Tensor<float> tensorB(shape, true);  // Initialize tensor B

  // Assign some values to tensor A
  float* dataA = tensorA.GetCPUDataPtr();
  dataA[0] = 1.0f; dataA[1] = 2.0f; dataA[2] = 3.0f;
  dataA[3] = 4.0f; dataA[4] = 5.0f; dataA[5] = 6.0f;

  // Assign some values to tensor B
  float* dataB = tensorB.GetCPUDataPtr();
  dataB[0] = 0.5f; dataB[1] = 1.5f; dataB[2] = 2.5f;
  dataB[3] = 3.5f; dataB[4] = 4.5f; dataB[5] = 5.5f;

  // Perform element-wise addition and subtraction
  Tensor<float> tensorSum = tensorA + tensorB;  // A + B
  Tensor<float> tensorDiff = tensorA - tensorB; // A - B

  // Print results for addition
  std::cout << "Addition Result (A + B):\n";
  const float* sumData = tensorSum.GetCPUDataPtr();
  for (size_t i = 0; i < tensorSum.GetTotalCount(); ++i) {
    std::cout << sumData[i] << " ";
  }
  std::cout << "\n";

  // Print results for subtraction
  std::cout << "Subtraction Result (A - B):\n";
  const float* diffData = tensorDiff.GetCPUDataPtr();
  for (size_t i = 0; i < tensorDiff.GetTotalCount(); ++i) {
    std::cout << diffData[i] << " ";
  }
  std::cout << "\n";

  return 0;
}
