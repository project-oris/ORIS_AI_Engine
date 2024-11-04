#include <iostream>
#include <vector>

#include "oris_ai/common/tensor.h"

int main(int argc, char** argv) {
  // 3D 텐서를 생성합니다. 텐서의 크기는 4x4x4 입니다.
  // Tensor()으 두번째 인자로 cpu_only=true로 설정하지 않으면
  // CPU와 GPU 모두 사용 가능합니다.
  std::vector<size_t> shape = {4, 4, 4};
  oris_ai::Tensor<float> tensor(shape);

  // 텐서의 shape을 확인하고 출력합니다.
  std::cout << "Tensor shape: ";
  const std::vector<size_t>& tensor_shape = tensor.Shape();
  for (size_t dim : tensor_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // 텐서의 모든 값을 1.0으로 초기화합니다 (CPU 메모리에).
  std:: cout << "Set CPU Data" << std::endl;
  tensor.SetCPUData(1.0f);

  // // 초기화된 값을 모두 출력합니다.
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

  // CPU에서 GPU로 데이터를 복사합니다.
  std:: cout << "Tensor To GPU" << std::endl;
  tensor.To(oris_ai::Device::GPU);

  // GPU에서 데이터를 가져와 (2, 3, 1) 위치의 값을 확인합니다.
  std::vector<size_t> indices = {2, 3, 1};
  std:: cout << "Get GPU Data" << std::endl;
  float& value = tensor.GetCPUData(indices); // GPU 데이터를 CPU로 복사하여 참조합니다.

  // 현재 (2, 3, 1) 위치의 값을 출력합니다.
  std::cout << "The value at (2, 3, 1) is: " << value << std::endl;

  // (2, 3, 1) 위치의 값을 5.0으로 수정합니다.
  std::cout << "New value = 5" << std::endl;
  value = 5.0f;

  // 수정된 값을 GPU에 반영합니다 (CPU에서 GPU로 데이터 복사).
  std:: cout << "Tensor To GPU" << std::endl;
  tensor.To(oris_ai::Device::GPU);

  // GPU에서 다시 (2, 3, 1) 위치의 값을 확인하고 CPU로 복사합니다.
  std:: cout << "Tensor To CPU" << std::endl;
  tensor.To(oris_ai::Device::CPU);  // 다시 CPU로 데이터 복사
  std::cout << "The modified value at (2, 3, 1) is: " << tensor.GetCPUData(indices) << std::endl;

  return 0;
}
