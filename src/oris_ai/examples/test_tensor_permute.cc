#include <iostream>
#include <vector>

#include "oris_ai/common/tensor.h"

int main(int argc, char** argv) {
    // 3D 텐서를 생성합니다. 텐서의 크기는 4x4x4 입니다.
    // Tensor()의 두 번째 인자로 cpu_only=true로 설정하여 CPU만 사용하도록 합니다.
    std::vector<size_t> shape = {2, 3, 4};
    oris_ai::Tensor<float> tensor(shape, true); // CPU 전용

    // 텐서의 shape을 확인하고 출력합니다.
    std::cout << "Tensor shape: ";
    const std::vector<size_t>& tensor_shape = tensor.Shape();
    for (size_t dim : tensor_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 텐서의 모든 값을 1.0으로 초기화합니다 (CPU 메모리에).
    std::cout << "Set CPU Data" << std::endl;
    tensor.SetCPUData(1.0f);

    // 초기화된 값을 모두 출력합니다.
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

    // 텐서의 차원을 재배열합니다 (예: (0, 2, 1) 순서로).
    std::cout << "Permuting tensor dimensions..." << std::endl;
    std::vector<size_t> new_order = {0, 2, 1};
    tensor.Permute(new_order);

    // 재배열된 텐서의 shape을 확인하고 출력합니다.
    std::cout << "Permuted tensor shape: ";
    const std::vector<size_t>& permuted_shape = tensor.Shape();
    for (size_t dim : permuted_shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // 재배열된 텐서의 데이터를 출력합니다.
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
