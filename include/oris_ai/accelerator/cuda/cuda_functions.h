/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/layer/upsample.h"

namespace oris_ai {

template <typename T>
void CUDAActivationSiLU(Tensor<T>* output);

template <typename T>
void CUDAAxpy(const T& alpha, const Tensor<T>& input, Tensor<T>& output);

template <typename T>
void CUDAElementWiseAdd(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C);

template <typename T>
void CUDAElementWiseSub(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C);

template <typename T>
void CUDAElementWiseMul(Tensor<T>* C, const Tensor<T>* A, size_t rows, size_t cols);

template <typename T>
void CUDASigmoid(Tensor<T>* tensor);

template <typename T>
void CUDATranspose(const Tensor<T>* input_tensor, Tensor<T>* output_tensor, const size_t* cuda_transposed_index);

template <typename T>
void CUDAUpsample(const Tensor<T>* input_tensor, Tensor<T>* output_tensor,
                  float scale_factor, UpsampleMode mode);

} // namespace oris_ai