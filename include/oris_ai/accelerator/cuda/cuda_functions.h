/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute (ETRI)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *******************************************************************************/
#pragma once

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/layer/upsample.h"

#include <cuda_runtime.h>

namespace oris_ai {

template <typename T>
void CUDAActivation(Tensor<T>* output, const ActivationType activation_type);

// template <typename T>
// void CUDABiasActivation(Tensor<T>* output, const T* bias, const ActivationType activation_type);

template <typename T>
// void CUDAElementWiseAdd(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C, ElementWiseOpType op_type);
void CUDAElementWiseAdd(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C);

template <typename T>
void CUDAElementWiseSub(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C);

template <typename T>
void CUDABroadcastMul(Tensor<T>* C, const Tensor<T>* A, int rows, int cols);

template <typename T>
void CUDASigmoid(Tensor<T>* tensor);

template <typename T>
void CUDATranspose(const Tensor<T>* input_tensor, Tensor<T>* output_tensor, const int* cuda_transposed_index);

template <typename T>
void CUDAUpsample(const Tensor<T>* input_tensor, Tensor<T>* output_tensor,
                  float scale_factor, UpsampleMode mode);

template <typename T>
void CUDASplitDim2(const Tensor<T>* input_tensor, Tensor<T>* output_tensor,
                  const int split_size, const int height_offset,
                  const bool use_view_input_shape = false);

template <typename T>
void CUDADepthwiseConvForward(const Tensor<T>* input, Tensor<T>* output, const Tensor<T>* weight,
                              const int kernel_height, const int kernel_width,
                              const int pad_height, const int pad_width,
                              const int stride_height, const int stride_width,
                              const Tensor<T>* bias, const ActivationType activation_type,
                              bool use_view_input_shape = false);

// template <typename T>
// void CUDAFindMaxConfidenceAndFilter(
//     const T* dbox_data, const T* cls_data,
//     T* result_dbox_x1, T* result_dbox_y1, T* result_dbox_x2, T* result_dbox_y2,
//     T* result_max_conf, int* result_max_class_idx, int* result_count,
//     int num_classes, int num_detections, T score_threshold);

void CUDAPreprocessFloat(const uint8_t* input_data, float* output_data,
                        const int height, const int width,
                        const float normalization_value);

template <typename T>
void CUDASoftmaxWidth(const Tensor<T>* input_tensor, Tensor<T>* output_tensor,
                      bool use_view_input_shape);

} // namespace oris_ai