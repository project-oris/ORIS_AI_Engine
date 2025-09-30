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

void CUDAPreprocessInt8NHWCtoNHWC(const uint8_t* input_data, int8_t* output_data,
                                  const int height, const int width,
                                  const float normalization_value,
                                  const float input_scale);

void CUDAChannelPaddingInt8(const Tensor<int8_t>* input_tensor, Tensor<int8_t>* output_tensor);

void CUDAQuantizeInt32ToInt8(Tensor<int32_t>* input_tensor, Tensor<int8_t>* output_tensor,
                            const float alpha);

// void CUDAQuantizeInt32ToFP32(Tensor<int32_t>* input_tensor, Tensor<float>* output_tensor,
//                             const float alpha);

// void CUDAInt32HardSwish(Tensor<int32_t>* output);

void CUDAQuantizeInt32ToInt8(const Tensor<int32_t>* input_tensor, Tensor<int8_t>* output_tensor,
                            const float alpha);

// void CUDAFloatHardSwishWithQuantization(const Tensor<float>* input, Tensor<int8_t>* output,
//   const float output_scale);

void CUDACombineQuantizeInt8HardSwish(Tensor<int32_t>* input_tensor, Tensor<int8_t>* output_tensor,
                                      const float conv_quant_scale, const float act_input_scale,
                                      const float act_output_scale);

void CUDACombineQuantizeInt8HardSwishRescale(Tensor<int32_t>* input_tensor, Tensor<int8_t>* output_tensor,
                                      const float conv_quant_scale, const float act_input_scale,
                                      const float act_output_scale, const float backbone_quant_scale);

// void CUDAInt8HardSwishWithQuantization(Tensor<int8_t>* output,
//   const float input_scale, const float output_scale);

void CUDACombineQuantizeInt8HardSwishResidual(Tensor<int32_t>* input_tensor, Tensor<int8_t>* output_tensor,
                                              Tensor<int8_t>* residual_tensor,
                                              const float conv_quant_scale, const float act_input_scale,
                                              const float act_output_scale, const float cv2_quant_scale,
                                              const float residual_quant_scale);

void CUDASplitInt8(const Tensor<int8_t>* input_tensor,
                  const std::vector<std::unique_ptr<Tensor<int8_t>>>& split_tensors,
                  const std::vector<size_t>& split_sizes);

// void CUDAConcatChannelInt8(const int8_t** input_tensors_gpu, Tensor<int8_t>* output_tensor,
//   const int input_size, const int* input_channels_per_input_gpu, const int* input_channel_offset_gpu);

void CUDAConcatChannelInt8(const int8_t** input_tensors_gpu, Tensor<int8_t>* output_tensor,
                          const int input_size, const int* input_channels_per_input_gpu,
                          const int* input_channel_offset_gpu, const float* output_scales_gpu);

void CUDAMaxPoolingInt8(const Tensor<int8_t>* input_tensor, Tensor<int8_t>* output_tensor,
                        const int kernel_size, const int stride, const int padding);

void CUDAUpsampleInt8(const Tensor<int8_t>* input_tensor, Tensor<int8_t>* output_tensor,
                      float scale_factor, UpsampleMode mode);

void CUDADequantInt8(const Tensor<int8_t>* input_tensor, Tensor<float>* output_tensor,
                    const std::vector<size_t>& output_shape);

} // namespace oris_ai