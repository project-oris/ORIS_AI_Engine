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

// CUDA kernels of the vision encoder and the shared activation,
// element-wise and LayerNorm primitives.

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/layer/activation_type.h"
#include "oris_ai/gguf_reader/gguf_types.h"

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace oris_ai {

void CUDAActivationInplace(Tensor<float>* output, const ActivationType activation_type);

void CUDAActivationForward(const Tensor<float>* input, Tensor<float>* output,
                           const ActivationType activation_type);

void CUDABiasActivationInplace(
  float* out,
  const float* bias,
  int N, int C, int H, int W,
  ActivationType activation_type);

void CUDASwiGLUForward(const Tensor<float>* gate, const Tensor<float>* up, Tensor<float>* output);

void CUDAGEGLUForward(const Tensor<float>* gate, const Tensor<float>* up, Tensor<float>* output);

void CUDAGEGLUQuickForward(const Tensor<float>* gate, const Tensor<float>* up, Tensor<float>* output,
    bool has_clamp = false, float gate_lo = 0.f, float gate_hi = 0.f,
    float up_lo = 0.f, float up_hi = 0.f);


template <typename T>
void CUDAElementWiseAdd(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C);

void CUDAElementWiseAddInplace(float* A, const float* B, size_t count,
                               cudaStream_t stream);

// C = A + act_b(B) with act_b GELU or SiLU; B is not modified. FP32 only.
void CUDAElementWiseAddActivatedB(const Tensor<float>* A, const Tensor<float>* B,
                                  Tensor<float>* C, ActivationType act_b);

// output[m, n] = bias[n].
template <typename T>
void CUDALinearBiasRowBroadcast(T* output, const T* bias, size_t rows,
                                size_t cols, cudaStream_t stream);

template <typename T>
void CUDAElementWiseSub(const Tensor<T>* A, const Tensor<T>* B, Tensor<T>* C);

template <typename T>
void CUDALayerNormForward(const Tensor<T>* input, Tensor<T>* output,
  const Tensor<T>* weight, const Tensor<T>* bias,
  const size_t batch_size, const size_t norm_size,
  const T eps, const bool elementwise_affine);

// LayerNorm with half output rows. fp32_output is not written; only its
// alignment selects the kernel. Call only when
// CUDALayerNormF16OutputSupported(norm_size) is true.
bool CUDALayerNormF16OutputSupported(size_t norm_size);
void CUDALayerNormForwardF16Output(const Tensor<float>* input, half* output,
  const float* fp32_output,
  const Tensor<float>* weight, const Tensor<float>* bias,
  const size_t batch_size, const size_t norm_size,
  const float eps, const bool elementwise_affine);

// The bias-residual LayerNorms below first set residual += addend +
// addend_bias on rows below update_rows, in place, then normalize
// batch_size rows. Call only when CUDABiasResidualLayerNormSupported(norm_size)
// is true.
bool CUDABiasResidualLayerNormSupported(size_t norm_size);

// LayerNorm stored as the Q8_1 MMQ activation of a Q8_0 consumer. The
// launchers return false and store nothing when the case does not apply;
// fp32_output is not written.
bool CUDALayerNormQ8_1OutputSupported(size_t norm_size);
bool CUDALayerNormForwardQ8_1Output(
  const Tensor<float>* input, block_q8_1_mmq* output, int rows, size_t padded_cols,
  const float* fp32_output, const Tensor<float>* weight, const Tensor<float>* bias,
  const size_t batch_size, const size_t norm_size, const float eps,
  const bool elementwise_affine);
bool CUDABiasResidualAddLayerNormQ8_1Output(
  float* residual, const float* addend, const float* addend_bias,
  size_t update_rows, block_q8_1_mmq* output, int rows, size_t padded_cols,
  const float* fp32_output, const float* weight, const float* bias,
  size_t batch_size, size_t norm_size, float eps, bool elementwise_affine,
  cudaStream_t stream);
void CUDABiasResidualAddLayerNorm(
  float* residual, const float* addend, const float* addend_bias,
  size_t update_rows, float* output,
  const float* weight, const float* bias,
  size_t batch_size, size_t norm_size, float eps,
  bool elementwise_affine, cudaStream_t stream);
// CUDABiasResidualAddLayerNorm with half output rows; fp32_output is not
// written.
void CUDABiasResidualAddLayerNormF16Output(
  float* residual, const float* addend, const float* addend_bias,
  size_t update_rows, half* output, const float* fp32_output,
  const float* weight, const float* bias,
  size_t batch_size, size_t norm_size, float eps,
  bool elementwise_affine, cudaStream_t stream);

// LayerNorm over the C axis of an NCHW tensor. fused_act honors SiLU only.
template <typename T>
void CUDALayerNormChannelFirst4D(const Tensor<T>* input, Tensor<T>* output,
  const Tensor<T>* weight, const Tensor<T>* bias,
  size_t N, size_t C, size_t HW,
  T eps, bool elementwise_affine,
  ActivationType fused_act = ActivationType::NONE);

}  // namespace oris_ai
