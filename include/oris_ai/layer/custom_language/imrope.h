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

#include "oris_ai/layer/layer.h"
#include "oris_ai/layer/custom_language/capture_aware_kernel_layer.h"
#include "oris_ai/tensor/tensor.h"

#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace oris_ai {

template <typename T> class RMSNorm;

// Qwen3-VL text decoder RoPE with four interleaved sections; GPU only.
// Reference: llama.cpp rope_multi (is_imrope).
template <typename T>
class Qwen3IMROPE : public HiddenLayerAbstract<T>,
                   public CaptureAwareKernelLayer {
 public:
  // Rotates the first `n_rot` dims (0: all); sum(sections) * 2 equals that.
  Qwen3IMROPE(const std::string& layer_name,
              size_t n_dims,
              float  freq_base,
              const std::array<int32_t, 4>& sections,
              size_t n_rot = 0);
  ~Qwen3IMROPE() override = default;

  // Frequency factors apply only when rope_freqs_count == n_dims / 2.
  void InitFromRaw(const float* rope_freqs_data,
                   size_t rope_freqs_count,
                   PoolType freq_pool_type = PoolType::STATIC);

  // Not owned, shape [4 * S]. Stays valid, at a stable device address, across
  // every Forward and replay.
  void SetPositionTensor(const Tensor<int32_t>* positions);

  void InitRMSNormFusion(RMSNorm<T>* norm);
  // UseRMSNormFusion says whether the current launch folds the norm.
  bool RMSNormFusionConfigured() const { return norm_source_ != nullptr; }
  bool UseRMSNormFusion() const {
    return norm_source_ && (fuse_prefill_norm_ ||
        norm_source_->GetEffectiveElementCount() == norm_decode_elements_);
  }

  // Rotates q4d [B, S, H_q, D] in place.
  void ForwardInPlace(Tensor<T>* q4d);

  // Rotates k4d and writes it to the FP16 KV cache.
  void ForwardWithKVWriteF16(Tensor<T>* k4d,
                             half*  kv_dst,
                             size_t dst_row_stride,
                             size_t cache_position);
  // Fixed once both position tensors are bound; an owner may keep the result.
  bool CanPairStructurally(const Qwen3IMROPE<T>& q_rope) const;
  // Whether ForwardPairWithKVWriteF16 can run for the current launch.
  bool CanPairWith(const Qwen3IMROPE<T>& q_rope) const;
  // `q_rope`'s ForwardInPlace and this layer's ForwardWithKVWriteF16 at once.
  void ForwardPairWithKVWriteF16(Qwen3IMROPE<T>* q_rope, Tensor<T>* q4d, Tensor<T>* k4d,
                                 half* kv_dst, size_t dst_row_stride,
                                 size_t cache_position);
  // Rotates k4d and writes it to the FP32 KV cache.
  void ForwardWithKVWrite(Tensor<T>* k4d,
                          T*     kv_dst,
                          size_t dst_row_stride,
                          size_t cache_position);

  // Driven through the Forward* calls above; Forward() does nothing.
  void Forward() override {}
  int32_t GetSubStepCount() const override { return 1; }

 private:
  struct ImropeLaunchParams {
    int B;
    int S;
    int H;
    int D;
    int n_rot;
    float theta_scale;
    const float* freq_factors_dev;
    const int32_t* pos_dev;
    cudaStream_t stream;
  };

  ImropeLaunchParams ResolveImropeLaunchParams(const Tensor<T>& x4d) const;

  Tensor<T>* norm_source_ = nullptr;
  const Tensor<T>* norm_weight_ = nullptr;
  float norm_eps_ = 0.f;
  size_t norm_decode_elements_ = 0;
  bool fuse_prefill_norm_ = false;

  size_t n_dims_;
  size_t n_rot_;
  float  freq_base_;
  std::array<int32_t, 4> sections_;
  float theta_scale_ = 0.f;

  std::unique_ptr<Tensor<float>> rope_freqs_;
  bool has_rope_freqs_ = false;

  const Tensor<int32_t>* positions_ = nullptr;
};

}  // namespace oris_ai
