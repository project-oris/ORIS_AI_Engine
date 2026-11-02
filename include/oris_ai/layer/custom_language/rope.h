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

#include <memory>
#include <vector>

#include "oris_ai/layer/custom_language/capture_aware_kernel_layer.h"
#include "oris_ai/layer/layer.h"

namespace oris_ai {

// Pairing of the rotated dims: NORM adjacent, NEOX split-half.
// Reference: llama.cpp rope_norm, rope_neox.
enum class RopeType { NORM = 0, NEOX = 1 };

// Rotary position embedding.
template <typename T>
class RotaryPositionEmbedding : public HiddenLayerAbstract<T>,
                                public CaptureAwareKernelLayer {
public:
  // theta_scale = pow(freq_base, -2.0 / n_dims).
  RotaryPositionEmbedding(const std::string &layer_name,
                          size_t n_dims, float freq_base = 10000.0f,
                          RopeType rope_type = RopeType::NORM);

  ~RotaryPositionEmbedding() = default;

  // With `gpu_output_needed` false the output has no GPU memory.
  void InitRotaryPositionEmbeddingFromRaw(
      const float* rope_freqs_data, size_t rope_freqs_count,
      PoolType freq_pool_type = PoolType::STATIC,
      PoolType output_pool_type = PoolType::ACTIVATION,
      bool gpu_output_needed = true);

  // Copies the frequency factors.
  void SetupFrequencyFactors(const float* weight_data, size_t weight_size,
                            PoolType pool_type);

  virtual void Forward() = 0;

  // dst_row_stride = num_kv_heads * head_dim. The default only calls Forward().
  virtual void ForwardWithKVWrite(T* kv_dst, size_t dst_row_stride, size_t cache_position) {
    Forward();
  }

protected:
  std::unique_ptr<Tensor<float>> freq_factors_;
  bool has_freq_factors_;

  size_t n_dims_;
  size_t batch_size_;
  size_t seq_len_;
  size_t num_heads_;
  size_t head_dim_;
  size_t d_model_;
  size_t s0_;
  size_t s1_;
  size_t s2_;

  float freq_scale_;
  float ext_factor_;
  float attn_factor_;
  float theta_scale_;
  float freq_base_;

  float corr_dims_[2];

  RopeType rope_type_;

public:
  RopeType rope_type() const { return rope_type_; }
};

} // namespace oris_ai
