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

#include "oris_ai/layer/custom_language/gqa.h"
#include "oris_ai/layer/custom_language/imrope.h"
#include "oris_ai/layer/custom_language/rmsnorm.h"
#include "oris_ai/layer/custom_language/rope.h"
#include "oris_ai/tensor/tensor.h"

#include <cstdint>
#include <memory>
#include <string>

namespace oris_ai {

// GQA with per-head Q/K RMSNorm before RoPE and Qwen3IMROPE by default.
template <typename T>
class Qwen3GQA : public GroupedQueryAttention<T> {
 public:
  using Base = GroupedQueryAttention<T>;
  using Base::Base;

  ~Qwen3GQA() override = default;

  // STANDARD_1D needs SetQwen3StandardComponents first.
  enum class RopeMode { IMROPE, STANDARD_1D };

  RopeMode rope_mode() const { return rope_mode_; }

  // Takes ownership of the helper layers.
  void SetQwen3Components(std::unique_ptr<RMSNorm<T>> q_norm,
                          std::unique_ptr<RMSNorm<T>> k_norm,
                          std::unique_ptr<Qwen3IMROPE<T>> imrope_q,
                          std::unique_ptr<Qwen3IMROPE<T>> imrope_k);

  void SetQwen3StandardComponents(
      std::unique_ptr<RMSNorm<T>> q_norm,
      std::unique_ptr<RMSNorm<T>> k_norm,
      std::unique_ptr<RotaryPositionEmbedding<T>> rope_q,
      std::unique_ptr<RotaryPositionEmbedding<T>> rope_k);

  // Not owned; stays valid for every Forward().
  void SetPositionTensor(const Tensor<int32_t>* positions);

  // Public for the owning model to wire q_norm and k_norm.
  using Base::q_after_proj_tensor;
  using Base::k_after_proj_tensor;

  const CaptureAwareKernelLayer* GetRopeKVWriteCaptureHook() const override {
    switch (rope_mode_) {
      case RopeMode::IMROPE:      return imrope_k_.get();
      case RopeMode::STANDARD_1D: return rope_k_.get();
    }
    return nullptr;
  }

 protected:
  void ReleaseBorrowedWeights() override;

  std::vector<typename HiddenLayerAbstract<T>::SubLayerInfo>
      GetActivationTensorInfo() override;

  void PreRopeQ() override;
  void PreRopeK() override;
  void DispatchRopeQ_inplace_(Tensor<T>* q4d) override;
  void DispatchRopeK_with_kv_write_(Tensor<T>* k4d,
                                    half* kv_dst,
                                    size_t dst_row_stride,
                                    size_t cache_position) override;

 private:
  std::unique_ptr<RMSNorm<T>>     q_norm_;
  std::unique_ptr<RMSNorm<T>>     k_norm_;
  std::unique_ptr<Qwen3IMROPE<T>> imrope_q_;
  std::unique_ptr<Qwen3IMROPE<T>> imrope_k_;
  bool q_rope_pending_ = false;
  bool rope_pair_ = false;
  void ResolveRopePairing();
  bool StandardRopeFoldsNorm_(const RotaryPositionEmbedding<T>* rope) const;
  const Tensor<int32_t>*          positions_ = nullptr;

  RopeMode rope_mode_ = RopeMode::IMROPE;

  std::unique_ptr<RotaryPositionEmbedding<T>> rope_q_;
  std::unique_ptr<RotaryPositionEmbedding<T>> rope_k_;

  void WireQKNorm(std::unique_ptr<RMSNorm<T>> q_norm,
                  std::unique_ptr<RMSNorm<T>> k_norm);

  template <typename U> friend struct Qwen3GQATestAccess;
};

}  // namespace oris_ai
