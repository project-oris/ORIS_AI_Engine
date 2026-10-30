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

#include <glog/logging.h>

namespace oris_ai {

/// Axes a LayerNorm normalizes over: the contiguous trailing axes, or the C
/// axis of an NCHW tensor.
enum class LayerNormAxisMode { kContiguousTrailing, kChannelFirst4D };

/// Layer normalization with optional weight and bias.
template <typename T>
class LayerNorm : public HiddenLayerAbstract<T> {
public:
  LayerNorm(const std::string& layer_name)
    : HiddenLayerAbstract<T>(layer_name),
      eps_(1e-5), elementwise_affine_(true) {}

  ~LayerNorm() = default;

  /// Borrows gamma and beta without copying; a null beta_data means no bias.
  /// Call SetInputTensor first.
  void InitLayerNormFromRaw(
      const void* gamma_data, size_t gamma_bytes,
      const void* beta_data, size_t beta_bytes,
      const std::vector<size_t>& normalized_shape, double eps,
      PoolType output_pool_type = PoolType::STATIC);

  /// Replaces the output tensor to fit the current input; layers bound to the
  /// old output must be rebound.
  void RebuildOutputForInput();

  /// Like InitLayerNormFromRaw for a rank-4 NCHW input. Fails unless
  /// SupportsChannelFirst() is true.
  void InitLayerNormChannelFirst4D(
      const void* gamma_data, size_t gamma_bytes,
      const void* beta_data, size_t beta_bytes,
      size_t N, size_t C, size_t H, size_t W, double eps,
      PoolType output_pool_type = PoolType::STATIC);

  virtual bool SupportsChannelFirst() const { return false; }

  /// Channel-first path only, SiLU only. Call after
  /// InitLayerNormChannelFirst4D.
  inline void SetChannelFirstFusedActivation(ActivationType act) {
    channelfirst_fused_act_ = act;
  }

  inline ActivationType GetChannelFirstFusedActivation() const {
    return channelfirst_fused_act_;
  }

  inline Tensor<T>* GetWeight() { return weight_.get(); }

  /// nullptr when there is no bias.
  inline Tensor<T>* GetBias() {
    return bias_ ? bias_.get() : nullptr;
  }

  inline const std::vector<size_t>& GetNormalizedShape() const { return normalized_shape_; }

  inline double GetEps() const { return eps_; }

  inline bool GetElementwiseAffine() const { return elementwise_affine_; }

  virtual void Forward() = 0;

  virtual void SetCpuFastPath(bool /*on*/) {}

  void ReleaseBorrowedWeights() override {
    if (weight_) weight_->ClearBorrowedState();
    if (bias_) bias_->ClearBorrowedState();
  }

  bool AllWeightsReleased() const override {
    return (weight_ == nullptr || !weight_->IsBorrowed()) &&
           (bias_   == nullptr || !bias_->IsBorrowed());
  }

  /// Call after resizing the output for a new batch without re-running Init;
  /// the kernel otherwise covers only the old batch.
  inline void UpdateBatchSize(size_t new_batch_size) {
    batch_size_ = new_batch_size;
  }

protected:
  void LayerNormSetupFromRaw(
      const void* gamma_data, size_t gamma_bytes,
      const void* beta_data, size_t beta_bytes,
      const std::vector<size_t>& normalized_shape, double eps,
      PoolType output_pool_type = PoolType::STATIC);

  void LayerNormSetupChannelFirst4D(
      const void* gamma_data, size_t gamma_bytes,
      const void* beta_data, size_t beta_bytes,
      size_t N, size_t C, size_t H, size_t W, double eps,
      PoolType output_pool_type = PoolType::STATIC);

  std::vector<size_t> normalized_shape_;
  double eps_;
  bool elementwise_affine_;
  std::unique_ptr<Tensor<T>> weight_;
  std::unique_ptr<Tensor<T>> bias_;

  size_t batch_size_;
  size_t norm_size_;

  LayerNormAxisMode axis_mode_ = LayerNormAxisMode::kContiguousTrailing;
  size_t cf_N_ = 0, cf_C_ = 0, cf_HW_ = 0;

  ActivationType channelfirst_fused_act_ = ActivationType::NONE;

};

} // namespace oris_ai
