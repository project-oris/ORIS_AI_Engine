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
#include "oris_ai/layer/activation_type.h"

namespace oris_ai {

/// Base class of every layer: a name and the input tensors.
template <typename Tin, typename Tout = Tin>
class LayerAbstract {
public:
  LayerAbstract(const std::string& layer_name);

  virtual ~LayerAbstract() = default;

  inline Tensor<Tin>* GetInputTensor(size_t input_idx = 0) {
    return input_tensors_.at(input_idx);
  }

  virtual void SetInputTensor(Tensor<Tin>* input_tensor);

  /// Throws if index is out of range.
  virtual void ResetInputTensor(Tensor<Tin>* input_tensor, size_t index = 0);

  virtual inline size_t GetInputSize() { return input_tensors_.size(); }

  virtual void Forward() = 0;

  /// Returns "" unless built with USE_DEBUG_MODE.
  std::string GetLayerName() const;

  /// Has no effect unless built with USE_DEBUG_MODE.
  void SetLayerName(const std::string& layer_name);

protected:
  std::vector<Tensor<Tin>*> input_tensors_;
};

/// Mixed-precision category of a layer, fixed at init time.
enum class AutocastPolicy {
  LOWER_PRECISION_FP,
  FP32_ONLY,
  PASSTHROUGH,
  PROMOTE
};

/// Layer that owns an output tensor.
template <typename Tin, typename Tout = Tin>
class HiddenLayerAbstract : public LayerAbstract<Tin, Tout> {
public:
  HiddenLayerAbstract(const std::string& layer_name) : LayerAbstract<Tin, Tout>(layer_name) {}

  AutocastPolicy GetAutocastPolicy() const { return autocast_policy_; }

  virtual ~HiddenLayerAbstract() = default;

  /// Detaches borrowed (mmap) weight pointers; call before the mapping is
  /// unmapped.
  virtual void ReleaseBorrowedWeights() {}

  virtual void SetCpuFastPath(bool /*on*/) {}

  /// True when no weight tensor still borrows mmap memory.
  virtual bool AllWeightsReleased() const { return true; }

  virtual inline Tensor<Tout>* GetOutputTensor() { return output_tensor_.get(); }

  /// Replaces the output tensor with an FP16 or BF16 one of the same shape.
  /// Call after Init and before downstream layers bind to the output.
  void UpgradeOutputToFP16();
  void UpgradeOutputToBF16();

  /// Lifetime of one ACTIVATION tensor inside a composite layer.
  struct SubLayerInfo {
      size_t   request_id;
      PoolType pool_type;
      int32_t  sub_step;
      int32_t  sub_last_use; // exclusive
      int32_t  alias_of_sub_step = -1;  // -1: not an in-place alias
      int32_t  sub_first_use = -1;  // -1: same as sub_step

    };

  virtual std::vector<SubLayerInfo> GetActivationTensorInfo() { return {}; }

  virtual int32_t GetSubStepCount() const { return 1; }

  /// (input index, exclusive last-use sub-step) pairs. Empty means every
  /// input lives until the last sub-step.
  virtual std::vector<std::pair<size_t, int32_t>> GetInputLastUseHints() const {
    return {};
  }

protected:
  std::unique_ptr<Tensor<Tout>> output_tensor_;
  AutocastPolicy autocast_policy_ = AutocastPolicy::PASSTHROUGH;
};

} // namespace oris_ai
