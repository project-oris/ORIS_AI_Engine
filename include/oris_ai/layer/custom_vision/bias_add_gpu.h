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
#include "oris_ai/tensor/tensor.h"

#include <string>

namespace oris_ai {

// In-place broadcast bias add: target[..., i] += bias[i].
template <typename T>
class BiasAddGPU : public HiddenLayerAbstract<T> {
 public:
  explicit BiasAddGPU(const std::string& layer_name);
  ~BiasAddGPU() override = default;

  // Borrows `bias_data`, FP32 of length `out_features`.
  void InitFromRaw(size_t out_features,
                   const void* bias_data,
                   size_t bias_bytes);

  // Not owned; Forward() rewrites it.
  void SetTargetTensor(Tensor<T>* target);

  void Forward() override;
  int32_t GetSubStepCount() const override { return 1; }

  Tensor<T>* GetOutputTensor() override { return target_; }
  Tensor<T>* GetBias() { return bias_.get(); }

  // Safe before any upload; call before the mmap region is reclaimed.
  void ReleaseBorrowedWeights() override {
    if (bias_) bias_->ClearBorrowedState();
  }

 private:
  size_t out_features_ = 0;
  Tensor<T>* target_ = nullptr;
  std::unique_ptr<Tensor<T>> bias_;
};

}  // namespace oris_ai
