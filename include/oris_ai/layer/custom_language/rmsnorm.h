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
#include "oris_ai/tensor/tensor_memory_planner.h"
#include "oris_ai/gguf_reader/gguf_types.h"
#include <glog/logging.h>

#include <utility>
#include <vector>

namespace oris_ai {

template <typename T> class LinearQuant;

// y = (x / RMS(x)) * weight, RMS(x) = sqrt(mean(x^2) + eps).
template <typename T>
class RMSNorm : public HiddenLayerAbstract<T> {
public:
  // `only_last_token` extracts the last token of a 3D input [B, S, D].
  RMSNorm(const std::string &layer_name, bool only_last_token = false);

  ~RMSNorm() = default;

  // Borrows `weight_data` without a copy.
  void InitRMSNormFromRaw(const std::vector<size_t>& normalized_shape,
                          float eps,
                          const void* weight_data, size_t weight_bytes,
                          ggml_type dtype = GGML_TYPE_F32);

  // y = x / RMS(x). Incompatible with Q8_1 output and fused add.
  void InitRMSNormWeightless(const std::vector<size_t>& normalized_shape,
                             float eps);

  // Resizes to the current input; bind readers of the old output again.
  // Not for only_last_token layers.
  void RebuildOutputForInput();

  virtual void Forward() = 0;

  const Tensor<T>* GetWeightTensor() const { return weight_.get(); }
  float GetEpsilon() const { return static_cast<float>(eps_); }
  bool IsPlainHeadNorm(size_t dim) const {
    return weight_ && norm_size_ == dim && !only_last_token_ &&
        !q8_1_dst_ && !mmq_consumer_ && !fused_add_src_ && !fused_residual_;
  }

  bool CanWriteMMQOutput(size_t dim) const {
    return weight_ && norm_size_ == dim && !only_last_token_ &&
        dim >= 1024 && dim <= 4096 && dim % 128 == 0;
  }
  // Detaching the consumer also drops the registered prefill FP16 readers.
  void SetMMQConsumer(LinearQuant<T>* consumer) {
    mmq_consumer_ = consumer;
    if (consumer == nullptr) prefill_f16_readers_.clear();
  }
  uint64_t GetMMQGeneration() const { return mmq_generation_; }
  // Whether a forward over `rows` rows writes its consumer's FP16 or MMQ rows.
  virtual bool RunsPrefillRows(size_t rows) const {
    (void)rows;
    return false;
  }
  // Whether a forward over `rows` rows applies the SetPostNormAddSource update.
  virtual bool TakesPostNormAdd(size_t rows) const {
    (void)rows;
    return false;
  }
  // residual = (residual + RMSNorm(addend) * post_weight) * out_scale before
  // normalizing. Consumed by one forward.
  void SetPostNormAddSource(const Tensor<T>* addend, const Tensor<T>* post_weight,
                            float post_eps, float out_scale) {
    post_add_src_ = addend;
    post_add_weight_ = post_weight;
    post_add_eps_ = post_eps;
    post_add_scale_ = out_scale;
  }
  // The only readers of the output, in execution order. When all of them run
  // the FP16 GEMM, Forward does not store the FP32 output.
  void SetPrefillF16Readers(std::vector<LinearQuant<T>*> readers) {
    prefill_f16_readers_ = std::move(readers);
  }
  // Whether the last Forward left `output` (its output data) without FP32 rows.
  bool SkippedF32Output(const T* output) const {
    return output != nullptr && output == skipped_f32_output_;
  }

  void ReleaseBorrowedWeights() override {
    if (weight_) weight_->ClearBorrowedState();
  }

  // `dst` is owned by the consumer.
  void SetQ8_1Output(void* dst, size_t max_rows = size_t(-1)) {
    q8_1_dst_ = dst;
    q8_1_max_rows_ = max_rows;
  }

  // Trailing tokens of only_last_token; a count above 1 needs batch 1.
  void SetLastTokenCount(size_t count) {
    CHECK_GT(count, 0u) << "RMSNorm (" << this->GetLayerName()
                        << ") last token count must be positive";
    last_token_count_ = count;
  }

  size_t GetLastTokenCount() const { return last_token_count_; }

  bool IsQ8_1Written() const { return q8_1_written_; }

  // residual += add, then RMSNorm of the result. Consumed by the next Forward.
  void SetFusedAddSource(Tensor<T>* add_tensor, Tensor<T>* residual_tensor) {
    DCHECK_EQ(add_tensor->GetEffectiveElementCount(),
              residual_tensor->GetEffectiveElementCount())
        << "Fused add tensor size mismatch: add="
        << add_tensor->GetEffectiveElementCount()
        << " vs residual=" << residual_tensor->GetEffectiveElementCount();
    fused_add_src_ = add_tensor;
    fused_residual_ = residual_tensor;
  }

protected:
  std::unique_ptr<Tensor<T>> weight_;

  std::vector<size_t> normalized_shape_;
  double eps_;
  size_t batch_size_;
  size_t norm_size_;
  bool only_last_token_;
  size_t last_token_count_ = 1;

  void* q8_1_dst_ = nullptr;
  size_t q8_1_max_rows_ = size_t(-1);
  bool q8_1_written_ = false;

  LinearQuant<T>* mmq_consumer_ = nullptr;
  uint64_t mmq_generation_ = 0;
  std::vector<LinearQuant<T>*> prefill_f16_readers_;
  const T* skipped_f32_output_ = nullptr;

  Tensor<T>* fused_add_src_ = nullptr;
  Tensor<T>* fused_residual_ = nullptr;

  const Tensor<T>* post_add_src_ = nullptr;
  const Tensor<T>* post_add_weight_ = nullptr;
  float post_add_eps_ = 0.f;
  float post_add_scale_ = 1.f;
};

} // namespace oris_ai
