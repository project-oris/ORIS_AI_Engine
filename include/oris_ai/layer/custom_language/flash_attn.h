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

#include <type_traits>

namespace oris_ai {

// Attention with online softmax. Tin: K/V type, Tout: Q and output type.
template <typename Tin, typename Tout = Tin>
class FlashAttention : public HiddenLayerAbstract<Tin, Tout> {
public:
  FlashAttention(const std::string& layer_name);

  ~FlashAttention() = default;

  using HiddenLayerAbstract<Tin, Tout>::SetInputTensor;

  template <typename U = Tout,
            typename = std::enable_if_t<!std::is_same_v<Tin, U>>>
  void SetInputTensor(Tensor<Tout>* q_tensor) {
    q_input_ = q_tensor;
  }

  // `index`: 0 = Q, 1 = K, 2 = V.
  template <typename T>
  void ResetInputTensor(Tensor<T>* tensor, size_t index) {
    if constexpr (std::is_same_v<Tin, Tout>) {
      this->input_tensors_.at(index) = tensor;
    } else {
      if constexpr (std::is_same_v<T, Tout>) {
        q_input_ = tensor;
      } else {
        this->input_tensors_.at(index - 1) = tensor;
      }
    }
  }

  // `index` as in ResetInputTensor.
  template <typename T>
  Tensor<T>* GetInputTensor(size_t index) {
    if constexpr (std::is_same_v<Tin, Tout>) {
      return this->input_tensors_.at(index);
    } else {
      if constexpr (std::is_same_v<T, Tout>) {
        return q_input_;
      } else {
        return this->input_tensors_.at(index - 1);
      }
    }
  }

  // `swa_window` 0 keeps full causal attention.
  virtual void InitFlashAttention(size_t num_query_heads,
                                  size_t num_kv_heads,
                                  size_t head_dim,
                                  float scale,
                                  PoolType output_pool_type = PoolType::ACTIVATION,
                                  size_t swa_window = 0);

  // Image spans of the following prefill Forward() calls.
  void SetForwardSpans(const int32_t* span_descs_gpu, int n_spans) {
    span_descs_gpu_ = span_descs_gpu;
    n_spans_ = n_spans;
  }

  virtual void Forward() = 0;

  // Bind before execution.
  void SetQ8_1Output(void* destination) {
    q8_1_output_ = destination;
    q8_1_written_ = false;
  }
  void SetForwardQ8_1Enabled(bool enabled) { forward_q8_1_enabled_ = enabled; }
  void* GetWrittenQ8_1Output() const {
    return q8_1_written_ ? q8_1_output_ : nullptr;
  }

  // Bind before a prefill (seq_len > 1) forward.
  void SetMMQOutput(void* destination, bool use_ds4) {
    mmq_output_ = destination;
    mmq_use_ds4_ = use_ds4;
    mmq_written_ = false;
  }
  void* GetWrittenMMQOutput() const {
    return mmq_written_ ? mmq_output_ : nullptr;
  }

  // Bind before a prefill (seq_len > 1) forward; rows as in the FP32 output.
  void SetF16Output(void* destination) {
    f16_output_ = destination;
    f16_written_ = false;
  }
  void* GetWrittenF16Output() const {
    return f16_written_ ? f16_output_ : nullptr;
  }

  // `kv_len` is the cache position.
  void SetForwardParams(size_t batch_size, size_t seq_len, size_t kv_len);

  void SetForwardQRowOffset(size_t q_row) { q_row_offset_ = q_row; }

  // First K/V row the cache tensors hold.
  void SetForwardKVRowBase(size_t base) { kv_row_base_ = base; }

  // Static after model init.
  void SetMMAEnabled(bool enabled) { mma_enabled_ = enabled; }
  bool IsMMAEnabled() const { return mma_enabled_; }

  // SIZE_MAX if no fixup buffer was registered.
  size_t GetFixupRequestId() const { return fixup_request_id_; }

  size_t GetSplitKVAccRequestId() const { return split_kv_acc_request_id_; }
  size_t GetSplitKVMRequestId()   const { return split_kv_m_request_id_; }
  size_t GetSplitKVLRequestId()   const { return split_kv_l_request_id_; }
  int    GetSplitKVMaxSplits()    const { return split_kv_max_splits_; }
  size_t GetNumKVHeads()          const { return num_kv_heads_; }
  size_t GetNumQueryHeads()       const { return num_query_heads_; }
  void GetSplitKVScratchPtrs(void** acc, void** m, void** l) const;

protected:
  void* q8_1_output_ = nullptr;
  bool forward_q8_1_enabled_ = false;
  bool q8_1_written_ = false;
  void* mmq_output_ = nullptr;
  bool mmq_use_ds4_ = false;
  bool mmq_written_ = false;
  void* f16_output_ = nullptr;
  bool f16_written_ = false;
  // Q input when Tin != Tout; otherwise input_tensors_[0].
  Tensor<Tout>* q_input_ = nullptr;

  size_t num_query_heads_;
  size_t num_kv_heads_;
  size_t num_queries_per_kv_;
  size_t head_dim_;
  float scale_;
  size_t swa_window_ = 0;
  bool mma_enabled_ = true;

  const int32_t* span_descs_gpu_ = nullptr;
  int n_spans_ = 0;

  size_t batch_size_;
  size_t seq_len_;
  size_t kv_len_;
  size_t q_row_offset_ = 0;
  size_t kv_row_base_ = 0;

  size_t fixup_request_id_ = SIZE_MAX;
  int    fixup_max_blocks_  = 0;

  size_t split_kv_acc_request_id_ = SIZE_MAX;
  size_t split_kv_m_request_id_   = SIZE_MAX;
  size_t split_kv_l_request_id_   = SIZE_MAX;
  int    split_kv_max_splits_     = 0;

  std::string tmp_model_scope_;
};

} // namespace oris_ai
