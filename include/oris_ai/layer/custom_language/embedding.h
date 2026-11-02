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

#include "oris_ai/layer/layer.h"
#include "oris_ai/tensor/tensor_memory_planner.h"
#include "oris_ai/gguf_reader/gguf_types.h"
#include "oris_ai/tensor/q8_0/tensor_q8_0.h"
#include "oris_ai/tensor/fp16/tensor_fp16.h"

namespace oris_ai {

// Token embedding; exactly one of the weight tensors is set, by dtype.
template <typename T>
class Embedding : public HiddenLayerAbstract<T> {
public:
  Embedding(const std::string &layer_name)
    : HiddenLayerAbstract<T>(layer_name),
      num_embeddings_(0),
      embedding_dim_(0),
      weight_dtype_(GGML_TYPE_F32) {}

  ~Embedding() = default;

  // Borrows `weight_data`; F16 widens to FP32 without SetF16NativePreference.
  virtual void InitEmbeddingFromRaw(size_t num_embeddings, size_t embedding_dim,
                                    const void* weight_data, size_t weight_bytes,
                                    ggml_type dtype = GGML_TYPE_F32);

  // Input [batch_size, seq_len], output [batch_size, seq_len, embedding_dim].
  virtual void Forward() = 0;

  // Multiplier of the FP32 lookup output.
  // Reference: llama.cpp ggml_scale.
  void SetOutputScale(float s) { output_scale_ = s; }
  float output_scale() const { return output_scale_; }

  // At most one of these is non-null.
  const Tensor<float>*       GetWeightF32Tensor()  const { return weight_.get(); }
  const Tensor<block_q8_0>*  GetWeightQ8_0Tensor() const { return weight_q8_0_.get(); }
  const Tensor<half>*        GetWeightF16Tensor()  const { return weight_f16_.get(); }

  // Call before InitEmbeddingFromRaw. For untied models only.
  void SetF16NativePreference(bool v) { f16_native_pref_ = v; }
  bool f16_native_pref() const { return f16_native_pref_; }

  ggml_type GetWeightDtype() const { return weight_dtype_; }

  size_t num_embeddings() const { return num_embeddings_; }
  size_t embedding_dim()  const { return embedding_dim_; }

  // Call before InitEmbeddingFromRaw. A tied model may use it only when its
  // lm_head owns a re-quantized copy.
  void EnableCpuResidentTable() { cpu_resident_table_ = true; }
  bool cpu_resident_table() const { return cpu_resident_table_; }

  // Call before graph replay; the captured graph does not hold the copy.
  virtual void StageTokenRows(const int32_t* /*ids*/, int32_t /*n*/) {}

  // nullptr when the CPU-resident mode is off.
  const void* CpuResidentScratchGpuBase() const {
    if (!cpu_resident_table_) return nullptr;
    switch (weight_dtype_) {
      case GGML_TYPE_Q8_0: return stage_rows_q8_0_ ? stage_rows_q8_0_->GetConstGPUDataPtr() : nullptr;
      case GGML_TYPE_F16:  return stage_rows_f16_  ? stage_rows_f16_->GetConstGPUDataPtr()  : nullptr;
      default: return nullptr;
    }
  }
  const void* CpuResidentIndicesGpuBase() const {
    return (cpu_resident_table_ && stage_indices_)
        ? stage_indices_->GetConstGPUDataPtr() : nullptr;
  }

  void ReleaseBorrowedWeights() override {
    if (weight_)      weight_->ClearBorrowedState();
    if (weight_q8_0_) weight_q8_0_->ClearBorrowedState();
    if (weight_f16_)  weight_f16_->ClearBorrowedState();
    if (!f16_widen_buf_.empty() && weight_ &&
        weight_->GetDevice() == Device::GPU) {
      std::vector<T>().swap(f16_widen_buf_);
    }
  }

protected:
  std::unique_ptr<Tensor<T>> weight_;

  std::vector<T> f16_widen_buf_;

  std::unique_ptr<Tensor<block_q8_0>>  weight_q8_0_;

  std::unique_ptr<Tensor<half>>        weight_f16_;
  bool f16_native_pref_ = false;

  size_t num_embeddings_;
  size_t embedding_dim_;
  ggml_type weight_dtype_;
  float output_scale_ = 1.0f;

  bool cpu_resident_table_ = false;
  // Points into the mmap region, not owned.
  const uint8_t* raw_weight_data_ = nullptr;
  size_t raw_row_bytes_ = 0;
  size_t max_stage_rows_ = 0;

  std::unique_ptr<Tensor<block_q8_0>> stage_rows_q8_0_;
  std::unique_ptr<Tensor<half>>       stage_rows_f16_;
  std::unique_ptr<Tensor<T>> stage_indices_;
};

} // namespace oris_ai
