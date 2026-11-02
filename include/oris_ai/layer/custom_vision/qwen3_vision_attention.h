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
#include "oris_ai/tensor/fp16/tensor_fp16.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace oris_ai {

// Non-causal vision attention on a row-major fused QKV buffer. Run it on a
// stream that is not being captured.
template <typename T>
class FusedQKVVisionAttention : public HiddenLayerAbstract<T> {
 public:
  explicit FusedQKVVisionAttention(const std::string& layer_name);
  ~FusedQKVVisionAttention() override = default;

  void Init(size_t batch, size_t max_M,
            size_t num_heads, size_t head_dim,
            PoolType output_pool_type = PoolType::ACTIVATION,
            bool output_layout_2d = false,
            PoolType workspace_pool_type = PoolType::STATIC);

  // [batch, M, 3 * embed_dim]: Q, K and V follow each other in a row.
  void SetFusedQKVTensor(Tensor<T>* qkv);

  // Each tensor is [batch, M, embed_dim].
  void SetSeparateQKVTensors(Tensor<T>* q, Tensor<T>* k, Tensor<T>* v);

  // The fused tensor is the raw projection output: adds `qkv_bias`, rotates Q/K
  // and packs to half2 in one kernel before the attention.
  void ConfigureQwen3BiasMRoPE(Tensor<T>* qkv_bias);
  // `positions` holds two rows of cur_M values.
  void SetQwen3MRoPEInputs(const int32_t* positions, int sec0,
                           float theta_scale);
  // `rows` already holds the packed half2 rows; nullptr restores the packing.
  void SetPrepackedQkvRows(const half* rows) { prepacked_qkv_rows_ = rows; }

  // `M` must not exceed max_M from Init.
  void SetCurrentSeqLen(size_t M);

  // Call after Init; the default is 1/sqrt(head_dim).
  void SetAttnScale(float scale) { attn_scale_ = scale; }

  // `batch` must not exceed the batch from Init.
  void SetCurrentBatch(size_t batch);

  // Routes the output to `sink` as half rows.
  void SetF16OutputSink(Tensor<half>* sink);
  bool HasF16OutputSink() const { return f16_output_sink_ != nullptr; }

  // Following Forwards store half rows at `rows` until reset with nullptr.
  void SetF16OutputRows(half* rows) { f16_output_rows_ = rows; }

  void Forward() override;
  int32_t GetSubStepCount() const override { return 1; }
  Tensor<T>* GetOutputTensor() override { return this->output_tensor_.get(); }

  // Planner request id of the prepack workspace, if registered.
  size_t GetQkvWorkspaceRequestId() const { return qkv_h2_workspace_request_id_; }
  bool   QkvWorkspaceRegistered()   const { return qkv_h2_workspace_registered_; }

 private:
  // Input mode set at Init.
  enum class QKVInputMode { kFused, kFusedQwen3BiasMRoPE, kSeparate };

  size_t batch_     = 0;
  size_t cur_batch_ = 0;
  size_t max_M_     = 0;
  size_t cur_M_     = 0;
  size_t num_heads_ = 0;
  size_t head_dim_  = 0;

  float attn_scale_ = 0.f;

  bool   output_layout_2d_ = false;

  QKVInputMode qkv_input_mode_ = QKVInputMode::kFused;
  Tensor<half>* f16_output_sink_ = nullptr;
  half* f16_output_rows_ = nullptr;
  const half* prepacked_qkv_rows_ = nullptr;
  Tensor<T>* qkv_inout_ = nullptr;
  Tensor<T>* q_inout_   = nullptr;
  Tensor<T>* k_inout_   = nullptr;
  Tensor<T>* v_inout_   = nullptr;
  Tensor<T>* qkv_bias_ = nullptr;
  const int32_t* mrope_positions_ = nullptr;
  int mrope_sec0_ = 0;
  float mrope_theta_scale_ = 0.f;

  size_t qkv_h2_workspace_request_id_ = 0;
  size_t qkv_h2_workspace_h2_count_   = 0;
  bool   qkv_h2_workspace_registered_ = false;
  PoolType qkv_h2_workspace_pool_type_ = PoolType::STATIC;
  std::string qkv_h2_workspace_model_id_;
};

}  // namespace oris_ai
