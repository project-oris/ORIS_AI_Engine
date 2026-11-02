/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute
 *(ETRI)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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

#include "oris_ai/layer/linear.h"
#include "oris_ai/layer/elementwise.h"
#include "oris_ai/gguf_reader/llm_layer_info.h"
#include "oris_ai/layer/custom_language/rope.h"
#include "oris_ai/layer/custom_language/flash_attn.h"
#include "oris_ai/layer/custom_language/kv_cache.h"
#include "oris_ai/layer/custom_language/rmsnorm.h"
#include "oris_ai/layer/custom_language/fp16/kv_cache_fp16.h"

namespace oris_ai {

// Grouped query attention.
template <typename T> class GroupedQueryAttention : public HiddenLayerAbstract<T> {
public:
  // `head_dim` 0 derives d_model / num_query_heads.
  // `attn_scale` multiplies the scores; 0 uses 1/sqrt(head_dim).
  GroupedQueryAttention(const std::string& layer_name,
                        Device target_device,
                        size_t d_model, size_t num_query_heads,
                        size_t num_kv_heads,
                        size_t rope_n_dims, float rope_freq_base,
                        const float* rope_freqs_data,
                        size_t rope_freqs_count,
                        bool mask, bool add,
                        bool inplace_residual = false,
                        size_t head_dim = 0,
                        float attn_scale = 0.0f);

  ~GroupedQueryAttention();

  void InitGroupedQueryAttentionFromRaw(
    const LLMLayerInfo& q_info, const void* q_data,
    const LLMLayerInfo& k_info, const void* k_data,
    const LLMLayerInfo& v_info, const void* v_data,
    const LLMLayerInfo& o_info, const void* o_data,
    const float* rope_freqs_data, size_t rope_freqs_count,
    size_t max_seq_len,
    size_t max_M_hint,
    PoolType linear_output_pool_type = PoolType::ACTIVATION,
    PoolType rope_freq_pool_type = PoolType::STATIC,
    PoolType rope_output_pool_type = PoolType::ACTIVATION,
    PoolType matmul_output_pool_type = PoolType::ACTIVATION,
    PoolType softmax_output_pool_type = PoolType::ACTIVATION,
    PoolType kv_cache_pool_type = PoolType::KV_CACHE,
    PoolType elementwise_output_pool_type = PoolType::ACTIVATION);

  // Q-only init for a layer sharing another layer's KV cache.
  void InitGQASharedKVFromRaw(
    const LLMLayerInfo& q_info, const void* q_data,
    const LLMLayerInfo& o_info, const void* o_data,
    size_t max_seq_len,
    size_t max_M_hint,
    PoolType linear_output_pool_type = PoolType::ACTIVATION,
    PoolType rope_freq_pool_type = PoolType::STATIC,
    PoolType rope_output_pool_type = PoolType::ACTIVATION,
    PoolType matmul_output_pool_type = PoolType::ACTIVATION,
    PoolType elementwise_output_pool_type = PoolType::ACTIVATION);

  void ReleaseBorrowedWeights() override;

  std::vector<typename HiddenLayerAbstract<T>::SubLayerInfo>
      GetActivationTensorInfo() override;
  int32_t GetSubStepCount() const override { return inplace_residual_ ? 8 : 9; }
  std::vector<std::pair<size_t, int32_t>> GetInputLastUseHints() const override;

  inline Tensor<T>* GetOutputTensor() override {
    if (inplace_residual_) return this->GetInputTensor(2);
    if (add_) return elementwise_->GetOutputTensor();
    CHECK(out_proj_ != nullptr) << "GroupedQueryAttention::GetOutputTensor "
                                   "called before init";
    return out_proj_->GetOutputTensor();
  }

  // Non-owning; null before init.
  inline Linear<T>* GetVProj() { return v_proj_.get(); }

  bool AllProjectionWeightsAre(ggml_type dtype) const {
    auto matches = [dtype](const std::unique_ptr<Linear<T>>& proj) {
      return proj == nullptr || proj->GetWeightDtype() == dtype;
    };
    if (!matches(q_proj_) || !matches(out_proj_)) return false;
    if (!computes_own_kv_) return true;
    return matches(k_proj_) && matches(v_proj_);
  }

  void Forward() override;

  // `kv_cache` is not owned.
  void SetKVCache(KVCache<half>* kv_cache, size_t layer_id) {
    kv_cache_ = kv_cache;
    layer_id_ = layer_id;
  }


  // `norm` is not owned.
  void SetAttnNorm(RMSNorm<T>* norm);
  void EnablePrefillF16Gemm(const PrefillF16Config& config,
                            PrefillF16Demand* demand) {
    for (Linear<T>* layer : {q_proj_.get(), k_proj_.get(), v_proj_.get(),
                             out_proj_.get()})
      if (layer) layer->EnablePrefillF16Gemm(config, demand);
  }

  // `norm` is not owned.
  void SetFFNNormForFusedAdd(RMSNorm<T>* norm) {
    ffn_norm_for_fused_add_ = norm;
  }
  void EnableResidualEpilogue() { residual_epilogue_ = true; }

  virtual const CaptureAwareKernelLayer* GetFlashAttnCaptureHook() const;
  virtual const CaptureAwareKernelLayer* GetRopeKVWriteCaptureHook() const;
  virtual const CaptureAwareKernelLayer* GetMMVQKVWriteCaptureHook() const;

  size_t GetFlashAttnSplitKVAccRequestId() const;
  size_t GetFlashAttnSplitKVMRequestId() const;
  size_t GetFlashAttnSplitKVLRequestId() const;
  int    GetFlashAttnSplitKVMaxSplits()    const;
  size_t GetFlashAttnNumKVHeads()          const;
  size_t GetFlashAttnNumQueryHeads()       const;
  bool   GetFlashAttnMMAEnabled()          const {
    return flash_attn_ != nullptr && flash_attn_->IsMMAEnabled();
  }
  // nullptr triple when the layer is absent or scratch is unregistered.
  void GetFlashAttnSplitKVScratchPtrs(void** acc, void** m, void** l) const;

  // Call on every block before an image prefill; nullptr and 0 clear.
  void SetImageSpans(const int32_t* span_descs_gpu, int n_spans) {
    if (flash_attn_) flash_attn_->SetForwardSpans(span_descs_gpu, n_spans);
  }

  // Must not be enabled for a forward that is being graph-captured.
  void SetRowSerialAttention(bool enable) { row_serial_attn_ = enable; }

  void SetF16NativePreference(bool enabled) {
    dense_f16_native_pref_ = enabled;
  }

  // Call after the raw init.
  void SetF16Scratch(Tensor<half>* input_scratch,
                     Tensor<half>* output_scratch);

protected:
  Tensor<T>* q_ = nullptr;

  // Subclass hooks; the base does nothing.
  virtual void PreRopeQ() {}
  virtual void PreRopeK() {}

  virtual void PostVProjBeforeCpyV(Tensor<T>* v4d) { (void)v4d; }

  virtual void DispatchVCacheWrite_(Tensor<T>* v_tensor, size_t seq_len,
                                    size_t kv_seq_stride);

  size_t gqa_layer_id() const { return layer_id_; }
  KVCache<half>* gqa_kv_cache() { return kv_cache_; }

  // False on shared-KV layers: Forward() runs the Q path only.
  bool computes_own_kv_ = true;

  bool fuse_v_kv_write_allowed_ = true;
  bool fuse_decode_qkv_ = false;

  // Subclasses set this before calling the init.
  size_t fa_swa_window_ = 0;

  void SetFlashAttnQInputForSubclass(Tensor<T>* q) {
    if (flash_attn_) flash_attn_->ResetInputTensor(q, 0);
  }

  void SetFlashAttnMMAEnabledForSubclass(bool enabled) {
    if (flash_attn_) flash_attn_->SetMMAEnabled(enabled);
  }

  Tensor<T>* q_after_proj_tensor() {
    CHECK(q_proj_ != nullptr)
        << "GroupedQueryAttention::q_after_proj_tensor called before init";
    return q_proj_->GetOutputTensor();
  }
  Tensor<T>* k_after_proj_tensor() {
    CHECK(k_proj_ != nullptr)
        << "GroupedQueryAttention::k_after_proj_tensor called before init";
    return k_proj_->GetOutputTensor();
  }

  virtual void DispatchRopeQ_inplace_(Tensor<T>* q4d /* [B,S,H_q,D] */);
  virtual void DispatchRopeK_with_kv_write_(Tensor<T>* k4d,
                                            half* kv_dst,
                                            size_t dst_row_stride,
                                            size_t cache_position);

private:
  void InitFAOutputQuantization();
  bool dense_f16_native_pref_ = false;

  std::unique_ptr<Linear<T>> q_proj_;
  std::unique_ptr<Linear<T>> k_proj_;
  std::unique_ptr<Linear<T>> v_proj_;
  std::unique_ptr<RotaryPositionEmbedding<T>> rope_q_;
  std::unique_ptr<RotaryPositionEmbedding<T>> rope_k_;
  std::unique_ptr<FlashAttention<half, T>> flash_attn_;
  std::unique_ptr<Linear<T>> out_proj_;
  std::unique_ptr<ElementWise<T>> elementwise_;

  KVCache<half>* kv_cache_ = nullptr;
  size_t layer_id_ = 0;

  const float* rope_freqs_data_ = nullptr;
  size_t rope_freqs_count_ = 0;
  bool use_raw_rope_ = false;

  bool row_serial_attn_ = false;
  bool fa_f16_output_ = false;
  bool decode_residual_split_ = false;
  bool prefill_v_kv_write_ = false;

  Device target_device_;
  size_t d_model_;
  size_t num_query_heads_;
  size_t num_kv_heads_;
  size_t num_queries_per_kv_;
  size_t head_dim_;
  size_t kv_dim_;
  size_t max_seq_len_;
  float scale_;
  float attn_scale_;
  bool add_;
  bool inplace_residual_;

  // 0 not taken, 1 taken, 2 taken with V fused to the KV cache.
  int ForwardSharedQuantizedQKV();

  void* shared_q8_1_buf_ = nullptr;
  size_t shared_q8_1_buf_size_ = 0;

  RMSNorm<T>* attn_norm_ptr_ = nullptr;

  RMSNorm<T>* ffn_norm_for_fused_add_ = nullptr;
  bool residual_epilogue_ = false;
};

} // namespace oris_ai
