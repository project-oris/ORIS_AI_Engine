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

#include "oris_ai/accelerator/active_scope_guard.h"
#include "oris_ai/accelerator/cuda/borrowed_aux_handle.h"
#include "oris_ai/accelerator/cuda/borrowed_aux_stream.h"
#include "oris_ai/gguf_reader/gguf_reader.h"
#include "oris_ai/gguf_reader/gguf_types.h"
#include "oris_ai/gguf_reader/qwen3vl_layer_info.h"
#include "oris_ai/gguf_reader/qwen3vl_parser.h"
#include "oris_ai/layer/custom_language/capture_aware_kernel_layer.h"
#include "oris_ai/layer/custom_language/embedding.h"
#include "oris_ai/layer/custom_language/ffn.h"
#include "oris_ai/layer/custom_language/qwen3_gqa.h"
#include "oris_ai/layer/custom_language/kv_cache.h"
#include "oris_ai/layer/custom_language/rmsnorm.h"
#include "oris_ai/model/language/image_span.h"
#include "oris_ai/model/language/kv_growth_record.h"
#include "oris_ai/model/language/language_model.h"
#include "oris_ai/model/language/llm_graph_cache.h"
#include "oris_ai/model/language/llm_progressive_reclaim.h"
#include "oris_ai/model/language/qwen3/qwen3_quant_frontier.h"
#include "oris_ai/tensor/tensor.h"

#include <cuda_fp16.h>

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <glog/logging.h>

namespace oris_ai {

using qwen3_token = int32_t;

// Qwen3-VL text decoder.
template <typename T>
class Qwen3 : public LanguageModel<T> {
 public:
  using value_type = T;

  // Hooks of the shared captured-Forward dispatcher.
  struct CapturedForwardPolicy {
    using Self = Qwen3<T>;
    static constexpr size_t kFASplitKVVariantCount = 1;

    static const Tensor<T>* SeqLenTensor(Self& self) {
      return self.tokens_tensor_.get();
    }

    static ActiveScopeGuard::AuxStreamType AuxStream(Self& /*self*/) {
      return nullptr;
    }
    static ActiveScopeGuard::AuxHandleType AuxHandle(Self& /*self*/) {
      return nullptr;
    }

    static bool IsPrefillScope(Self& self, const LlmGraphKey& key) {
      const bool f16_piece = self.prefill_f16_demand_.layers != 0 &&
          self.prefill_f16_.Find(key.M) != nullptr;
      return ((key.mode == LlmGraphMode::kPrefill ||
               key.mode == LlmGraphMode::kPrefillTextImage) &&
              key.M == self.max_prefill_seq_len_ && !f16_piece);
    }

    static auto ActiveKV(Self& self) { return self.ActiveKV(); }

    static bool AllowCapture(Self& self) {
      if (self.main_feature_replace_armed_) {
        const auto* kv = self.ActiveKV();
        if (kv != nullptr && kv->GetPosition() != 0) return false;
      }
      return self.logits_row_count_ == 1 && !self.row_serial_attn_;
    }

    static void PreDispatchGuard(Self& self) {
      if (self.tokens_tensor_ && self.HasActivePositionsTensor()) {
        CHECK_GT(self.tokens_generation_, self.last_forward_tokens_gen_)
            << "Qwen3::Forward: tokens setter not called since last Forward.";
        CHECK_GT(self.positions_generation_, self.last_forward_positions_gen_)
            << "Qwen3::Forward: positions setter not called since last Forward "
               "— stale GPU positions risk.";
      }
    }

    static void PreDispatchDebug(Self& self, size_t current_seq_len) {
      (void)self;
      (void)current_seq_len;
    }

    static void OnForwardExit(Self& self) {
      self.last_forward_tokens_gen_    = self.tokens_generation_;
      self.last_forward_positions_gen_ = self.positions_generation_;
      self.main_feature_replace_armed_ = false;
    }

    static bool PrepareSlots(Self& self, const LlmGraphKey& key,
                             size_t expected_blocks, int32_t s1, int32_t s2,
                             int32_t s3) {
      if (!self.graph_cache_->PrepareDecodeReplaySlots(key, expected_blocks, s1,
                                                       s2, s3)) {
        LOG(ERROR) << "[CUDAGraph] qwen3 decode ParamSlot prep failed "
                      "— invalidating decode cache";
        return false;
      }
      return true;
    }
  };

  explicit Qwen3(Device target_device);
  ~Qwen3() override;

  void CreateInputTensor(size_t batch_size, size_t seq_len);
  void SetMaxSeqLen(size_t max_seq_len);
  // Call with CreateInputTensor before Open().
  void SetMaxPrefillSeqLen(size_t len);
  void SetKvInitialTokens(size_t tokens);
  size_t forward_token_capacity() const { return max_forward_M_; }

  // Trailing rows the lm_head can ever emit logits for.
  void SetMaxLogitsRows(size_t rows);
  size_t max_logits_rows() const { return max_logits_rows_; }

  // Trailing rows the next Forward emits logits for.
  void SetLogitsRowCount(size_t rows);
  size_t logits_row_count() const { return logits_row_count_; }

  // Raises the logits row count for a scope and restores it on exit.
  class LogitsRowScope {
   public:
    LogitsRowScope(Qwen3<T>* model, size_t rows)
        : model_(model), previous_(model->logits_row_count()) {
      model_->SetLogitsRowCount(rows);
    }
    ~LogitsRowScope() { model_->SetLogitsRowCount(previous_); }
    LogitsRowScope(const LogitsRowScope&) = delete;
    LogitsRowScope& operator=(const LogitsRowScope&) = delete;

   private:
    Qwen3<T>* model_;
    size_t previous_;
  };

  // Returns false on a parser or open failure. Call FinalizeMemoryPlan() before
  // the first Forward().
  bool Open(const std::string& model_path);

  // Takes a reader and parser that passed the model file checks.
  bool Open(const std::string& model_path,
            std::unique_ptr<GGUFReader> reader,
            std::unique_ptr<Qwen3VLParser> parser);

  void Forward() override;

  void SetInputTokens(const qwen3_token* tokens, int32_t n_tokens);
  void SetInputPositions(const int32_t* positions, int32_t n_tokens);

  void ResetKVCache();
  void ReleaseWeightsMemory();

  bool cuda_graph_enabled() const { return !graph_cache_disabled_; }

  void FinalizeMemoryPlan();

  // nullptr once the parsing state is gone.
  const GGUFReader* MetadataReaderIfAlive() const {
    return memory_plan_finalized_ ? nullptr : gguf_reader_.get();
  }

  const Tensor<T>*       GetTokensTensor()    const { return tokens_tensor_.get(); }
  const Tensor<int32_t>* GetPositionsTensor() const { return positions_tensor_.get(); }

  bool HasActivePositionsTensor() const {
    return (cfg_.rope_mode == Qwen3RopeMode::IMROPE)
               ? (positions_tensor_ != nullptr)
               : (standard_positions_tensor_ != nullptr);
  }

  Tensor<T>* GetOutputTensor() const {
    if (this->layers_.empty()) return nullptr;
    return this->layers_.back()->GetOutputTensor();
  }

  const Qwen3GQA<T>* GetFirstGQA() const {
    for (const auto& layer : this->layers_) {
      const auto* gqa = dynamic_cast<const Qwen3GQA<T>*>(layer.get());
      if (gqa != nullptr) return gqa;
    }
    return nullptr;
  }

  size_t embedding_dim() const { return cfg_.embedding_dim; }
  PlanVisionPrecision PrefillPlanVision() const { return prefill_f16_.plan_vision; }
  size_t vocab_size() const { return cfg_.vocab_size; }
  size_t head_dim() const { return cfg_.head_dim; }
  uint32_t n_deepstack_layers() const { return cfg_.n_deepstack_layers; }
  size_t GetKVCachePosition() const {
    return kv_cache_ ? kv_cache_->GetPosition() : 0;
  }

  // Ensures the KV cache can hold `needed_tokens` rows.
  bool GrowKVCache(size_t needed_tokens);

  // The next Forward of each key runs uncaptured and recaptures.
  void InvalidateAllGraphs();
  void DropDecodeSelectGraphs() override;

  size_t kv_capacity() const {
    return kv_cache_ ? kv_cache_->GetMaxSeqLen() : 0;
  }
  bool kv_grows_in_place() const {
    return kv_cache_ && kv_cache_->grows_in_place();
  }
  size_t kv_growth_count() const {
    return kv_cache_ ? kv_cache_->growth_count() : 0;
  }
  size_t kv_cache_bytes() const {
    return kv_cache_ ? kv_cache_->GetAllocatedBytes() : 0;
  }
  const KvGrowthRecord& last_kv_growth() const { return last_kv_growth_; }
  const KvGrowthTotals& kv_growth_totals() const { return kv_growth_totals_; }

  // Pulls the KV position back after rejected trailing tokens.
  void RewindKVCache(size_t position);

  void SetRowSerialAttention(bool enable);

  bool AllQuantizedWeightsAre(ggml_type dtype) const;

  // K rows of every block first, then V rows, as raw bytes.
  bool SnapshotKVRows(size_t position, size_t rows,
                      std::vector<unsigned char>* out) const;

  void SetMultiRowResidualSplit(bool enable);

  const LlmGraphCache* graph_cache() const { return graph_cache_.get(); }

  // The second form takes GPU-resident `spans`, with max_tokens_per_span > 0.
  void SetUnifiedPrefillState(int32_t total_tokens,
                              const T* hidden_concat_gpu_ptr,
                              size_t stride,
                              int32_t deepstack_token_offset = 0,
                              int32_t deepstack_n_tokens    = 0);

  void SetUnifiedPrefillState(int32_t total_tokens,
                              const T* hidden_concat_gpu_ptr,
                              size_t stride,
                              const ImageSpan* spans,
                              int32_t n_spans,
                              int32_t max_tokens_per_span,
                              int32_t deepstack_token_offset = 0,
                              int32_t deepstack_n_tokens    = 0);

  // Stream and handle of the capture-disabled vision work.
  void SetAuxStream(BorrowedAuxStream s);
  void SetAuxHandle(BorrowedAuxHandle h);

 private:
  void ParsingModelFromGGUF();
  bool OpenImpl(const std::string& model_path,
                std::unique_ptr<GGUFReader> reader,
                std::unique_ptr<Qwen3VLParser> parser);

  void AllocatePositionTensorForMode_(size_t seq_len_envelope);

  void MakeBlockFromGGUF(uint32_t block_id,
                         const std::vector<TensorInfo>& tensors,
                         const std::map<std::string, size_t>& tensor_map);

  bool language_f16_native_ = false;
  PrefillF16Config prefill_f16_;
  PrefillF16Demand prefill_f16_demand_;
  std::unique_ptr<Tensor<half>> lang_f16_input_scratch_;
  std::unique_ptr<Tensor<half>> lang_f16_output_scratch_;
  void EnsureLanguageF16Scratch_();

  void BuildCaptureHookWalkers();

  void ForwardNonCaptured();

  void DispatchDeepStackInjection_(uint32_t il);

  void DispatchMainFeatureReplace_();

  LlmGraphKey MakeCurrentKey() const;

  bool ReadsImageCarrier() const {
    return image_present_ && main_feature_replace_armed_;
  }

  const std::vector<void*>& CollectGraphSentinels(LlmGraphMode mode) const;
  mutable std::vector<void*> graph_sentinels_scratch_;

  void WireTiedLMHead(Embedding<T>* embed,
                      Tensor<T>* lm_head_input,
                      const void* embed_source_data,
                      size_t embed_source_bytes);

  void WireUntiedLMHead(Tensor<T>* lm_head_input);

  std::unique_ptr<GGUFReader>    gguf_reader_;
  std::unique_ptr<Qwen3VLParser> parser_;
  Qwen3VLLLMConfig               cfg_;

  size_t batch_size_           = 0;
  size_t seq_len_              = 0;
  size_t max_seq_len_          = 0;
  size_t max_prefill_seq_len_  = 0;
  size_t kv_initial_tokens_    = 0;
  size_t max_forward_M_        = 0;

  std::unique_ptr<Tensor<T>>       tokens_tensor_;
  std::unique_ptr<Tensor<int32_t>> positions_tensor_;
  std::unique_ptr<Tensor<T>>       standard_positions_tensor_;

  std::unique_ptr<KVCache<half>> kv_cache_;
  KvGrowthRecord last_kv_growth_;
  KvGrowthTotals kv_growth_totals_;
  KVCache<half>* ActiveKV() const { return kv_cache_.get(); }

  std::vector<const CaptureAwareKernelLayer*> flash_attn_hooks_;
  std::vector<const CaptureAwareKernelLayer*> rope_kvwrite_hooks_;
  std::vector<const CaptureAwareKernelLayer*> mmvq_kvwrite_hooks_;
  GroupedQueryAttention<T>* capture_fa_representative_ = nullptr;

  bool post_release_weights_ = false;
  bool graph_cache_disabled_ = false;
  bool linear_shared_buffers_acquired_ = false;
  bool mmap_released_ = false;

  ProgressiveMmapReclaimer reclaimer_;
  bool progressive_reclaim_done_     = false;
  bool progressive_reclaim_disabled_ = false;
  bool memory_plan_finalized_ = false;

  std::unique_ptr<LlmGraphCache> graph_cache_;

  size_t tokens_generation_           = 0;
  size_t positions_generation_        = 0;
  size_t last_forward_tokens_gen_     = 0;
  size_t last_forward_positions_gen_  = 0;

  bool first_forward_done_ = false;

  bool cpu_embd_disabled_ = false;
  bool cpu_embd_enabled_ = false;
  Embedding<T>* token_embd_layer_ = nullptr;
  const uint8_t* embd_mmap_lo_ = nullptr;
  size_t embd_mmap_bytes_ = 0;

  FFN<T>* prev_ffn_for_fused_add_ = nullptr;

  bool             image_present_                  = false;
  bool             main_feature_replace_armed_     = false;
  const T*         hidden_concat_gpu_ptr_          = nullptr;
  // Bytes per token.
  size_t           hidden_concat_stride_           = 0;
  int32_t          total_tokens_for_unified_prefill_ = 0;
  int32_t          deepstack_token_offset_            = 0;
  int32_t          deepstack_n_tokens_               = 0;

  const ImageSpan* span_descs_gpu_ptr_             = nullptr;
  int32_t          n_spans_                        = 0;
  int32_t          max_tokens_per_span_            = 0;

  std::vector<HiddenLayerAbstract<T>*> block_tail_layers_;

  HiddenLayerAbstract<T>* output_norm_layer_ = nullptr;
  RMSNorm<T>* output_norm_rms_ = nullptr;

  size_t max_logits_rows_  = 1;
  size_t logits_row_count_ = 1;

  bool row_serial_attn_ = false;

  BorrowedAuxStream aux_stream_;
  BorrowedAuxHandle aux_handle_;

  template <typename Self> friend void RunCapturedForward(Self&);

  template <typename Self> friend void AppendBaseSentinels(const Self&, std::vector<void*>*);
  template <typename Self> friend void AppendFASplitKVSentinels(const Self&, LlmGraphMode, std::vector<void*>*);
  template <typename Self> friend void BuildCaptureHookWalkersImpl(Self&, const char*);

  template <typename U> friend struct CaptureSentinelCountTestAccess;

  template <typename U> friend struct Qwen3TestAccess;
};

}  // namespace oris_ai
