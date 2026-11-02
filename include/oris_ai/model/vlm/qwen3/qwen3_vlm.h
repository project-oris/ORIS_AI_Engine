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

#include "oris_ai/gguf_reader/gguf_release_util.h"

#include "oris_ai/accelerator/cuda/unique_aux_handle.h"
#include "oris_ai/accelerator/cuda/unique_aux_stream.h"
#include "oris_ai/gguf_reader/gguf_reader.h"
#include "oris_ai/gguf_reader/qwen3vl_parser.h"
#include "oris_ai/model/language/llama/tokenizer.h"
#include "oris_ai/model/language/qwen3/qwen3.h"
#include "oris_ai/model/vision/qwen3/qwen3_image_preprocessor.h"
#include "oris_ai/model/vision/qwen3/qwen3_vision_encoder.h"
#include "oris_ai/model/vlm/qwen3/qwen3vl_artifact_preflight.h"
#include "oris_ai/model/vlm/vlm_artifact_resolver.h"
#include "oris_ai/model/vlm/vlm_scope_util.h"
#include "oris_ai/tensor/tensor.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oris_ai {

// Multimodal container of the Qwen3-VL text decoder and vision encoder.
template <typename T>
class Qwen3VLM {
 public:
  explicit Qwen3VLM(Device target_device);
  ~Qwen3VLM();

  void SetMaxPrefillSeqLen(size_t len) { max_prefill_seq_len_ = len; }
  void SetMaxSeqLen(size_t len)        { max_seq_len_ = len; }
  // 0 allocates the whole max_seq_len.
  void SetKvInitialTokens(size_t tokens) { kv_initial_tokens_ = tokens; }
  void SetMaxImageTokens(size_t max_tokens);
  void SetMaxImagePatchGrid(size_t h_patches_max, size_t w_patches_max);
  // Ceiling on the pre-merge patch count; Open refuses a larger tower.
  void SetVisionForwardEnvelope(size_t n_pos_pre) {
    vision_forward_envelope_ = n_pos_pre;
  }

  bool Open(const std::string& llm_path,
            const std::string& mmproj_path,
            size_t max_prefill_tokens = 4096,
            const ArtifactExpectedIdentity* expected_identity = nullptr);

  // Consumes the readers and parsers of a checked pair.
  bool Open(PreparedQwen3VLArtifactPair prepared,
            size_t max_prefill_tokens = 4096);

  const ArtifactObservedIdentity* observed_artifact_identity() const {
    return artifact_identity_observed_ ? &observed_artifact_identity_ : nullptr;
  }

  // Moves the vision side to a new post-merge token envelope.
  bool ReconfigureVisionTokenEnvelope(size_t max_tokens,
                                      bool defer_scratch_trim = false);

  size_t vision_tower_builds() const { return vision_tower_builds_; }
  size_t vision_tower_resizes() const { return vision_tower_resizes_; }

  bool vision_tower_resizable() const {
    return vit_ != nullptr && tower_resize_enabled_ &&
           vit_->CanResizeTokenEnvelope();
  }

  void Reset();

  void PrepareMultimodalPrefill(int32_t total_tokens,
                                int32_t deepstack_token_offset);
  // `n_rows` carrier rows from `carrier_row_begin`, at `token_offset`.
  void PrepareMultimodalPrefillRows(int32_t total_tokens, int32_t token_offset,
                                    int32_t carrier_row_begin, int32_t n_rows);

  void ReleaseWeightsMemory();
  // Leaves the vision tower and the mmproj mapping alone.
  void ReleaseLlmWeightsMemory();

  // Call after the final encoder Forward and after its output was staged.
  void ReleaseVisionScope();

  // Call after the final multimodal prefill Forward() has completed.
  void ReleaseMultimodalCarrier();

  void SetVisionTowerDeferred(bool deferred) {
    vision_tower_deferred_ = deferred;
  }
  bool vision_tower_deferred() const { return vision_tower_deferred_; }
  bool vision_tower_alive() const { return vit_ != nullptr; }
  bool image_carrier_alive() const { return deepstack_static_buf_ != nullptr; }
  bool image_carrier_staged() const {
    return carrier_staged_ && staged_generation_ == carrier_generation_;
  }
  bool EnsureImageCarrier();
  bool EnsureVisionTower();
  void StageEncodedImage();
  // Set before Open.
  void SetVisionTowerKept(bool kept) {
    vision_tower_kept_ = kept;
    release_activations_after_encode_ = release_activations_requested_ && kept;
  }
  // Keeps the weights; EnsureVisionTower plans the buffers again.
  void ReleaseVisionActivations(const char* reason = "stage");
  bool vision_activations_released() const {
    return vision_activations_released_;
  }
  size_t vision_activation_releases() const {
    return vision_activation_releases_;
  }
  size_t vision_activation_restores() const {
    return vision_activation_restores_;
  }
  bool vision_scratch_trim_pending() const { return scratch_trim_pending_; }
  void TrimVisionScratch();
  void ReleaseVisionTower();
  // Keeps the KV cache.
  void ReleaseImageCarrier();

  bool EnableSharedVisionActivations();
  // Call before the first write of an encode into the tower's activations.
  void BeginVisionEncode();
  // Call before the decoder's first work after an encode.
  void BeginLlmPrefill();

  Qwen3<T>*               llm()      { return llm_.get(); }
  oris_ai::Tokenizer*     tokenizer() { return tokenizer_.get(); }
  Qwen3VisionEncoder<T>*  vit()      { return vit_.get(); }
  Qwen3ImagePreprocessor* preproc()  { return preproc_.get(); }

  cudaStream_t            aux_stream() const { return aux_stream_.get(); }
  cublasLtHandle_t        aux_handle() const { return aux_handle_.get(); }

  bool                    is_open() const { return is_open_; }
  // Height and width are free; their product must fit this.
  size_t                  max_image_tokens() const {
    return max_image_tokens_;
  }

  void EnsureLlmMemoryPlanFinalized();

 private:
  template <typename U> friend struct Qwen3VLMTestAccess;

  bool InitializeVision();
  void ResizeVisionTower(size_t from_tokens);
  void DropFailedVisionTower();
  bool EncoderOutputUnstaged() const;
  // Phases of a vision activation replan.
  struct ReplanTiming {
    double release_ms = 0.0;
    double scratch_ms = 0.0;
    double rewire_ms = 0.0;
    double plan_ms = 0.0;
    TensorMemoryPlanner::PlanStageTiming stages;
  };
  ReplanTiming ReplanVisionActivations(bool retain_storage = false);

  Device                                  target_device_;

  size_t max_prefill_seq_len_      = 0;
  size_t max_seq_len_              = 0;
  size_t kv_initial_tokens_        = 0;
  size_t max_image_tokens_         = 0;
  size_t vision_forward_envelope_  = 0;
  size_t carrier_tokens_           = 0;
  bool   is_open_                  = false;
  bool   vision_tower_deferred_    = false;
  bool   shrink_resident_scratch_  = true;
  bool   defer_scratch_trim_allowed_ = true;
  bool   tower_resize_enabled_     = true;
  size_t vision_tower_builds_      = 0;
  size_t vision_tower_resizes_     = 0;
  uint64_t carrier_generation_     = 0;
  uint64_t staged_generation_      = 0;
  bool     carrier_staged_         = false;
  size_t   staged_image_tokens_    = 0;
  uint64_t staged_forward_serial_  = 0;
  bool release_activations_requested_ = false;
  bool release_activations_after_encode_ = false;
  bool vision_tower_kept_ = false;
  bool vision_activations_released_ = false;
  bool scratch_trim_pending_ = false;
  size_t vision_activation_releases_ = 0;
  size_t vision_activation_restores_ = 0;
  MmapReclaimTicket mmproj_reclaim_;
  bool   carrier_released_         = false;
  ArtifactObservedIdentity observed_artifact_identity_;
  bool artifact_identity_observed_ = false;
  VlmScopePair scopes_{"qwen3vl"};
  VlmSharedActivations shared_activations_;

  // Declaration order sets the teardown order; the aux channels go last.
  UniqueAuxStream  aux_stream_;
  UniqueAuxHandle  aux_handle_;

  std::unique_ptr<Qwen3<T>>               llm_;
  PlanVisionPrecision                     vision_precision_ = PlanVisionPrecision::kNoPlan;
  size_t                                  vision_q8_frag_max_rows_ = 0;
  std::unique_ptr<oris_ai::Tokenizer>     tokenizer_;
  std::unique_ptr<Tensor<T>>              deepstack_static_buf_;
  std::unique_ptr<LinearSharedBufferLease<T>> vision_scratch_lease_;
  std::unique_ptr<Qwen3VisionEncoder<T>>  vit_;
  std::unique_ptr<F16Scratch<T>>          f16_scratch_;
  std::unique_ptr<GGUFReader>             mmproj_reader_;
  std::unique_ptr<Qwen3VLParser>          mmproj_parser_;
  std::unique_ptr<Qwen3ImagePreprocessor> preproc_;
};

}  // namespace oris_ai

