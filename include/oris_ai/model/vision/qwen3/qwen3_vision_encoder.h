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

#include "oris_ai/layer/activation.h"
#include "oris_ai/layer/layer.h"
#include "oris_ai/layer/layernorm.h"
#include "oris_ai/layer/linear.h"
#include "oris_ai/layer/custom_vision/bias_add_gpu.h"
#include "oris_ai/layer/custom_vision/f16_scratch.h"
#include "oris_ai/layer/custom_language/patch_embed_im2col.h"
#include "oris_ai/layer/custom_vision/qwen3_vision_attention.h"
#include "oris_ai/gguf_reader/qwen3vl_layer_info.h"
#include "oris_ai/gguf_reader/gguf_release_util.h"
#include "oris_ai/model/language/llm_reclaim_session.h"
#include "oris_ai/model/vlm/vision_weight_ring.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace oris_ai {

// Qwen3-VL ViT vision encoder.
template <typename T>
class Qwen3VisionEncoder : public HiddenLayerAbstract<T> {
 public:
  Qwen3VisionEncoder(const std::string& layer_name, Device target_device);
  ~Qwen3VisionEncoder() override;

  void SetModelScope(const std::string& scope_id);
  const std::string& model_scope_id() const { return model_scope_id_; }

  // Mandatory before Init; post-merge token count.
  void SetMaxImageTokens(size_t max_tokens);

  // Set before the weights are bound.
  void SetVisionPrecision(PlanVisionPrecision precision) { vision_precision_ = precision; }

  // Set before the weights are bound.
  void SetVisionQ8FragMaxRows(size_t rows) { vision_q8_frag_max_rows_ = rows; }

  // 0 keeps every block resident.
  void SetWeightRing(size_t slots) { ring_requested_slots_ = slots; }
  bool weight_ring_enabled() const { return ring_.enabled(); }
  const VisionWeightRing& weight_ring() const { return ring_; }

  void SetMaxImagePatchGrid(size_t h_patches_max, size_t w_patches_max);

  // Applied at InitVisionFromRaw.
  void SetAllLinearF16Mode(LinearF16Mode mode);

  void SetF16Scratch(F16Scratch<T>* scratch);
  size_t f16_native_linear_count() const { return f16_native_linears_.size(); }
  size_t f16_scratch_input_elems() const { return f16_scratch_in_elems_; }
  size_t f16_scratch_output_elems() const { return f16_scratch_out_elems_; }

  void InitVisionFromRaw(const std::vector<Qwen3VLLayerInfo>& vit_layers,
                         const Qwen3VisionConfig& cfg,
                         const std::vector<bool>& is_deepstack_layers,
                         const void* mmap_base,
                         MmapInitSourceLedger* init_sources = nullptr);

  bool CanResizeTokenEnvelope() const;

  void ResizeTokenEnvelope(size_t max_tokens);

  void SetInputImage(Tensor<T>* img_3hw,
                     size_t h_patches, size_t w_patches);

  // A Forward() before the next SetInputImage fails its input check.
  void ClearInputImage();

  // nullptr unless VlmPlannedImageInputSelected().
  Tensor<T>* InputImageBuffer() { return input_image_buf_.get(); }

  void Forward() override;

  // Drains the active stream first.
  void ReleaseWeightsMemory();
  void ReleaseBorrowedWeights() override;

  void CollectActivationLiveness();

  // [n_image_tokens, (1 + n_deepstack) * projection_dim]: main, then ds rows.
  Tensor<T>* GetOutputTensor() override { return final_output_.get(); }

  int32_t GetSubStepCount() const override { return 1; }

  bool is_forward_ready()  const { return sublayers_bound_; }
  bool has_forwarded()     const { return forward_done_; }
  // 0 before the first Forward.
  uint64_t forward_serial() const { return forward_serial_; }

  void SetWeightReclaimWindow(const void* lo, const void* hi) {
    weight_reclaim_.Bind(lo, hi);
  }
  void SetWeightReclaimReader(GGUFReader& reader) {
    weight_reclaim_.Bind(reader);
  }
  void RearmWeightReclaim(GGUFReader& reader) {
    weight_reclaim_.done = false;
    weight_reclaim_.Bind(reader);
  }

  size_t n_image_tokens() const {
    return sublayers_bound_ ? (h_patches_ * w_patches_)
                            : max_image_tokens_;
  }

 private:
  template <typename U> friend struct Qwen3VisionEncoderTestAccess;

  std::unique_ptr<Tensor<T>> ds_in_scratch_;
  std::unique_ptr<Tensor<T>> ds_fc1_out_scratch_;
  std::unique_ptr<Tensor<T>> merger_in_scratch_;
  std::unique_ptr<Tensor<T>> mm0_out_scratch_;

  // One ViT block.
  struct ViTBlock {
    std::unique_ptr<LayerNorm<T>>             ln1;
    std::unique_ptr<LayerNorm<T>>             ln2;
    std::unique_ptr<Linear<T>>                qkv_proj;
    std::unique_ptr<BiasAddGPU<T>>            qkv_bias;
    std::unique_ptr<FusedQKVVisionAttention<T>> attn;
    std::unique_ptr<Linear<T>>                attn_out;
    std::unique_ptr<BiasAddGPU<T>>            attn_out_bias;
    std::unique_ptr<Linear<T>>                ffn_up;
    std::unique_ptr<BiasAddGPU<T>>            ffn_up_bias;
    std::unique_ptr<Linear<T>>                ffn_down;
    std::unique_ptr<BiasAddGPU<T>>            ffn_down_bias;
    bool                                      ffn_bias_gelu_folded = false;
    bool                                      ffn_gelu_epilogue_armed = false;
    bool                                      qkv_rope_epilogue_armed = false;
    bool                                      qkv_rope_packed = false;
    bool                                      ln1_f16_direct = false;
    bool                                      attn_f16_direct = false;
    bool                                      ln2_f16_direct = false;
    bool                                      ln1_q8_direct = false;
    bool                                      ln2_q8_direct = false;
    bool                                      resid2_next_ln1 = false;

    bool                                      has_deepstack = false;
    std::unique_ptr<LayerNorm<T>>             ds_norm;
    std::unique_ptr<Linear<T>>                ds_fc1;
    std::unique_ptr<BiasAddGPU<T>>            ds_fc1_bias;
    std::unique_ptr<Linear<T>>                ds_fc2;
    std::unique_ptr<BiasAddGPU<T>>            ds_fc2_bias;
    std::unique_ptr<Tensor<T>>                ds_output;
  };

  // Per-call geometry.
  struct ForwardGeom {
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    size_t ffn_int;
    size_t projection;
    size_t spatial_merge;
    size_t H_pre;
    size_t W_pre;
    size_t n_pos_pre;
    size_t n_pos_post;
    size_t merge_factor;
    size_t n_ds;
    float  theta_scale;
    int    sec0;
    int    sec1;
    size_t block_count;
  };

  // Init-time context.
  struct InitCtx {
    const void* mmap_base;
    const std::unordered_map<std::string, const Qwen3VLLayerInfo*>* info_by_name;
    const Qwen3VLLayerInfo* source_infos;
    MmapInitSourceLedger* init_sources;
    size_t embed_dim;
    size_t num_heads;
    size_t head_dim;
    size_t patch_size;
    size_t ffn_int;
    size_t projection;
    size_t block_count;
    size_t spatial_merge;
    size_t n_pos_pre;
    size_t n_pos_post;
    size_t n_image_tokens;
    size_t block_max_M;
    size_t merger_max_M;
    float  layer_norm_eps;
    size_t image_size;
  };

  void InitPatchEmbed_(const InitCtx& ctx);
  void InitPositionEmbed_(const InitCtx& ctx);
  void InitBlocks_(const InitCtx& ctx);
  void InitMerger_(const InitCtx& ctx);
  void ConfigureBlockFusions_();
  void NoteInitLinearCopy_(const InitCtx& ctx, const Qwen3VLLayerInfo& info,
                           const Linear<T>& linear);

  void RunPatchEmbed_(Tensor<T>* image_tensor, const ForwardGeom& geom);
  T* PreparePositionInputs_(const ForwardGeom& geom, cudaStream_t stream,
                            bool perf_enabled,
                            std::chrono::steady_clock::time_point& enqueue_t0_out,
                            std::chrono::steady_clock::time_point& enqueue_t1_out);
  void RunViTBlock_(size_t il, const ForwardGeom& geom, T* residual_dev,
                    const int32_t* pos_dev, cudaStream_t stream);
  Tensor<T>* RunMerger_(const ForwardGeom& geom, cudaStream_t stream);
  void AssembleFinalOutput_(Tensor<T>* mm2_out, const ForwardGeom& geom,
                            cudaStream_t stream);

  Device                                       target_device_;
  Qwen3VisionConfig                            config_;
  std::vector<bool>                            is_deepstack_layers_;
  size_t                                       n_deepstack_     = 0;
  size_t                                       max_image_tokens_ = 0;
  PlanVisionPrecision                          vision_precision_ = PlanVisionPrecision::kNoPlan;
  size_t                                       vision_q8_frag_max_rows_ = kVisionQ8FragUnresolved;
  size_t                                       h_patches_       = 0;
  size_t                                       w_patches_       = 0;
  bool                                         producer_write_fusion_ = true;
  bool                                         qkv_rope_pack_fusion_ = true;
  bool                                         residual_norm_fusion_ = true;
  bool                                         direct_projection_input_ = true;
  bool                                         ffn_gelu_epilogue_ = true;
  bool                                         qkv_rope_epilogue_ = true;
  bool                                         residual_epilogue_ = true;
  bool                                         next_ln1_q8_written_ = false;

  const void*                                  mmap_base_ = nullptr;

  std::unique_ptr<PatchEmbedIm2Col<T>>         patch_embd_im2col_;
  std::unique_ptr<BiasAddGPU<T>>               patch_bias_add_;

  std::unique_ptr<Tensor<T>>                   position_embd_base_gpu_;
  bool                                         position_embd_base_warmed_ = false;
  size_t                                       position_embd_base_h_ = 0;
  size_t                                       position_embd_base_w_ = 0;
  std::unique_ptr<Tensor<T>>                   position_embd_;
  size_t                                       pos_resize_cached_h_ = 0;
  size_t                                       pos_resize_cached_w_ = 0;
  bool                                         pos_resize_valid_    = false;

  cudaEvent_t                                  perf_ev_vit_start_     = nullptr;
  cudaEvent_t                                  perf_ev_resize_start_  = nullptr;
  cudaEvent_t                                  perf_ev_resize_end_    = nullptr;
  cudaEvent_t                                  perf_ev_vit_end_       = nullptr;
  bool                                         perf_events_created_   = false;

  std::vector<ViTBlock>                        blocks_;

  std::unique_ptr<LayerNorm<T>>                post_ln_;
  std::unique_ptr<Linear<T>>                   mm0_;
  std::unique_ptr<BiasAddGPU<T>>               mm0_bias_;
  std::unique_ptr<Linear<T>>                   mm2_;
  std::unique_ptr<BiasAddGPU<T>>               mm2_bias_;

  std::unique_ptr<Tensor<T>>                   final_output_;

  std::unique_ptr<Tensor<T>>                   block_residual_buf_;
  std::unique_ptr<Tensor<T>>                   conv_token_major_buf_;
  std::unique_ptr<Tensor<int32_t>>             positions_tensor_;
  std::unique_ptr<Tensor<T>>                   input_image_buf_;

  std::string                                  model_scope_id_;

  LinearF16Mode                                desired_linear_f16_mode_ =
      LinearF16Mode::DequantFallback;
  std::vector<Linear<T>*>                      f16_native_linears_;
  size_t                                       f16_scratch_in_elems_ = 0;
  size_t                                       f16_scratch_out_elems_ = 0;

  bool                                         is_initialized_ = false;
  bool                                         sublayers_bound_ = false;
  bool                                         forward_done_ = false;
  uint64_t                                     forward_serial_ = 0;
  ModuleReclaimState weight_reclaim_;
  void SetupWeightRing_();
  VisionWeightRing ring_;
  size_t ring_requested_slots_ = 0;
};

}  // namespace oris_ai
