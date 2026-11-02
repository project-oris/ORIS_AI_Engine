/*******************************************************************************
 * Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "oris_ai/layer/custom_language/capture_aware_kernel_layer.h"

namespace oris_ai {

// Graph mode of a cache key.
enum class LlmGraphMode : uint8_t {
  kPrefill          = 0,
  kDecode           = 1,
  kPrefillTextImage = 2,
};

// Key of a captured graph.
struct LlmGraphKey {
  LlmGraphMode mode = LlmGraphMode::kPrefill;
  size_t  M = 0;
  int32_t     deepstack_token_offset = 0;
  int32_t     deepstack_n_tokens     = 0;
  const void* hidden_concat_gpu_ptr  = nullptr;
  const void* span_descs_gpu_base_ptr = nullptr;
  int32_t     n_spans                 = 0;
  int32_t     max_tokens_per_span     = 0;
  size_t      hidden_concat_stride    = 0;
  uint16_t    fa_split_variant_a      = 0;
  uint16_t    fa_split_variant_b      = 0;
  uint64_t    decode_select_id        = 0;

  bool operator<(const LlmGraphKey& o) const {
    const uintptr_t a_ptr  = reinterpret_cast<uintptr_t>(hidden_concat_gpu_ptr);
    const uintptr_t b_ptr  = reinterpret_cast<uintptr_t>(o.hidden_concat_gpu_ptr);
    const uintptr_t a_span = reinterpret_cast<uintptr_t>(span_descs_gpu_base_ptr);
    const uintptr_t b_span = reinterpret_cast<uintptr_t>(o.span_descs_gpu_base_ptr);
    return std::tie(mode, M, deepstack_token_offset, deepstack_n_tokens,
                    a_ptr, a_span, n_spans, max_tokens_per_span,
                    hidden_concat_stride, fa_split_variant_a,
                    fa_split_variant_b, decode_select_id)
         < std::tie(o.mode, o.M, o.deepstack_token_offset,
                    o.deepstack_n_tokens, b_ptr, b_span, o.n_spans,
                    o.max_tokens_per_span, o.hidden_concat_stride,
                    o.fa_split_variant_a, o.fa_split_variant_b,
                    o.decode_select_id);
  }
  bool operator==(const LlmGraphKey& o) const {
    return mode                    == o.mode
        && M                       == o.M
        && deepstack_token_offset  == o.deepstack_token_offset
        && deepstack_n_tokens      == o.deepstack_n_tokens
        && hidden_concat_gpu_ptr   == o.hidden_concat_gpu_ptr
        && span_descs_gpu_base_ptr == o.span_descs_gpu_base_ptr
        && n_spans                 == o.n_spans
        && max_tokens_per_span     == o.max_tokens_per_span
        && hidden_concat_stride    == o.hidden_concat_stride
        && fa_split_variant_a      == o.fa_split_variant_a
        && fa_split_variant_b      == o.fa_split_variant_b
        && decode_select_id        == o.decode_select_id;
  }
};

// Counters of one key.
struct LlmGraphCounters {
  size_t hit = 0;
  size_t miss = 0;
  size_t fallback = 0;
  size_t eviction = 0;
  size_t prepatch_issued = 0;
  size_t prepatch_hit = 0;
};

// The three scalars a decode replay rewrites.
struct DecodeReplayScalars {
  int32_t kv_len = 0;
  int32_t cache_position = 0;
  int32_t kv_pos = 0;

  bool operator==(const DecodeReplayScalars& o) const {
    return kv_len == o.kv_len && cache_position == o.cache_position &&
           kv_pos == o.kv_pos;
  }
};

// Scalars of the step at KV position `position` feeding `seq_len` tokens.
inline DecodeReplayScalars DecodeReplayScalarsAt(size_t position,
                                                 size_t seq_len) {
  DecodeReplayScalars scalars;
  scalars.kv_len = static_cast<int32_t>(position + seq_len);
  scalars.cache_position = static_cast<int32_t>(position);
  scalars.kv_pos = scalars.cache_position;
  return scalars;
}

// Replay state of one parametric node; not movable once built.
struct ParamSlot {
  ParamSlot() = default;
  ParamSlot(const ParamSlot&) = delete;
  ParamSlot& operator=(const ParamSlot&) = delete;
  ParamSlot(ParamSlot&&) = delete;
  ParamSlot& operator=(ParamSlot&&) = delete;

  cudaGraphNode_t node = nullptr;
  CUDA_KERNEL_NODE_PARAMS_v2 params{};
  std::vector<void*> owned_kernel_params;
  uint64_t mutating_arg_storage = 0;
  size_t mutating_arg_idx = SIZE_MAX;
  size_t mutating_arg_size = 0;
  size_t arg_count = 0;
};

// One cached graph; it moves only before its slots are built.
struct CudaGraphEntry {
  CudaGraphEntry() = default;
  CudaGraphEntry(const CudaGraphEntry&) = delete;
  CudaGraphEntry& operator=(const CudaGraphEntry&) = delete;
  CudaGraphEntry(CudaGraphEntry&&) = default;
  CudaGraphEntry& operator=(CudaGraphEntry&&) = default;

  LlmGraphKey key{};
  uint64_t capture_order = 0;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  std::vector<void*> sentinel_ptrs;
  uint64_t resource_epoch = 0;
  std::vector<cudaGraphNode_t> flash_attn_nodes;
  std::vector<cudaGraphNode_t> rope_kvwrite_nodes;
  std::vector<cudaGraphNode_t> mmvq_kvwrite_nodes;
  std::vector<size_t> flash_attn_sentinel_args;
  std::vector<size_t> rope_kvwrite_sentinel_args;
  std::vector<size_t> mmvq_kvwrite_sentinel_args;
  std::vector<cudaGraphNode_t> flash_attn_stage2_nodes;
  std::vector<ParamSlot> flash_attn_slots;
  std::vector<ParamSlot> rope_kvwrite_slots;
  std::vector<ParamSlot> mmvq_kvwrite_slots;
  std::optional<DecodeReplayScalars> exec_scalars;
  double instantiate_ms = 0.0;
  size_t node_count = 0;
};

static_assert(!std::is_copy_constructible<ParamSlot>::value &&
                  !std::is_move_constructible<ParamSlot>::value,
              "ParamSlot holds interior self-pointers once built");
static_assert(!std::is_copy_constructible<CudaGraphEntry>::value &&
                  std::is_move_constructible<CudaGraphEntry>::value,
              "CudaGraphEntry must move only via vector buffer steals");

// Key-indexed CUDA graph cache.
class LlmGraphCache {
public:
  LlmGraphCache();
  ~LlmGraphCache();

  LlmGraphCache(const LlmGraphCache&) = delete;
  LlmGraphCache& operator=(const LlmGraphCache&) = delete;

  // Returns whether the graph was replayed; the sentinels must match.
  bool TryReplay(const LlmGraphKey& key,
                 const std::vector<void*>& current_sentinels,
                 uint64_t current_resource_epoch = 0);

  // Returns the result of the synchronization that drains `stream`.
  cudaError_t Capture(const LlmGraphKey& key, cudaStream_t stream,
                      bool post_release_weights,
                      const std::vector<void*>& sentinels,
                      const std::function<void()>& forward_fn,
                      uint64_t resource_epoch = 0);

  void InvalidateKey(const LlmGraphKey& key);

  void InvalidateAll();

  void CollectParametricNodes(
      const LlmGraphKey& key,
      const std::vector<const CaptureAwareKernelLayer*>& flash_attn,
      const std::vector<const CaptureAwareKernelLayer*>& flash_attn_stage2,
      const std::vector<const CaptureAwareKernelLayer*>& rope_kvwrite,
      const std::vector<const CaptureAwareKernelLayer*>& mmvq_kvwrite);

  // Returns false on a sentinel mismatch.
  bool PrepareDecodeReplaySlots(const LlmGraphKey& key,
                                size_t expected_blocks,
                                int32_t flash_attn_sentinel,
                                int32_t rope_cache_position_sentinel,
                                int32_t mmvq_kv_pos_sentinel,
                                bool allow_absent_nodes = false);

  // Returns whether every update and the launch succeeded.
  bool ReplayDecodeStep(const LlmGraphKey& key,
                        const std::vector<void*>& current_sentinels,
                        int32_t kv_len,
                        int32_t cache_position,
                        int32_t kv_pos);

  // Writes the next step's scalars while the current step still runs.
  void PrepatchDecodeStep(const LlmGraphKey& key, int32_t kv_len,
                          int32_t cache_position, int32_t kv_pos);

  void SetDecodePrepatch(bool enabled, bool upload);
  bool decode_prepatch() const { return prepatch_enabled_; }
  bool decode_prepatch_upload() const { return prepatch_upload_; }

  // Drains `stream`, then drops the graphs holding a decode token epilogue.
  void InvalidateDecodeSelect(cudaStream_t stream);

  const LlmGraphCounters& CountersFor(const LlmGraphKey& key) const;

  const std::map<LlmGraphKey, LlmGraphCounters>& counters() const {
    return counters_;
  }

  // First slot of mode `m`, 0 if none; safe with one live decode entry at most.
  double last_instantiate_ms(LlmGraphMode m) const;

  size_t last_node_count(LlmGraphMode m) const;

  // 0 when `key` is not resident.
  double last_instantiate_ms(const LlmGraphKey& key) const;
  size_t last_node_count(const LlmGraphKey& key) const;

  // Test helpers.
  void TestCorruptSentinelForPM1(const LlmGraphKey& key, size_t idx);

  void TestRedirectDecodeSlot(const LlmGraphKey& key, cudaGraphNode_t node);

private:
  static constexpr size_t kMaxSlots = 6;

  size_t FindSlotIndex(const LlmGraphKey& key) const;
  size_t FindFirstEmptySlot() const;
  size_t FindReplacementSlot(LlmGraphMode mode) const;

  void InvalidateSlotImpl(size_t slot_idx);

  bool PushDecodeScalars(CudaGraphEntry* slot, const DecodeReplayScalars& scalars,
                         bool skip_first_mmvq);

  bool prepatch_enabled_;
  bool prepatch_upload_;

  std::array<std::optional<CudaGraphEntry>, kMaxSlots> slots_;

  uint64_t capture_order_ = 0;
  std::map<LlmGraphKey, LlmGraphCounters> counters_;
  static const LlmGraphCounters kEmptyCounters;
};

}  // namespace oris_ai
