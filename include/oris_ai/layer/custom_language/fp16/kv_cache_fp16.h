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

#include "oris_ai/accelerator/cuda/growable_device_buffer.h"
#include "oris_ai/layer/custom_language/kv_cache.h"
#include "oris_ai/tensor/fp16/tensor_fp16.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <memory>
#include <string>
#include <vector>

namespace oris_ai {

// FP16 KV cache that can own a planner scope, grow and hold a sliding window.
template <>
class KVCache<half> {
public:
  // An empty `scope_id` keeps the cache in the caller's scope. `reserve_rows`
  // above `max_seq_len` reserves ranges for a scoped cache to grow in place.
  KVCache(size_t n_layers, size_t max_seq_len,
          size_t num_kv_heads, size_t head_dim,
          Device target_device,
          PoolType pool_type = PoolType::KV_CACHE,
          const std::string& scope_id = "",
          size_t reserve_rows = 0);

  ~KVCache();

  KVCache(const KVCache&) = delete;
  KVCache& operator=(const KVCache&) = delete;

  // Converts FP32 to FP16.
  void CpyK(size_t layer, const Tensor<float>* k_tensor, const int64_t* indices,
            size_t n_tokens, size_t src_seq_stride);

  void CpyV(size_t layer, const Tensor<float>* v_tensor, const int64_t* indices,
            size_t n_tokens, size_t src_seq_stride);

  // `cache_position` is the base row of a sequential write.
  void CpyVAt(size_t layer, const Tensor<float>* v_tensor, const int64_t* indices,
              size_t n_tokens, size_t src_seq_stride, size_t cache_position);

  Tensor<half>* GetK(size_t layer);

  Tensor<half>* GetV(size_t layer);

  void Clear() {
    cache_position_ = 0;
    high_water_ = 0;
    row_base_ = 0;
    anchor_set_ = false;
    anchor_saved_ = false;
  }

  void AdvancePosition(size_t n_tokens) {
    cache_position_ += n_tokens;
    if (cache_position_ > high_water_) high_water_ = cache_position_;
  }

  // Allocates the owned scope, or backs and zeroes the first reserved rows.
  void FinalizePlan();

  bool CanGrow() const { return in_place_ || !scope_id_.empty(); }

  // In place, K/V pointers never move.
  bool grows_in_place() const { return in_place_; }

  // Row capacity a growth to `needed_rows` should request, at most `limit`.
  size_t GrowthTarget(size_t needed_rows, size_t limit) const;

  // Keeps every written row. True when the cache holds `new_max_seq_len` rows.
  bool Grow(size_t new_max_seq_len, cudaStream_t stream);

  // First half of Grow(); the current pool keeps serving.
  bool PrepareGrow(size_t new_max_seq_len);

  // Second half of Grow(). Synchronizes `stream`, except for growth in place.
  void CommitGrow(cudaStream_t stream);

  // Releases a prepared growth without changing the cache.
  void AbortGrow();

  bool has_prepared_growth() const { return pending_rows_ != 0; }
  size_t GetPreparedBytes() const {
    return 2 * n_layers_ * pending_rows_ * seq_stride_ * sizeof(half);
  }

  const KVCacheGrowthTiming& last_growth_timing() const {
    return last_growth_timing_;
  }

  // Pulls the position back to an earlier value.
  void SetPosition(size_t position);

  // Holds only the rows a sliding window can read; call before any write.
  void ConfigureSlidingWindow(size_t window, size_t tile_rows) {
    CHECK_GT(window, 0u);
    CHECK_GT(tile_rows, 0u);
    CHECK_EQ(cache_position_, 0u)
        << "KVCacheFP16::ConfigureSlidingWindow after rows were written";
    window_ = window;
    tile_rows_ = tile_rows;
  }

  // Logical row at physical row 0.
  size_t GetRowBase() const { return row_base_; }

  // Lowest position SetPosition() may pull back to, other than 0.
  size_t MinRewindPosition() const {
    return row_base_ == 0 ? 0 : row_base_ + window_ - 1;
  }

  // When true, CompactForWrite() must run before writing `n_tokens` rows.
  bool NeedsCompaction(size_t n_tokens) const {
    return window_ != 0 && cache_position_ + n_tokens > row_base_ + max_seq_len_;
  }

  // Most rows a compaction keeps: the staging a caller provides.
  size_t MaxCompactionKeepRows() const {
    return window_ == 0 ? 0 : window_ - 1 + tile_rows_ - 1;
  }

  void CompactForWrite(size_t n_tokens, half* staging, size_t staging_rows,
                       cudaStream_t stream);

  size_t compaction_count() const { return compaction_count_; }

  void MarkRewindAnchor();

  // Largest position at or below `target` that SetPosition() can reach.
  size_t ReachableRewindPosition(size_t target) const;

  size_t anchor_restore_count() const { return anchor_restore_count_; }

  // Storage at physical row 0.
  half* GetKPhysicalGPUPtr(size_t layer) {
    return k_cache_[layer]->GetGPUDataPtr();
  }
  half* GetVPhysicalGPUPtr(size_t layer) {
    return v_cache_[layer]->GetGPUDataPtr();
  }

  size_t GetPosition() const { return cache_position_; }

  size_t GetMaxSeqLen() const { return max_seq_len_; }

  // Highest position since Clear(); the rows Grow() copies.
  size_t GetHighWater() const { return high_water_; }

  // Owned scope id, empty when the cache lives in the caller's scope.
  const std::string& scope_id() const { return scope_id_; }

  size_t growth_count() const { return growth_count_; }

  size_t GetAllocatedBytes() const {
    return 2 * n_layers_ * max_seq_len_ * seq_stride_ * sizeof(half);
  }

  size_t GetNumLayers() const { return n_layers_; }

  size_t GetRowSize() const { return seq_stride_; }

  // Based on logical rows: valid from GetRowBase() up.
  half* GetKCacheGPUPtr(size_t layer) {
    return RowBased(k_cache_[layer]->GetGPUDataPtr());
  }

  half* GetVCacheGPUPtr(size_t layer) {
    return RowBased(v_cache_[layer]->GetGPUDataPtr());
  }

  size_t GetNumKVHeads() const { return num_kv_heads_; }

  size_t GetHeadDim() const { return head_dim_; }

private:
  const size_t n_layers_;
  size_t max_seq_len_;
  const size_t num_kv_heads_;
  const size_t head_dim_;
  const size_t seq_stride_;
  const Device target_device_;
  const PoolType pool_type_;

  size_t cache_position_ = 0;
  size_t high_water_ = 0;

  size_t window_ = 0;
  size_t tile_rows_ = 0;
  size_t row_base_ = 0;
  size_t compaction_count_ = 0;

  bool anchor_set_ = false;
  bool anchor_saved_ = false;
  size_t anchor_ = 0;
  size_t anchor_base_ = 0;
  size_t anchor_rows_capacity_ = 0;
  size_t anchor_restore_count_ = 0;
  std::string anchor_scope_;
  std::unique_ptr<Tensor<half>> anchor_rows_;

  void SaveAnchorRows(cudaStream_t stream);

  half* RowBased(half* physical) const {
    return reinterpret_cast<half*>(reinterpret_cast<uintptr_t>(physical) -
                                   row_base_ * seq_stride_ * sizeof(half));
  }

  std::string scope_base_;
  std::string scope_id_;
  size_t scope_generation_ = 0;
  size_t growth_count_ = 0;

  bool in_place_ = false;
  size_t reserve_rows_ = 0;
  size_t growth_step_rows_ = 0;
  std::vector<GrowableDeviceBuffer> k_ranges_;
  std::vector<GrowableDeviceBuffer> v_ranges_;

  std::string pending_scope_;
  size_t pending_rows_ = 0;
  size_t pending_held_bytes_ = 0;
  double pending_allocate_ms_ = 0.0;
  std::vector<std::unique_ptr<Tensor<half>>> pending_k_;
  std::vector<std::unique_ptr<Tensor<half>>> pending_v_;
  KVCacheGrowthTiming last_growth_timing_;

  std::vector<std::unique_ptr<Tensor<half>>> k_cache_;
  std::vector<std::unique_ptr<Tensor<half>>> v_cache_;

  bool ReserveInPlaceStorage(size_t reserve_rows);

  bool MapInPlaceRows(size_t rows);

  void ZeroInPlaceRows(size_t from_rows, size_t to_rows, cudaStream_t stream);

  static void CreateLayerTensors(
      size_t n_layers, size_t rows, size_t num_kv_heads, size_t head_dim,
      PoolType pool_type,
      std::vector<std::unique_ptr<Tensor<half>>>* k_cache,
      std::vector<std::unique_ptr<Tensor<half>>>* v_cache);
};

}  // namespace oris_ai
