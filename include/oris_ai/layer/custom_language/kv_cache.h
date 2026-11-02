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

#include "oris_ai/tensor/tensor.h"

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <memory>
#include <string>
#include <vector>

namespace oris_ai {

// Host wall-clock split of one KV cache growth.
struct KVCacheGrowthTiming {
  double allocate_ms = 0.0;
  double copy_ms = 0.0;
  double release_ms = 0.0;
};

// K and V cache of all layers, [1, max_seq_len, num_kv_heads, head_dim].
// Reference: llama.cpp cpy_k, cpy_v, ggml_set_rows.
template <typename T>
class KVCache {
public:
  KVCache(size_t n_layers, size_t max_seq_len,
          size_t num_kv_heads, size_t head_dim,
          Device target_device,
          PoolType pool_type = PoolType::KV_CACHE);

  ~KVCache() = default;

  // Stores tokens at rows `indices`, or from the cache position when nullptr.
  // Does not advance the position; call AdvancePosition() after all layers.
  void CpyK(size_t layer, const Tensor<T>* k_tensor, const int64_t* indices,
            size_t n_tokens, size_t src_seq_stride);

  // The same for V.
  void CpyV(size_t layer, const Tensor<T>* v_tensor, const int64_t* indices,
            size_t n_tokens, size_t src_seq_stride);

  // ViewShape is [1, position, num_kv_heads, head_dim].
  Tensor<T>* GetK(size_t layer);

  Tensor<T>* GetV(size_t layer);

  // Keeps the memory and the old bytes.
  void Clear() {
    cache_position_ = 0;
    high_water_ = 0;
  }

  // Call once per forward, after all layers have stored.
  void AdvancePosition(size_t n_tokens) {
    cache_position_ += n_tokens;
    if (cache_position_ > high_water_) high_water_ = cache_position_;
  }

  // Pulls the position back only; rows beyond it keep their bytes.
  void SetPosition(size_t position) {
    CHECK_LE(position, cache_position_)
        << "KVCache::SetPosition can only pull the position back";
    cache_position_ = position;
  }

  size_t GetPosition() const { return cache_position_; }

  size_t GetMaxSeqLen() const { return max_seq_len_; }

  size_t GetHighWater() const { return high_water_; }

  // Growth interface of KVCache<half>: this cache never grows.
  void FinalizePlan() {}
  bool Grow(size_t /*new_max_seq_len*/, cudaStream_t /*stream*/) {
    return false;
  }
  bool PrepareGrow(size_t /*new_max_seq_len*/) { return false; }
  void CommitGrow(cudaStream_t /*stream*/) {}
  void AbortGrow() {}
  bool has_prepared_growth() const { return false; }
  size_t GetPreparedBytes() const { return 0; }
  bool CanGrow() const { return false; }
  bool grows_in_place() const { return false; }
  size_t GrowthTarget(size_t /*needed_rows*/, size_t /*limit*/) const {
    return max_seq_len_;
  }
  const KVCacheGrowthTiming& last_growth_timing() const {
    static const KVCacheGrowthTiming kNone;
    return kNone;
  }
  // Sliding-window interface of KVCache<half>: this cache holds every row.
  size_t GetRowBase() const { return 0; }
  size_t MinRewindPosition() const { return 0; }
  bool NeedsCompaction(size_t /*n_tokens*/) const { return false; }
  size_t MaxCompactionKeepRows() const { return 0; }
  void CompactForWrite(size_t /*n_tokens*/, T* /*staging*/,
                       size_t /*staging_rows*/, cudaStream_t /*stream*/) {
    LOG(FATAL) << "KVCache<T>::CompactForWrite: the generic cache has no "
                  "sliding-window storage";
  }
  size_t compaction_count() const { return 0; }
  void MarkRewindAnchor() {}
  size_t ReachableRewindPosition(size_t target) const { return target; }
  size_t anchor_restore_count() const { return 0; }
  const std::string& scope_id() const {
    static const std::string kUnscoped;
    return kUnscoped;
  }
  size_t growth_count() const { return 0; }
  size_t GetAllocatedBytes() const {
    return 2 * n_layers_ * max_seq_len_ * seq_stride_ * sizeof(T);
  }

  size_t GetNumLayers() const { return n_layers_; }

  // num_kv_heads * head_dim.
  size_t GetRowSize() const { return seq_stride_; }

  T* GetKCacheGPUPtr(size_t layer) {
    return k_cache_[layer]->GetGPUDataPtr();
  }

  T* GetVCacheGPUPtr(size_t layer) {
    return v_cache_[layer]->GetGPUDataPtr();
  }

private:
  const size_t n_layers_;
  const size_t max_seq_len_;
  const size_t num_kv_heads_;
  const size_t head_dim_;
  const size_t seq_stride_;
  const Device target_device_;

  size_t cache_position_ = 0;
  size_t high_water_ = 0;

  std::vector<std::unique_ptr<Tensor<T>>> k_cache_;
  std::vector<std::unique_ptr<Tensor<T>>> v_cache_;
};

}  // namespace oris_ai
