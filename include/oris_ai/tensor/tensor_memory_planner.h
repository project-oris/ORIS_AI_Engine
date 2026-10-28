// *****************************************************************************
// Copyright (c) 2025 Electronics and Telecommunications Research Institute (ETRI)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// *****************************************************************************
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace oris_ai {

// GPU memory pool of a tensor: weights, activations or KV cache.
enum class PoolType : uint8_t {
  STATIC = 0,
  ACTIVATION = 1,
  KV_CACHE = 2
};

// How the first GPU write of a STATIC entry covers its extent.
enum class StaticFill : uint8_t {
  kZero = 0,
  kBorrowUpload = 1,
  kOwnerWrites = 2
};

// Process-wide GPU memory planner. Tensors register requests, and
// PlanAndAllocateGPU() places them in per-model pools.
class TensorMemoryPlanner {
public:
  static TensorMemoryPlanner& Instance();

  std::string CreateModelScope(const std::string& model_id = "");

  // Per-scope planning choices fixed when the scope is created.
  struct ScopeOptions {
    bool skip_overwritten_static_zero = false;
  };

  std::string CreateModelScope(const std::string& model_id,
                               const ScopeOptions& options);

  void SetActiveModel(const std::string& model_id);

  std::string GetActiveModel() const;

  // Synchronize non-blocking streams that used the scope first.
  void ReleaseModel(const std::string& model_id);

  void DetachModelStorage(const std::string& model_id);

  void ReattachModelStorage(const std::string& model_id);

  void ResetModelPool(const std::string& model_id, PoolType pool_type,
                      bool retain_storage = false);

  bool IsModelStorageDetached(const std::string& model_id) const;

  bool HasModelScope(const std::string& model_id) const;

  std::vector<std::string> ModelScopeIds() const;

  void EnsureScope(const std::string& model_id);

  void EnsureScopeClearedToLegacy();

  size_t RegisterGPURequest(size_t bytes,
                          PoolType pool_type = PoolType::STATIC);

  void PlanAndAllocateGPU(size_t alignment = 256);

  // Stage times of the calling thread's last PlanAndAllocateGPU().
  struct PlanStageTiming {
    double offsets_ms = 0.0;
    double alloc_ms = 0.0;
    double zero_ms = 0.0;
    double views_ms = 0.0;
    size_t bytes = 0;
    size_t zero_skipped_bytes = 0;
    size_t zero_ranges = 0;
    int zero_poison = -1;
  };
  static PlanStageTiming LastPlanStageTiming();

  // Valid once the pool of the request has been planned.
  void* GetGPUPtr(size_t request_id,
                  PoolType pool_type = PoolType::STATIC) const;

  void* GetGPUPtr(size_t request_id,
                  PoolType pool_type,
                  const std::string& model_id) const;

  bool IsGPUPoolPlanned(PoolType pool_type, const std::string& model_id) const;

  void Release();

  bool IsAllocated() const;

  bool IsAllocated(const std::string& model_id) const;

  size_t GetRequestCount(PoolType pool_type = PoolType::STATIC) const;

  size_t GetRequestCount(PoolType pool_type,
                        const std::string& model_id) const;

  size_t GetTotalPoolSize(PoolType pool_type = PoolType::STATIC) const;

  size_t GetTotalPoolSize(PoolType pool_type,
                          const std::string& model_id) const;

  size_t GetPoolCapacityBytes(PoolType pool_type,
                              const std::string& model_id) const;

  // Snapshot of pool utilization metrics collected during planning.
  struct PoolStatistics {
    size_t static_bytes = 0;
    size_t activation_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t activation_sum_of_all = 0;
    double sharing_ratio = 1.0;
    size_t max_concurrent = 0;
    int32_t max_concurrent_step = 0;
    size_t static_entries = 0;
    size_t activation_entries = 0;
    size_t kv_cache_entries = 0;
  };

  PoolStatistics GetPoolStatistics(const std::string& model_id = "") const;

  // One live scope's resident per-pool byte totals (profiling only).
  struct ScopeUsage {
    std::string scope_id;
    size_t static_bytes = 0;
    size_t activation_bytes = 0;
    size_t activation_capacity_bytes = 0;
    size_t activation_borrowed_bytes = 0;
    size_t kv_cache_bytes = 0;
    bool static_planned = false;
    bool activation_planned = false;
    bool kv_cache_planned = false;
  };

  std::vector<ScopeUsage> GetActiveScopeUsage() const;

  static bool StaticZeroSkipActive();

  static bool AnyStaticSkipScope();

  static int StaticSkipPoisonByte();

  bool MarkStaticOverwritten(const std::string& model_id, size_t request_id,
                             size_t gpu_bytes, StaticFill kind);

  void RevokeStaticOverwritten(const std::string& model_id, size_t request_id);

  void NoteStaticWritten(const std::string& model_id, size_t request_id);

  bool IsStaticZeroSkipped(const std::string& model_id, size_t request_id) const;

  bool StaticEntryReadable(const std::string& model_id, size_t request_id) const;

  static bool StaticWriteTrackingEnabled();
  static void OverrideStaticWriteTracking(int mode);

  static bool StaticDigestEnabled();
  static void OverrideStaticDigest(int mode);

  struct StaticFillSummary {
    size_t marked_entries = 0;
    size_t marked_bytes = 0;
    size_t skipped_entries = 0;
    size_t skipped_bytes = 0;
    size_t unwritten_entries = 0;
    size_t unwritten_bytes = 0;
    size_t zero_ranges = 0;
    size_t plan_count = 0;
  };
  StaticFillSummary GetStaticFillSummary(const std::string& model_id) const;

  void LogStaticDigestOnce(const std::string& model_id, const char* tag);

  void LendActivationStorage(const std::string& owner_id,
                             const std::string& borrower_id,
                             bool allow_split = false);

  void UnlendActivationStorage(const std::string& owner_id);

  void MarkActivationUnshared(size_t request_id);

  struct ActivationShare {
    std::string owner;
    size_t shareable_bytes = 0;
    size_t borrowed_bytes = 0;
    size_t own_bytes = 0;
    size_t own_liveness_bytes = 0;
    bool attached = false;
  };
  ActivationShare GetActivationShare(const std::string& borrower_id) const;

  bool GetActivationBorrowRange(const std::string& borrower_id, void** base,
                                size_t* bytes) const;

  bool GetActivationOwnLivenessRange(const std::string& borrower_id, void** base,
                                     size_t* bytes) const;

  // `first_use` is inclusive and `last_use` exclusive. Requests whose
  // intervals do not overlap may share memory.
  void SetLiveness(size_t request_id, PoolType pool_type,
                  int32_t first_use, int32_t last_use);

  void SetAlias(size_t alias_request_id, PoolType pool_type,
                size_t source_request_id);

  void SetAliasOffset(size_t alias_request_id, PoolType pool_type,
                      size_t source_request_id, size_t byte_offset);

  size_t ResolveAliasRoot(size_t request_id, PoolType pool_type) const;

private:
  TensorMemoryPlanner() = default;
  ~TensorMemoryPlanner();

  void ReleaseImpl(bool clear_active_model, bool device_release = true);

  TensorMemoryPlanner(const TensorMemoryPlanner&) = delete;
  TensorMemoryPlanner& operator=(const TensorMemoryPlanner&) = delete;

  struct Entry {
      size_t bytes;
      size_t offset;
      void* view = nullptr;
      int32_t first_use = -1;
      int32_t last_use  = -1;
      StaticFill fill = StaticFill::kZero;
      bool fill_revoked = false;
      bool zero_skipped = false;
      bool written = false;
      bool unshared = false;
      uint8_t arena = 0;
      static constexpr int32_t ALIAS_SENTINEL = -2;
      int32_t alias_of = -1;
      size_t alias_offset = 0;
    };

  // One memory pool.
  struct Pool {
      std::vector<Entry> entries;
      void*  gpu_pool   = nullptr;
      size_t total_bytes = 0;
      size_t alignment   = 256;
      bool   planned     = false;
      bool   detached    = false;
      size_t capacity_bytes = 0;
      void*  retained_pool  = nullptr;
      size_t retained_bytes = 0;
      size_t sum_of_all  = 0;
      double sharing_ratio = 1.0;
      size_t max_concurrent = 0;
      int32_t max_concurrent_step = 0;
      uint64_t plan_count = 0;
      size_t zero_ranges = 0;
      size_t liveness_prefix_bytes = 0;
      void* borrowed_base = nullptr;
      size_t borrowed_bytes = 0;
      size_t own_base_offset = 0;
      size_t layout_borrowed_bytes = 0;
    };

  // The three pools of one model.
  struct ModelPools {
      Pool static_pool;
      Pool activation_pool;
      Pool kv_cache_pool;
      ScopeOptions options;
      std::string lent_to;
      bool lent_split = false;
      bool digest_logged = false;

      Pool& GetPool(PoolType pool_type);
      const Pool& GetPool(PoolType pool_type) const;
    };

  // How a plan or attach fills the storage it is about to hand out.
  struct FillPolicy {
    unsigned char fill_byte = 0;
    bool skip_marked = false;
    int poison = -1;
  };

  // Everything a plan needs to know about the pool's scope.
  struct PlanContext {
    const std::string* scope_id = nullptr;
    const char* pool_name = "";
    FillPolicy fill;
    const Pool* lender = nullptr;
    const std::string* lender_id = nullptr;
    bool lender_split = false;
    const char* lender_reason = "";
    size_t lender_shareable = 0;
  };

  static void PlanPool(Pool& pool, size_t alignment, const PlanContext& ctx);

  static void PlanPoolWithSharing(Pool& pool, size_t alignment,
                                  const PlanContext& ctx);

  // Result of one offset assignment over a pool's entries.
  struct SharedLayout {
    size_t sum_of_all = 0;
    size_t max_concurrent = 0;
    int32_t max_concurrent_step = 0;
    int32_t total_steps = 0;
    size_t prefix_end = 0;
    bool by_size = false;
    size_t shared_end = 0;
  };
  static SharedLayout AssignSharedOffsets(Pool& pool, size_t alignment,
                                          bool exclude_unshared,
                                          size_t region0_cap = SIZE_MAX);

  static void CreateEntryViews(Pool& pool);

  static void ReleaseEntryViews(Pool& pool);

  static void AttachPlannedStorage(Pool& pool, const PlanContext& ctx);

  // What FillPoolStorage() did.
  struct FillResult {
    size_t skipped_bytes = 0;
    size_t ranges = 0;
  };

  static FillResult FillPoolStorage(Pool& pool, const FillPolicy& policy);

  static void DetachPlannedStorage(Pool& pool);

  static void* AcquirePoolStorage(Pool& pool, size_t bytes);

  static void FreeRetainedStorage(Pool& pool);

  static FillPolicy PolicyFor(const ModelPools& mp, PoolType pool_type);

  PlanContext ContextFor(const std::string& scope_id, PoolType pool_type) const;

  const std::string* LenderOf(const std::string& borrower_id) const;

  bool HasAttachedBorrower(const ModelPools& mp) const;

  ModelPools& GetActiveModelPools();
  const ModelPools& GetActiveModelPools() const;

  ModelPools& GetModelPools(const std::string& model_id);
  const ModelPools& GetModelPools(const std::string& model_id) const;

  ModelPools legacy_pools_;

  std::unordered_map<std::string, ModelPools> model_pools_;

  static thread_local std::string active_model_id_;

  size_t auto_id_counter_ = 0;

  mutable std::mutex mutex_;
};

namespace static_fill {

inline bool MarkBorrowUpload(const std::string& scope_id, size_t request_id,
                             PoolType pool_type, size_t gpu_bytes) {
  if (pool_type != PoolType::STATIC || scope_id.empty() ||
      !TensorMemoryPlanner::AnyStaticSkipScope()) {
    return false;
  }
  return TensorMemoryPlanner::Instance().MarkStaticOverwritten(
      scope_id, request_id, gpu_bytes, StaticFill::kBorrowUpload);
}

inline bool MarkOwnerWrites(const std::string& scope_id, size_t request_id,
                            PoolType pool_type, size_t gpu_bytes) {
  if (pool_type != PoolType::STATIC || scope_id.empty() ||
      !TensorMemoryPlanner::AnyStaticSkipScope()) {
    return false;
  }
  return TensorMemoryPlanner::Instance().MarkStaticOverwritten(
      scope_id, request_id, gpu_bytes, StaticFill::kOwnerWrites);
}

inline void Revoke(const std::string& scope_id, size_t request_id,
                   PoolType pool_type) {
  if (pool_type != PoolType::STATIC || scope_id.empty() ||
      !TensorMemoryPlanner::AnyStaticSkipScope()) {
    return;
  }
  TensorMemoryPlanner::Instance().RevokeStaticOverwritten(scope_id, request_id);
}

inline void NoteWritten(const std::string& scope_id, size_t request_id,
                        PoolType pool_type) {
  if (pool_type != PoolType::STATIC || scope_id.empty() ||
      !TensorMemoryPlanner::AnyStaticSkipScope() ||
      !TensorMemoryPlanner::StaticWriteTrackingEnabled()) {
    return;
  }
  TensorMemoryPlanner::Instance().NoteStaticWritten(scope_id, request_id);
}

inline bool Readable(const std::string& scope_id, size_t request_id,
                     PoolType pool_type) {
  if (pool_type != PoolType::STATIC || scope_id.empty() ||
      !TensorMemoryPlanner::AnyStaticSkipScope()) {
    return true;
  }
  return TensorMemoryPlanner::Instance().StaticEntryReadable(scope_id,
                                                             request_id);
}

}  // namespace static_fill

namespace activation_share {

inline void MarkUnshared(PoolType pool_type, size_t request_id) {
  if (pool_type != PoolType::ACTIVATION ||
      request_id == static_cast<size_t>(-1)) {
    return;
  }
  TensorMemoryPlanner::Instance().MarkActivationUnshared(request_id);
}

}  // namespace activation_share

}  // namespace oris_ai
