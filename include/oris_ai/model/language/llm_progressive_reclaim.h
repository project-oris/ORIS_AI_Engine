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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace oris_ai {

class GGUFReader;

inline bool ProgressiveReclaimDisabledByEnv() {
  if (const char* env = std::getenv("ORIS_DISABLE_PROGRESSIVE_RECLAIM")) {
    return env[0] == '1';
  }
  return false;
}

// One contiguous mmap byte range inside window number `window`.
struct MmapReclaimRange {
  const void* start;
  size_t bytes;
  size_t window;
};

// Address window [lo, hi) of a GGUF mmap.
struct MmapReclaimWindow {
  const void* lo;
  const void* hi;
};

class MmapReclaimQueue;
class MmapReclaimEpoch;
struct ReclaimAsyncResults;

// Returns the pages of uploaded borrowed weights during a first Forward.
class ProgressiveMmapReclaimer {
 public:
  ProgressiveMmapReclaimer();
  ~ProgressiveMmapReclaimer();
  ProgressiveMmapReclaimer(const ProgressiveMmapReclaimer&) = delete;
  ProgressiveMmapReclaimer& operator=(const ProgressiveMmapReclaimer&) = delete;

  // Rebinding clears the pending set and must happen while unarmed.
  void Configure(const std::vector<MmapReclaimWindow>& windows, long pg);
  void Configure(const void* lo, const void* hi, long pg) {
    Configure(std::vector<MmapReclaimWindow>{{lo, hi}}, pg);
  }
  bool configured() const { return !windows_.empty() && page_size_ > 0; }

  // Bind at init.
  void SetReclaimQueue(std::shared_ptr<MmapReclaimQueue> queue);
  void SetReclaimEpoch(std::shared_ptr<MmapReclaimEpoch> epoch);
  void WaitForReclaims();

  // Prefer ScopedReclaimArm.
  void Arm();
  void Disarm();
  bool armed() const { return armed_; }

  // Keeps [p, p + n) only when it lies fully inside a window.
  void NoteUpload(const void* p, size_t n);

  bool TryChunkedUpload(void* dst, const void* src, size_t bytes,
                        cudaStream_t s);

  // Waits until every madvise handed over so far has completed.
  void FlushPending();
  // Does not wait.
  void FlushIfAbove();
  // Call once at the end of the first Forward.
  void FinishFirstForward(const char* tag);

  // Already submitted jobs keep their own state.
  void Reset();

  size_t pending_bytes() const { return pending_bytes_; }
  // These results are complete once FlushPending has returned.
  size_t reclaimed_bytes() const { return reclaimed_bytes_; }
  size_t flush_count() const { return flush_count_; }
  size_t chunked_bytes() const { return chunked_bytes_; }
  size_t span_count() const { return span_count_; }
  uint64_t span_digest() const { return span_digest_; }
  size_t failed_spans() const { return failed_spans_; }
  double madvise_ms() const { return madvise_ms_; }
  double forward_wait_ms() const { return forward_wait_ms_; }
  bool madvise_off_thread() const {
    return !InlineMadvise() && !reclaim_thread_unavailable_ &&
           (!reader_queue_bound_ || reclaim_thread_ != nullptr);
  }

  // In bytes, read once.
  static size_t FlushThresholdBytes();
  static size_t ChunkBytes();
  static bool InlineMadvise();
  static size_t MaxOutstandingBytes();

 private:
  int WindowIndex(const void* p, size_t n) const;
  void AppendAlignedSpan(size_t window, const char* start, const char* end,
                         std::vector<MmapReclaimRange>* spans) const;
  void SubmitPending_();
  bool UseReclaimThread_();
  void HandOff_(std::vector<MmapReclaimRange>&& spans, bool counts_as_flush);
  void DrainReclaimThread_();
  bool OnOwnerThread() const;

  std::vector<MmapReclaimWindow> windows_;
  long page_size_ = 0;
  std::vector<MmapReclaimRange> pending_;
  size_t pending_bytes_ = 0;
  bool armed_ = false;
  std::thread::id owner_;
  ProgressiveMmapReclaimer* previous_sink_ = nullptr;
  size_t reclaimed_bytes_ = 0;
  size_t flush_count_ = 0;
  size_t chunked_bytes_ = 0;
  size_t span_count_ = 0;
  uint64_t span_digest_ = 0;
  size_t failed_spans_ = 0;
  double madvise_ms_ = 0.0;
  double forward_wait_ms_ = 0.0;
  std::shared_ptr<MmapReclaimQueue> reclaim_thread_;
  std::shared_ptr<MmapReclaimEpoch> reclaim_epoch_;
  std::shared_ptr<ReclaimAsyncResults> async_results_;
  bool reader_queue_bound_ = false;
  bool reclaim_thread_unavailable_ = false;
};

// The thread's active sink.
extern thread_local ProgressiveMmapReclaimer* g_reclaim_sink;

// Arms `r` for the scope; a null `r` arms nothing.
class ScopedReclaimArm {
 public:
  explicit ScopedReclaimArm(ProgressiveMmapReclaimer* r) : r_(r) {
    if (r_ != nullptr) r_->Arm();
  }
  ~ScopedReclaimArm() {
    if (r_ != nullptr) r_->Disarm();
  }
  ScopedReclaimArm(const ScopedReclaimArm&) = delete;
  ScopedReclaimArm& operator=(const ScopedReclaimArm&) = delete;

 private:
  ProgressiveMmapReclaimer* r_;
};

// Threshold flush of the reclaimer armed on this thread, if any.
inline void FlushArmedReclaimSink() {
  if (g_reclaim_sink != nullptr) g_reclaim_sink->FlushIfAbove();
}

}  // namespace oris_ai
