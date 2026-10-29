/*******************************************************************************
 * Copyright (c) 2025 Electronics and Telecommunications Research Institute (ETRI)
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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "oris_ai/gguf_reader/gguf_reader.h"

namespace oris_ai {

// Process-wide count of accepted madvise jobs that have not finished.
void BeginMmapMadviseJob();
void EndMmapMadviseJob();
void WaitForMmapMadviseJobs();

// Jobs and bytes of every reclaim queue that have not finished.
struct MmapReclaimBacklog {
  size_t jobs = 0;
  size_t bytes = 0;
};
MmapReclaimBacklog GetMmapReclaimBacklog();

// Ordered mapping reclaim jobs with byte-bounded admission. Jobs must not
// throw or capture the queue.
class MmapReclaimQueue {
 public:
  explicit MmapReclaimQueue(size_t max_outstanding_bytes);
  ~MmapReclaimQueue();
  MmapReclaimQueue(const MmapReclaimQueue&) = delete;
  MmapReclaimQueue& operator=(const MmapReclaimQueue&) = delete;

  // Returns the admission wait. Only a terminal mapping job may bypass the
  // byte limit.
  double Submit(std::function<void()> job, size_t bytes,
                bool wait_for_room = true,
                std::function<void()> on_admit = {});
  double Drain();
  bool Idle() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

enum class MmapReclaimStatus { kNotSubmitted, kPending, kComplete, kFailed };

// Successful madvise spans of one mapping read pass.
class MmapReclaimEpoch {
 public:
  MmapReclaimEpoch(const void* base, size_t bytes);
  void Record(const void* ptr, size_t bytes);
  std::vector<std::pair<uintptr_t, uintptr_t>> Remaining(
      const std::vector<std::pair<uintptr_t, uintptr_t>>& spans) const;
  size_t reclaimed_bytes() const;

 private:
  const uintptr_t base_;
  const uintptr_t end_;
  mutable std::mutex mu_;
  std::vector<std::pair<uintptr_t, uintptr_t>> reclaimed_;
  bool usable_ = true;
};

struct MadviseExclusion;

// State of a terminal mapping job. Reset only after the mapping's queue has
// drained.
class MmapReclaimTicket {
 public:
  MmapReclaimStatus status() const;
  void Reset();

 private:
  struct State {
    std::atomic<MmapReclaimStatus> status{MmapReclaimStatus::kPending};
  };
  std::shared_ptr<State> state_;
  friend void QueueMadviseGGUFReaderOnce(
      GGUFReader*, MmapReclaimTicket&, const char*,
      const std::vector<MadviseExclusion>&);
};

// A byte range to keep resident. An empty range reclaims the whole region.
struct MadviseExclusion {
  const void* ptr = nullptr;
  size_t bytes = 0;
};

// Init-time consumers of a reader's sources. Mark a source copied only once
// its consumer no longer reads it.
class MmapInitSourceLedger {
 public:
  explicit MmapInitSourceLedger(size_t source_count = 0) {
    sources_.reserve(source_count);
  }
  size_t AddSource(const void* ptr, size_t bytes);
  void MarkCopied(size_t source);

  // Submit once, after every consumer has finished Init.
  size_t Submit(GGUFReader& reader, const char* log_prefix) const;

 private:
  struct Source {
    MadviseExclusion range;
    bool copied = false;
  };
  std::vector<Source> sources_;
  bool has_copies_ = false;
};

// Reclaims a reader's mapped pages once. The latch is set only on success.
void MadviseGGUFReaderOnce(GGUFReader* reader, bool& already_released_latch,
                           const char* log_prefix,
                           MadviseExclusion exclude = MadviseExclusion());

void MadviseGGUFReaderOnce(GGUFReader* reader, bool& already_released_latch,
                           const char* log_prefix,
                           const std::vector<MadviseExclusion>& excludes);

// Runs after the earlier jobs on the reader's queue, or inline without one.
void QueueMadviseGGUFReaderOnce(
    GGUFReader* reader, MmapReclaimTicket& ticket, const char* log_prefix,
    const std::vector<MadviseExclusion>& excludes = {});

// Test hooks. nullptr restores normal execution.
void SetReclaimJobGateForTesting(void (*gate)());
void SetMmapMadviseForTesting(int (*fn)(void*, size_t, int));

int ReclaimMmapPages(void* ptr, size_t bytes);

}  // namespace oris_ai
