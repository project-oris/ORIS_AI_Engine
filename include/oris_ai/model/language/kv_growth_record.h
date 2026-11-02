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

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "oris_ai/accelerator/cuda/cuda_define.h"
#include "oris_ai/accelerator/cuda/cuda_stream_manager.h"

namespace oris_ai {

// One /proc/self/status field in KiB; 0 when it cannot be read.
inline size_t ReadProcSelfStatusKiB(const char* field) {
  FILE* f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) return 0;
  const size_t field_len = std::strlen(field);
  char line[256];
  size_t value = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, field, field_len) == 0 && line[field_len] == ':') {
      value = static_cast<size_t>(std::strtoull(line + field_len + 1,
                                                nullptr, 10));
      break;
    }
  }
  std::fclose(f);
  return value;
}

// One decoder KV growth transaction.
struct KvGrowthRecord {
  size_t needed_tokens = 0;
  size_t from_rows = 0;
  size_t to_rows = 0;
  size_t bytes_before = 0;
  size_t bytes_after = 0;
  size_t rss_anon_kib_before = 0;
  size_t rss_anon_kib_prepared = 0;
  size_t rss_anon_kib_after = 0;
  double drain_ms = 0.0;
  double allocate_ms = 0.0;
  double copy_ms = 0.0;
  double release_ms = 0.0;
  double total_ms = 0.0;
  bool in_place = false;
};

// Totals over every growth since Open.
struct KvGrowthTotals {
  size_t count = 0;
  double total_ms = 0.0;
  size_t peak_rss_anon_kib_prepared = 0;

  void Add(const KvGrowthRecord& record) {
    ++count;
    total_ms += record.total_ms;
    if (record.rss_anon_kib_prepared > peak_rss_anon_kib_prepared) {
      peak_rss_anon_kib_prepared = record.rss_anon_kib_prepared;
    }
  }
};

inline double KvGrowthElapsedMs(std::chrono::steady_clock::time_point begin,
                                std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void LogKvGrowthRecord(const char* model, const KvGrowthRecord& r) {
  LOG(INFO) << "KV_GROWTH_V1,model=" << model
            << ",needed_tokens=" << r.needed_tokens
            << ",from_rows=" << r.from_rows << ",to_rows=" << r.to_rows
            << ",bytes_before=" << r.bytes_before
            << ",bytes_after=" << r.bytes_after
            << ",rss_anon_kib_before=" << r.rss_anon_kib_before
            << ",rss_anon_kib_prepared=" << r.rss_anon_kib_prepared
            << ",rss_anon_kib_after=" << r.rss_anon_kib_after
            << ",drain_ms=" << r.drain_ms
            << ",allocate_ms=" << r.allocate_ms
            << ",copy_ms=" << r.copy_ms
            << ",release_ms=" << r.release_ms
            << ",total_ms=" << r.total_ms
            << ",in_place=" << (r.in_place ? 1 : 0);
}

// Grows `cache` to hold needed_tokens, at most max_seq_len.
template <typename Cache, typename DropGraphs>
bool GrowSingleKvCache(Cache* cache, size_t needed_tokens, size_t max_seq_len,
                       DropGraphs drop_graphs, const char* model,
                       KvGrowthRecord* last, KvGrowthTotals* totals) {
  if (cache == nullptr) return false;
  const size_t capacity = cache->GetMaxSeqLen();
  if (needed_tokens <= capacity) return true;
  if (needed_tokens > max_seq_len) return false;
  if (!cache->CanGrow()) return false;

  const size_t target = cache->GrowthTarget(needed_tokens, max_seq_len);
  KvGrowthRecord record;
  record.needed_tokens = needed_tokens;
  record.from_rows = capacity;
  record.bytes_before = cache->GetAllocatedBytes();
  record.in_place = cache->grows_in_place();
  const auto t_begin = std::chrono::steady_clock::now();
  cudaStream_t stream = CUDAStreamManager::GetInstance().GetStream();
  if (!record.in_place) {
    drop_graphs();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  const auto t_drained = std::chrono::steady_clock::now();
  record.rss_anon_kib_before = ReadProcSelfStatusKiB("RssAnon");
  if (!cache->PrepareGrow(target)) return false;
  const auto t_prepared = std::chrono::steady_clock::now();
  record.rss_anon_kib_prepared = ReadProcSelfStatusKiB("RssAnon");
  cache->CommitGrow(stream);
  const auto t_end = std::chrono::steady_clock::now();
  record.rss_anon_kib_after = ReadProcSelfStatusKiB("RssAnon");
  record.to_rows = cache->GetMaxSeqLen();
  record.bytes_after = cache->GetAllocatedBytes();
  record.drain_ms = KvGrowthElapsedMs(t_begin, t_drained);
  record.allocate_ms = KvGrowthElapsedMs(t_drained, t_prepared);
  record.copy_ms = cache->last_growth_timing().copy_ms;
  record.release_ms = cache->last_growth_timing().release_ms;
  record.total_ms = KvGrowthElapsedMs(t_begin, t_end);
  *last = record;
  totals->Add(record);
  LogKvGrowthRecord(model, record);
  return true;
}

}  // namespace oris_ai
