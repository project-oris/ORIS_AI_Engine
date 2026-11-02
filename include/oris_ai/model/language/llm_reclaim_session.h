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

#include <unistd.h>

#include <vector>

#include "oris_ai/gguf_reader/gguf_reader.h"
#include "oris_ai/model/language/llm_progressive_reclaim.h"

#include <glog/logging.h>

namespace oris_ai {

// The GGUF mmap of a decoder.
inline std::vector<MmapReclaimWindow> DecoderReclaimWindows(
    const GGUFReader& reader) {
  std::vector<MmapReclaimWindow> windows;
  const char* lo = static_cast<const char*>(reader.get_mmap_data());
  if (lo != nullptr) windows.push_back({lo, lo + reader.get_mmap_size()});
  return windows;
}

// The reader's mapping and its companion mappings.
inline std::vector<MmapReclaimWindow> ReaderReclaimWindow(
    const GGUFReader& reader) {
  std::vector<MmapReclaimWindow> windows;
  const char* lo = static_cast<const char*>(reader.get_mmap_data());
  if (lo != nullptr) windows.push_back({lo, lo + reader.get_mmap_size()});
  for (const auto& companion : reader.reclaim_companions()) {
    const char* base = static_cast<const char*>(companion.first);
    windows.push_back({base, base + companion.second});
  }
  return windows;
}

inline long ReclaimPageSize() { return sysconf(_SC_PAGESIZE); }

// First-Forward reclaim state of a module; call Finish after the tail.
struct ModuleReclaimState {
  ProgressiveMmapReclaimer reclaimer;
  bool done = false;

  void Bind(const void* lo, const void* hi) {
    reclaimer.Configure(lo, hi, ReclaimPageSize());
    reclaimer.SetReclaimEpoch(nullptr);
  }
  void Bind(GGUFReader& reader) {
    reclaimer.Configure(ReaderReclaimWindow(reader), ReclaimPageSize());
    reclaimer.SetReclaimEpoch(reader.GetOrCreateMmapReclaimEpoch());
    const bool inline_reclaim = ProgressiveReclaimDisabledByEnv() ||
                                ProgressiveMmapReclaimer::InlineMadvise();
    reclaimer.SetReclaimQueue(inline_reclaim ? nullptr :
        reader.GetOrCreateMmapReclaimQueue(
            ProgressiveMmapReclaimer::MaxOutstandingBytes()));
  }
  bool ShouldArm(bool on_gpu) const {
    return on_gpu && !done && reclaimer.configured() &&
           !ProgressiveReclaimDisabledByEnv();
  }
  void Finish(const char* tag) {
    reclaimer.FinishFirstForward(tag);
    done = true;
  }
};

inline void BindDecoderReclaim(ProgressiveMmapReclaimer& reclaimer,
                               GGUFReader& reader) {
  reclaimer.Configure(DecoderReclaimWindows(reader), ReclaimPageSize());
  reclaimer.SetReclaimEpoch(reader.GetOrCreateMmapReclaimEpoch());
  const bool inline_reclaim = ProgressiveMmapReclaimer::InlineMadvise();
  reclaimer.SetReclaimQueue(inline_reclaim ? nullptr :
      reader.GetOrCreateMmapReclaimQueue(
          ProgressiveMmapReclaimer::MaxOutstandingBytes()));
}

}  // namespace oris_ai
