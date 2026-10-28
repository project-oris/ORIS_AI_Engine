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

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace oris_ai {

// A read-only span of a borrowed file mapping.
struct MmapSpan {
  const uint8_t* data = nullptr;
  size_t bytes = 0;
};

// Pre-faults borrowed file-mapping spans on worker threads, one batch at a
// time. A disabled worker starts no thread and ignores every call.
class MmapPopulateWorker {
 public:
  static bool EnabledByEnv();

  explicit MmapPopulateWorker(bool enable);
  ~MmapPopulateWorker();
  MmapPopulateWorker(const MmapPopulateWorker&) = delete;
  MmapPopulateWorker& operator=(const MmapPopulateWorker&) = delete;

  // Waits for the previous batch, then queues `spans`.
  void Submit(const std::vector<MmapSpan>& spans);
  // Blocks until the queued batch is done; call it before copying the spans.
  void Wait();

  static constexpr int kThreads = 4;

  size_t populated_bytes() const { return populated_bytes_; }
  size_t skipped_bytes() const { return skipped_bytes_; }
  double wait_ms() const { return wait_ms_; }
  bool failed() const { return failed_; }

 private:
  void Run(int index);

  bool enabled_ = false;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<MmapSpan> batch_;
  unsigned long long generation_ = 0;
  int busy_ = 0;
  bool pending_ = false;
  bool stop_ = false;
  bool failed_ = false;
  size_t populated_bytes_ = 0;
  size_t skipped_bytes_ = 0;
  double wait_ms_ = 0.0;
};

}  // namespace oris_ai
