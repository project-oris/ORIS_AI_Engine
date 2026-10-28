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
#include <vector>

namespace oris_ai {

// Device memory at a fixed virtual address whose backing grows in place.
class GrowableDeviceBuffer {
public:
  GrowableDeviceBuffer() = default;
  ~GrowableDeviceBuffer();

  GrowableDeviceBuffer(const GrowableDeviceBuffer&) = delete;
  GrowableDeviceBuffer& operator=(const GrowableDeviceBuffer&) = delete;
  GrowableDeviceBuffer(GrowableDeviceBuffer&& other) noexcept;
  GrowableDeviceBuffer& operator=(GrowableDeviceBuffer&&) = delete;

  // 0 when the device has no virtual memory management.
  static size_t Granularity();

  bool Reserve(size_t max_bytes);

  // Bytes beyond the previous backed size are uninitialized.
  bool Extend(size_t bytes);

  // `bytes` must be a size the buffer was backed to before.
  void Truncate(size_t bytes);

  void* data() const { return reinterpret_cast<void*>(base_); }

  size_t backed_bytes() const { return backed_; }

  size_t reserved_bytes() const { return reserved_; }

private:
  struct Mapping {
    size_t offset;
    size_t bytes;
  };

  unsigned long long base_ = 0;
  size_t reserved_ = 0;
  size_t backed_ = 0;
  int device_ = 0;
  std::vector<Mapping> mappings_;

  void Reset();
};

}  // namespace oris_ai
