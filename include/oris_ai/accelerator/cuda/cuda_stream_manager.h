/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute (ETRI)
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

#include "oris_ai/accelerator/cuda/cuda_define.h"

namespace oris_ai {

// Process-wide CUDA stream with a per-thread override.
class CUDAStreamManager {
public:
  CUDAStreamManager();

  ~CUDAStreamManager();

  CUDAStreamManager(const CUDAStreamManager&) = delete;

  CUDAStreamManager& operator=(const CUDAStreamManager&) = delete;

  static CUDAStreamManager& GetInstance() {
    static CUDAStreamManager instance;
    return instance;
  }

  inline cudaStream_t GetStream() const {
    return active_stream_override_ ? active_stream_override_ : stream_;
  }

  inline cudaStream_t GetDefaultStream() const { return stream_; }

  // Makes GetStream() return `s` on the calling thread; nullptr clears it.
  void SetActiveStream(cudaStream_t s);

  cudaStream_t GetActiveStreamOverride() const {
    return active_stream_override_;
  }

  // The caller destroys the stream with DestroyAuxStream().
  static cudaStream_t CreateAuxStream();

  static void DestroyAuxStream(cudaStream_t s);

private:
  cudaStream_t stream_;
  static thread_local cudaStream_t active_stream_override_;
};

}  // namespace oris_ai
