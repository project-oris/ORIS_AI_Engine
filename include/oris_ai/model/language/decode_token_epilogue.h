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

#include <cstdint>

#include <cuda_runtime.h>

#include "oris_ai/tensor/tensor.h"

namespace oris_ai {

// Token selection that runs at the end of a decode step.
class DecodeTokenEpilogue {
 public:
  virtual ~DecodeTokenEpilogue() = default;

  // Non-zero while usable; different after every re-preparation.
  virtual uint64_t resource_id() const = 0;

  virtual void EnqueueEager(const Tensor<float>& logits,
                            cudaStream_t stream) = 0;

  // Returns false when the graph must not be kept.
  virtual bool EnqueueCaptured(const Tensor<float>& logits,
                               cudaStream_t stream) = 0;

  // The selection recorded in that graph becomes the pending one.
  virtual void MarkGraphLaunched(cudaStream_t stream) = 0;

  virtual void ClearPending() = 0;

  // Kept and reported when the pending selection is consumed.
  virtual void FailPending(cudaError_t error) = 0;

  // Waits for every selection enqueued or launched so far.
  virtual cudaError_t Drain() = 0;
};

}  // namespace oris_ai
