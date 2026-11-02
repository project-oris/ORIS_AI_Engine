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

#include "oris_ai/accelerator/cuda/unique_aux_stream.h"
#include "oris_ai/tensor/fp16/tensor_fp16.h"
#include "oris_ai/tensor/mmap_populate_worker.h"
#include "oris_ai/tensor/q8_0/tensor_q8_0.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace oris_ai {

// One weight matrix of a ring block: an F16 or a Q8_0 tensor and its source.
struct VisionWeightRingMember {
  Tensor<half>* f16 = nullptr;
  Tensor<block_q8_0>* q8 = nullptr;
  const void* source = nullptr;
  Q8_0ByteLayout layout = Q8_0ByteLayout::kNative;
  const void* superseded = nullptr;
  size_t superseded_bytes = 0;
  size_t bytes = 0;
  size_t offset = 0;
};

// Holds the block weights of an encoder in a few slots of device memory.
// A failed encode must call Abandon. Sources stay mapped until ReleaseSources.
class VisionWeightRing {
 public:
  VisionWeightRing() = default;
  ~VisionWeightRing();
  VisionWeightRing(const VisionWeightRing&) = delete;
  VisionWeightRing& operator=(const VisionWeightRing&) = delete;

  // Members have the same order and sizes in every block.
  bool Setup(std::vector<std::vector<VisionWeightRingMember>> blocks,
             size_t slots, const std::string& scope, const char* label,
             std::string* reason);

  bool enabled() const { return slots_ != 0; }
  size_t slots() const { return slots_; }
  size_t slot_bytes() const { return slot_bytes_; }
  size_t block_bytes() const { return block_bytes_; }
  size_t block_count() const { return blocks_.size(); }
  uint64_t encodes() const { return encodes_; }
  size_t staged_blocks() const { return next_stage_; }
  double stage_ms() const { return stage_ms_; }
  double rearm_ms() const { return rearm_ms_; }
  bool sources_released() const { return sources_released_; }

  // Fill byte before each staging; -1 for none.
  void SetPoison(int poison) { poison_ = poison; }
  void SetPopulateSpans(std::vector<std::vector<MmapSpan>> spans);

  // `populate` must outlive the encode. Throws after ReleaseSources.
  void BeginEncode(cudaStream_t compute,
                   MmapPopulateWorker* populate = nullptr);
  // Stages blocks, in order, through block `limit - 1`.
  void StageThrough(size_t limit);
  void WaitBlock(size_t l, cudaStream_t compute);
  // Call when block l's kernels are queued on `compute`.
  void FinishBlock(size_t l, cudaStream_t compute);
  // After a failed encode: drains the streams. Never throws.
  void Abandon(cudaStream_t compute) noexcept;
  // The sources stay; the next encode borrows them again.
  void ClearBorrows();
  // No encode runs afterwards; the mapping may go away.
  void ReleaseSources();
  void LogEncode() const;

 private:
  void StageBlock_(size_t j);

  std::vector<std::vector<VisionWeightRingMember>> blocks_;
  size_t slots_ = 0;
  size_t slot_bytes_ = 0;
  size_t block_bytes_ = 0;
  size_t root_request_id_ = 0;
  std::string scope_;
  std::string label_;
  void* base_ = nullptr;
  UniqueAuxStream stream_;
  std::vector<cudaEvent_t> upload_done_;
  std::vector<cudaEvent_t> slot_free_;
  std::vector<int64_t> uploaded_block_;
  std::vector<int64_t> freed_block_;
  cudaEvent_t encode_fence_ = nullptr;
  std::vector<std::vector<MmapSpan>> populate_spans_;
  MmapPopulateWorker* populate_ = nullptr;
  int poison_ = -1;
  uint64_t encodes_ = 0;
  size_t next_stage_ = 0;
  double stage_ms_ = 0.0;
  double rearm_ms_ = 0.0;
  bool sources_released_ = false;
};

template <typename T>
class Linear;

// False for a weight the ring cannot hold.
template <typename T>
bool DescribeVisionWeightRingMember(Linear<T>* linear,
                                    VisionWeightRingMember* member);
// Call after Setup.
template <typename T>
void ArmVisionWeightRingMember(Linear<T>* linear);

// `fallback` when the variable is unset or not a number.
size_t VisionWeightRingSlotsFromEnv(size_t fallback);
// A byte, or -1.
int VisionWeightRingPoisonFromEnv();

}  // namespace oris_ai
