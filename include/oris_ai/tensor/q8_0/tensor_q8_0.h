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

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/gguf_reader/gguf_types.h"

namespace oris_ai {

// Byte order of a Q8_0 weight's blocks.
enum class Q8_0ByteLayout : uint8_t { kNative, kFragment, kRowChunks };

// Q8_0 weight tensor of GGUF blocks.
template <>
class Tensor<block_q8_0> {
public:
  Tensor(const std::vector<size_t>& shape,
        bool cpu_only = true,
        PoolType pool_type = PoolType::STATIC);

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&&) = delete;

  ~Tensor();

  block_q8_0* GetCPUDataPtr();
  const block_q8_0* GetConstCPUDataPtr() const;

  block_q8_0* GetGPUDataPtr();
  const block_q8_0* GetConstGPUDataPtr() const;

  // The tensor does not own `external_ptr`; it must outlive the borrow.
  void SetBorrowedCPUData(const block_q8_0* external_ptr, size_t count,
                          Q8_0ByteLayout source_layout = Q8_0ByteLayout::kNative);
  bool IsBorrowed() const { return is_borrowed_; }
  const block_q8_0* GetBorrowedCPUData() const {
    return is_borrowed_ ? cpu_data_ptr_ : nullptr;
  }
  void SetSupersededSource(const void* ptr, size_t bytes) {
    superseded_ptr_ = ptr;
    superseded_bytes_ = bytes;
  }

  void ReleaseBorrowed();
  void ClearBorrowedState();

  void DeclareOwnerWritesFullExtent();
  // The caller writes the whole extent on the device before the first read.
  block_q8_0* GetGPUDataPtrForFullOverwrite();
  void NoteFullOverwrite(size_t written_bytes);

  // Call only after all device work on the tensor has finished.
  void InvalidatePlannedGPUStorage();

  void RearmBorrowed(const block_q8_0* external_ptr, Q8_0ByteLayout source_layout);

  inline Device GetDevice() const { return head_; }
  inline PoolType GetPoolType() const { return pool_type_; }
  inline size_t GetGPURequestId() const { return gpu_request_id_; }

  inline uint64_t GetGpuUploadGeneration() const { return gpu_upload_generation_; }

  // A tensor is either rewritten in place or read as native blocks, not both.
  bool ReserveRewrite(Q8_0ByteLayout layout) {
    if (layout != Q8_0ByteLayout::kNative && native_consumer_) return false;
    rewrite_reservation_ = layout;
    return true;
  }
  void ReleaseRewrite() { rewrite_reservation_ = Q8_0ByteLayout::kNative; }
  Q8_0ByteLayout RewriteReservation() const { return rewrite_reservation_; }
  bool HasNativeConsumer() const { return native_consumer_; }
  bool AttachNativeConsumer() const {
    if (rewrite_reservation_ != Q8_0ByteLayout::kNative) return false;
    native_consumer_ = true;
    return true;
  }

  inline Q8_0ByteLayout GetCpuLayout() const { return cpu_layout_; }
  inline Q8_0ByteLayout GetGpuLayout() const { return gpu_layout_; }
  void NoteGpuLayout(Q8_0ByteLayout layout);

  inline const std::vector<size_t>& GetShape() const { return shape_; }
  inline size_t GetTotalCount() const { return total_count_; }
  inline size_t GetBlockCount() const { return block_count_; }
  inline size_t GetTotalBytes() const { return total_bytes_; }

  void QuantizeFromFloat(const Tensor<float>& src);
  void DequantizeToFloat(Tensor<float>& dst) const;
  float GetBlockScale(size_t block_idx) const;
  int8_t GetBlockElement(size_t block_idx, size_t elem_idx) const;

private:
  std::vector<size_t> shape_;
  size_t total_count_;
  size_t block_count_;
  size_t total_bytes_;

  block_q8_0* cpu_data_ptr_;
  bool is_borrowed_ = false;

  block_q8_0* cuda_data_ptr_ = nullptr;
  Device head_ = Device::UNINIT;
  bool cpu_only_;
  PoolType pool_type_ = PoolType::STATIC;
  size_t gpu_request_id_ = static_cast<size_t>(-1);
  uint64_t gpu_upload_generation_ = 0;
  Q8_0ByteLayout cpu_layout_ = Q8_0ByteLayout::kNative;
  Q8_0ByteLayout gpu_layout_ = Q8_0ByteLayout::kNative;
  const void* superseded_ptr_ = nullptr;
  size_t superseded_bytes_ = 0;
  Q8_0ByteLayout rewrite_reservation_ = Q8_0ByteLayout::kNative;
  mutable bool native_consumer_ = false;
  std::string model_scope_id_;

  void AllocateGPUMemory();
  void EnsureGPUDataReady();

  void AllocateCPUMemory();
  void QuantizeRowQ8_0(const float* src, block_q8_0* dst, size_t k);
  void DequantizeRowQ8_0(const block_q8_0* src, float* dst, size_t k) const;
};

}  // namespace oris_ai
