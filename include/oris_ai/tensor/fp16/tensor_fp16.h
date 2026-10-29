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

#include <cuda_fp16.h>
#include <functional>
#include <numeric>

#include "oris_ai/tensor/tensor.h"

namespace oris_ai {

// FP16 tensor. GPU-resident by default.
template <>
class Tensor<half> {
public:
  Tensor(const std::vector<size_t>& shape,
         bool cpu_only = false,
         PoolType pool_type = PoolType::STATIC);

  // Wraps GPU memory the caller owns; it is never freed by the tensor.
  Tensor(const std::vector<size_t>& shape, half* external_gpu_data);

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&&) = delete;

  ~Tensor();

  half* GetGPUDataPtr();

  const half* GetConstGPUDataPtr() const;

  half* GetCPUDataPtr();

  const half* GetConstCPUDataPtr(bool keep_head_gpu = false) const;

  // The tensor does not own `external_ptr`; it must outlive the borrow.
  void SetBorrowedCPUData(const half* external_ptr, size_t count);

  bool IsBorrowed() const { return is_borrowed_; }

  void ReleaseBorrowed();

  // Call only after all device work on the tensor has finished.
  void InvalidatePlannedGPUStorage();

  void RearmBorrowed(const half* external_ptr);

  void ClearBorrowedState() {
    if (!is_borrowed_) return;
    cpu_data_ptr_ = nullptr;
    is_borrowed_ = false;
  }

  void SetCPUData(half init);

  // Exchanges the storage of two planner-registered GPU tensors; no bytes move.
  void SwapStorage(Tensor<half>& other);

  void ResizeExternalGPU(const std::vector<size_t>& shape);

  bool IsExternalGPU() const { return is_external_gpu_; }
  void To(Device device);

  inline Device GetDevice() const { return head_; }

  inline const std::vector<size_t>& GetShape() const { return shape_; }

  inline size_t GetTotalCount() const { return total_count_; }

  inline size_t GetTotalBytes() const { return total_bytes_; }

  inline PoolType GetPoolType() const { return pool_type_; }

  inline size_t GetGPURequestId() const { return gpu_request_id_; }

  inline void SetViewShape(const std::vector<size_t>& view_shape) {
    view_shape_ = view_shape;
  }

  inline const std::vector<size_t>& GetViewShape() const { return view_shape_; }

  inline const std::vector<size_t>& GetEffectiveShape() const {
    return view_shape_.empty() ? shape_ : view_shape_;
  }

  inline size_t GetEffectiveElementCount() const {
    const auto& eff_shape = GetEffectiveShape();
    return std::accumulate(eff_shape.begin(), eff_shape.end(),
                           static_cast<size_t>(1), std::multiplies<size_t>());
  }

private:
  std::vector<size_t> shape_;
  std::vector<size_t> view_shape_;
  size_t total_count_;
  size_t total_bytes_;
  Device head_;

  half* cpu_data_ptr_;
  half* gpu_data_ptr_;
  bool cpu_only_;
  bool is_borrowed_ = false;
  bool is_external_gpu_ = false;
  PoolType pool_type_;
  size_t gpu_request_id_;
  std::string model_scope_id_;

  void AllocateCPUMemory();

  void AllocateGPUMemory();

  void EnsureCPUDataReady(bool keep_head_gpu = false);

  void EnsureGPUDataReady();
};

}  // namespace oris_ai
