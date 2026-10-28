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

#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <cstdint>
#include <vector>

#include "oris_ai/tensor/tensor_memory_planner.h"

 #include <cuda_fp16.h>
 #include <cuda_bf16.h>

#define ORIS_TENSOR_GPU_SUBVIEW 1

namespace oris_ai {

// Device that holds the current data of a tensor.
enum class Device { 
  UNINIT,
  CPU,
  GPU
};

// GPU storage precision of Tensor<float>; the CPU copy is always FP32.
enum class DataFormat {
  FP32,
  FP16,
  BF16
};

// Multi-dimensional array with CPU memory and planner-managed GPU memory.
template <typename T>
class Tensor {
public:
  Tensor(const std::vector<size_t>& shape,
         bool cpu_only = false
         , PoolType pool_type = PoolType::STATIC
        );

  Tensor(const std::vector<size_t>& shape,
         DataFormat data_format,
         bool cpu_only = false
         , PoolType pool_type = PoolType::STATIC
        );

  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&&) = delete;

  ~Tensor();

  // The tensor does not own `external_ptr`; it must outlive the borrow.
  void SetBorrowedCPUData(const T* external_ptr, size_t count);

  bool IsBorrowed() const { return is_borrowed_; }

  void ReleaseBorrowed();

  void ClearBorrowedState() {
    if (!is_borrowed_) return;
    cpu_data_ptr_ = nullptr;
    is_borrowed_ = false;
  }

  T* GetCPUDataPtr();

  const T* GetConstCPUDataPtr(bool keep_head_gpu = false) const;

  // Like GetCPUDataPtr() without copying the GPU data back first.
  T* GetCPUDataPtrForOverwrite();

  // Upload only elements [0, count); the GPU tail keeps older data. FP32 only.
  void UploadCPUPrefixToGPU(size_t count);
  void UploadCPUPrefixToGPU(const T* source, size_t count);

  T* GetGPUDataPtr();

  const T* GetConstGPUDataPtr() const;

  inline DataFormat GetDataFormat() const { return data_format_; }

  inline bool IsFP16() const { return data_format_ == DataFormat::FP16; }
  inline bool IsBF16() const { return data_format_ == DataFormat::BF16; }

  inline size_t GetGPUBytes() const { return gpu_alloc_bytes_; }

  half* GetGPUHalfPtr();

  const half* GetConstGPUHalfPtr() const;

  __nv_bfloat16* GetGPUBF16Ptr();

  const __nv_bfloat16* GetConstGPUBF16Ptr() const;

  T& GetCPUData(const std::vector<size_t>& indices);

  void SetCPUData(T init);

  void To(Device device);

  inline Device GetDevice() const { return head_; }

  inline bool IsCPUOnly() const { return cpu_only_; }

  inline const std::vector<size_t>& GetShape() const { return shape_; }

  inline size_t GetTotalCount() const { return total_count_; }

  inline void SetViewShape(const std::vector<size_t>& view_shape) { view_shape_ = view_shape; }

  inline const std::vector<size_t>& GetViewShape() const { return view_shape_; }

  inline const std::vector<size_t>& GetEffectiveShape() const {
    return view_shape_.empty() ? shape_ : view_shape_;
  }

  inline size_t GetEffectiveElementCount() const {
    const auto& eff_shape = GetEffectiveShape();
    return std::accumulate(eff_shape.begin(), eff_shape.end(),
                           static_cast<size_t>(1),
                           std::multiplies<size_t>());
  }

  void Transpose(size_t dim1, size_t dim2, bool use_view_shape = false);

  size_t GetDimensionStride(size_t dim, bool use_view_shape = false) const;

  inline PoolType GetPoolType() const { return pool_type_; }

  inline size_t GetGPURequestId() const { return gpu_request_id_; }

  // Releases all memory; it is allocated again on the next access.
  void Resize(const std::vector<size_t>& new_shape);

  void ReleaseGPUMemory();

  // Call only after all device work on the tensor has finished.
  void InvalidatePlannedGPUStorage();

  // Frees the CPU copy only while the GPU copy is current.
  void ReleaseCPUMemory();

  // Shares the GPU storage of `root` from `element_offset`. Call it before
  // planning and before either tensor touches the GPU.
  void BindGPUSubview(Tensor<T>* root, size_t element_offset);

  inline bool IsGPUSubview() const { return subview_root_ != nullptr; }
  inline const Tensor<T>* GetGPUSubviewRoot() const { return subview_root_; }
  inline size_t GetGPUSubviewElementOffset() const {
    return subview_element_offset_;
  }
  inline bool HasGPUSubviews() const { return has_gpu_subviews_; }

private:
  std::vector<size_t> shape_;
  std::vector<size_t> view_shape_;
  size_t total_count_;
  size_t total_bytes_;
  Device head_;

  T* cpu_data_ptr_;

  T* gpu_data_ptr_;

  bool cpu_only_;
  
private:
  bool is_borrowed_ = false;

  Tensor<T>* subview_root_ = nullptr;
  size_t subview_element_offset_ = 0;
  bool has_gpu_subviews_ = false;
  std::vector<Tensor<T>*> gpu_subviews_;

  PoolType pool_type_ = PoolType::STATIC;
  size_t gpu_request_id_ = static_cast<size_t>(-1);
  std::string model_scope_id_;

  DataFormat data_format_ = DataFormat::FP32;
  half* gpu_data_fp16_ptr_ = nullptr;
  __nv_bfloat16* gpu_data_bf16_ptr_ = nullptr;
  size_t gpu_alloc_bytes_ = 0;

  bool cpu_cache_valid_ = false;

  void AllocateCPUMemory();

  void AllocateGPUMemory();

  void EnsureCPUDataReady(bool keep_head_gpu = false, bool read_only = false);

  void EnsureGPUDataReady();

  size_t FlattenIndex(const size_t* indices) const;

  bool IncrementIndices(size_t* indices, const std::vector<size_t>& shape) const;
};

// Defined in oris_ai/tensor/fp16/tensor_fp16.h.
template <>
class Tensor<half>;

// Bytes copied host to device by all tensors in this process.
uint64_t TensorHostToDeviceBytes();

} // namespace oris_ai
