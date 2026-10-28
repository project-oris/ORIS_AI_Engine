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

#include <string>

#include "oris_ai/tensor/tensor_memory_planner.h"

#include "oris_ai/accelerator/cuda/cuda_stream_manager.h"
#include <cuda_runtime.h>
#include "oris_ai/accelerator/cuda/cublas_manager.h"

namespace oris_ai {

// Sets the planner's active model and, when non-null, the thread's CUDA
// stream and cuBLASLt handle; restores the previous values when destroyed.
class ActiveScopeGuard {
 public:
  using AuxStreamType = cudaStream_t;

  using AuxHandleType = cublasLtHandle_t;

  explicit ActiveScopeGuard(const std::string& new_scope,
                            AuxStreamType new_stream = nullptr,
                            AuxHandleType new_handle = nullptr) {
    auto& planner = TensorMemoryPlanner::Instance();
    saved_scope_ = planner.GetActiveModel();
    if (new_scope != saved_scope_) {
      planner.SetActiveModel(new_scope);
      restore_scope_ = true;
    }
    if (new_stream != nullptr) {
      auto& smgr = CUDAStreamManager::GetInstance();
      saved_stream_override_ = smgr.GetActiveStreamOverride();
      smgr.SetActiveStream(new_stream);
      restore_stream_ = true;
    }
    if (new_handle != nullptr) {
      auto& hmgr = CUBLASLtManager::GetInstance();
      saved_handle_override_ = hmgr.GetActiveHandleOverride();
      hmgr.SetActiveHandle(new_handle);
      restore_handle_ = true;
    }
  }

  ~ActiveScopeGuard() {
    if (restore_handle_) {
      CUBLASLtManager::GetInstance().SetActiveHandle(saved_handle_override_);
    }
    if (restore_stream_) {
      CUDAStreamManager::GetInstance().SetActiveStream(saved_stream_override_);
    }
    if (restore_scope_) {
      TensorMemoryPlanner::Instance().SetActiveModel(saved_scope_);
    }
  }

  ActiveScopeGuard(const ActiveScopeGuard&) = delete;
  ActiveScopeGuard& operator=(const ActiveScopeGuard&) = delete;

 private:
  std::string saved_scope_;
  bool restore_scope_ = false;
  cudaStream_t saved_stream_override_ = nullptr;
  bool restore_stream_ = false;
  cublasLtHandle_t saved_handle_override_ = nullptr;
  bool restore_handle_ = false;
};

}  // namespace oris_ai
