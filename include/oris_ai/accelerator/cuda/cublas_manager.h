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

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdint>
#include <mutex>

#define CUBLAS_CHECK(condition)                                       \
  do {                                                                \
    cublasStatus_t status = condition;                                \
    if (status != CUBLAS_STATUS_SUCCESS) {                            \
      fprintf(stderr, "CUBLAS Error: %s\n",                           \
              oris_ai::cublasGetErrorString(status));                 \
      abort();                                                        \
    }                                                                 \
  } while (0)

namespace oris_ai {

const char* cublasGetErrorString(cublasStatus_t error);

// Process-wide cublasLt handle and shared workspace. Not locked: one thread
// issues all GPU work of a process.
class CUBLASLtManager {
public:
  CUBLASLtManager(const CUBLASLtManager&) = delete;
  CUBLASLtManager& operator=(const CUBLASLtManager&) = delete;

  static CUBLASLtManager& GetInstance() {
    static CUBLASLtManager instance;
    return instance;
  }

  inline cublasLtHandle_t GetHandle() const {
    return active_handle_override_ ? active_handle_override_ : handle_;
  }

  inline cublasLtHandle_t GetDefaultHandle() const { return handle_; }

  // Makes GetHandle() return `h` on the calling thread; nullptr clears it.
  void SetActiveHandle(cublasLtHandle_t h);

  cublasLtHandle_t GetActiveHandleOverride() const {
    return active_handle_override_;
  }

  // The caller destroys the handle with DestroyAuxHandle().
  static cublasLtHandle_t CreateAuxHandle();

  static void DestroyAuxHandle(cublasLtHandle_t h);

  // nullptr for 0 bytes or on failure. A larger request reallocates the buffer
  // and advances GetSharedWorkspaceGeneration().
  void* GetSharedWorkspace(size_t required_size);

  inline size_t GetSharedWorkspaceSize() const { return workspace_size_; }

  // A CUDA graph that reads the shared workspace must be recaptured when this
  // value changes.
  inline uint64_t GetSharedWorkspaceGeneration() const {
    return workspace_generation_;
  }

private:
  CUBLASLtManager();
  ~CUBLASLtManager();

  cublasLtHandle_t handle_;
  static thread_local cublasLtHandle_t active_handle_override_;
  void* workspace_ = nullptr;
  size_t workspace_size_ = 0;
  uint64_t workspace_generation_ = 0;
};

// Process-wide classic cuBLAS handle.
class CUBLASManager {
public:
  CUBLASManager(const CUBLASManager&) = delete;
  CUBLASManager& operator=(const CUBLASManager&) = delete;

  static CUBLASManager& GetInstance() {
    static CUBLASManager instance;
    return instance;
  }

  inline cublasHandle_t GetHandle() const { return handle_; }

  inline cublasHandle_t GetDefaultHandle() const { return handle_; }

  // Use instead of cublasSetStream(); each stream gets its own workspace.
  void SetStreamWithWorkspace(cublasHandle_t handle, cudaStream_t stream);

  // Frees the workspace of `s`; call it before cudaStreamDestroy(s).
  static void DropStream(cudaStream_t s);

private:
  CUBLASManager();
  ~CUBLASManager();

  cublasHandle_t handle_;
};

} // namespace oris_ai

