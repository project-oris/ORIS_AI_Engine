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

// Capture-time hook mixin for parametric kernel node recording.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace oris_ai {

// Sentinel override for the three parametric-kernel wrappers.
struct CaptureSentinelState {
  bool active = false;
  int32_t flash_attn_kv_len = 0x5C01A000;
  int32_t rope_cache_position = 0x5C01B000;
  int32_t mmvq_kv_pos = 0x5C01C000;
};

extern thread_local CaptureSentinelState t_capture_sentinel;

// Sentinel substitution for the duration of a scope.
class CaptureSentinelGuard {
public:
  CaptureSentinelGuard() { t_capture_sentinel.active = true; }
  ~CaptureSentinelGuard() { t_capture_sentinel.active = false; }
  CaptureSentinelGuard(const CaptureSentinelGuard&) = delete;
  CaptureSentinelGuard& operator=(const CaptureSentinelGuard&) = delete;
};

// Records the graph nodes of a layer's kernel launches during capture.
class CaptureAwareKernelLayer {
public:
  // EndClaim's argument when no scalar is substituted; decode graphs reject it.
  static constexpr size_t kNoSentinelArg = SIZE_MAX;

  cudaGraphNode_t captured_node() const { return my_captured_node_; }
  bool has_captured_node() const { return my_captured_node_ != nullptr; }
  // Zero-based index of the primary node's substituted kernel argument.
  size_t captured_sentinel_arg() const { return my_sentinel_arg_; }
  // Second node; nullptr for a single-node claim.
  cudaGraphNode_t captured_node2() const { return my_captured_node2_; }
  bool has_captured_node2() const { return my_captured_node2_ != nullptr; }
  void ResetCapturedNode()  {
    my_captured_node_ = nullptr;
    my_sentinel_arg_ = kNoSentinelArg;
  }
  void ResetCapturedNode2() { my_captured_node2_ = nullptr; }
  void ResetCapturedNodes() {
    ResetCapturedNode();
    ResetCapturedNode2();
  }

protected:
  // Safe to call outside capture.
  void BeginClaim(cudaStream_t stream);

  void EndClaim(cudaStream_t stream, size_t sentinel_arg_index);

  // Claim of a second kernel launch.
  void BeginClaim2(cudaStream_t stream);
  void EndClaim2(cudaStream_t stream);

private:
  cudaGraphNode_t my_captured_node_  = nullptr;
  cudaGraphNode_t my_captured_node2_ = nullptr;
  size_t my_sentinel_arg_ = kNoSentinelArg;
  std::vector<cudaGraphNode_t> pre_deps_;
  std::vector<cudaGraphNode_t> pre_deps2_;
  bool claim_active_  = false;
  bool claim_active2_ = false;
};

}  // namespace oris_ai
