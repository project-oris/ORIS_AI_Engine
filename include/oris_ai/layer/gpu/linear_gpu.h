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

#include "oris_ai/layer/custom_language/capture_aware_kernel_layer.h"
#include "oris_ai/layer/linear.h"

#include <memory>

namespace oris_ai {

// GPU base of Linear.
template <typename T>
class LinearGPU : public Linear<T>, public CaptureAwareKernelLayer {
public:
  LinearGPU(const std::string& layer_name);

  virtual ~LinearGPU();

  // Weights borrowed from an external tensor, not owned.
  virtual void InitFromBorrowedQ8_0WeightTensor(
      const Tensor<block_q8_0>* borrowed,
      size_t in_features, size_t out_features, size_t max_M_hint,
      PoolType output_pool_type = PoolType::ACTIVATION);

  virtual void InitFromBorrowedFP32WeightTensor(
      const Tensor<T>* borrowed,
      size_t in_features, size_t out_features, size_t max_M_hint,
      PoolType output_pool_type = PoolType::ACTIVATION);

  // Pass the same `owner` to Release; null retains count-only ownership.
  // Callers serialize reservation, trimming and execution of shared scratch.
  static void PreallocateSharedBuffers(size_t max_M, size_t max_K,
                                       size_t max_N, int nsm,
                                       const void* owner = nullptr);
  // Extends a named owner's lease with the prefill FP16 GEMM scratch.
  static void ReservePrefillF16Buffers(const void* owner, size_t max_M,
                                       const PrefillF16Demand& demand);
  static bool Q8W8A16FragEligible(size_t out_features, size_t in_features);
  static void ReleaseSharedBuffers(const void* owner = nullptr);
  // Call outside Forward and capture, after unleased users have finished.
  static bool TrimSharedActivationBuffer();
  // Grow-only; SharedResourceEpoch identifies the addresses.
  static void ReserveSharedBuffers(size_t M, size_t K, size_t N, int nsm,
                                   size_t padded_K);
  static uint64_t SharedResourceEpoch();

  // GPU buffers of LinearGPU that live outside TensorMemoryPlanner.
  struct ExternalBufferStats {
    size_t shared_mmq_activation_size = 0;
    size_t shared_fixup_size          = 0;
    size_t shared_padded_fp32_size    = 0;
    size_t shared_mmvq_q8_1_size      = 0;
    size_t total_q8_1_buf_size        = 0;
    size_t total_q8_1_buf_count       = 0;
    size_t shared_f16_weight_size     = 0;
  };
  static ExternalBufferStats GetExternalBufferStats();

  bool IsBorrowedQuantWeight() const {
    return borrowed_weight_q8_0_ != nullptr;
  }

  bool IsBorrowedFP32Weight() const {
    return borrowed_weight_fp32_ != nullptr;
  }

  void ReleaseBorrowedWeights() override;

  virtual void SetF16Scratch(Tensor<half>* input_scratch,
                             Tensor<half>* output_scratch);

  virtual void SetF16OutputBiasEpilogue(Tensor<T>* /*bias*/) {}
  virtual bool HasF16OutputBiasEpilogue() const { return false; }

  // Writes GELU_tanh(x W^T + bias) as F16 rows into the output storage.
  virtual void SetF16GeluBiasOutputEpilogue(Tensor<T>* /*bias*/,
                                            size_t /*min_rows*/) {}
  virtual bool HasF16GeluBiasOutput() const { return false; }
  virtual const half* F16GeluBiasOutputRows() const {
    LOG(FATAL) << "F16GeluBiasOutputRows is not supported on ("
               << this->GetLayerName() << ")";
    return nullptr;
  }

  // Runs on F16 rows another layer wrote instead of casting the input.
  virtual void ForwardF16OnHalfInput(const half* /*input_rows*/) {
    LOG(FATAL) << "ForwardF16OnHalfInput is not supported on ("
               << this->GetLayerName() << ")";
  }
  virtual bool CanForwardF16OnHalfInput() const { return false; }

  // Fused QKV projection storing half2 attention input rows in its output.
  virtual void SetQkvMRoPEHalfOutput(Tensor<T>* /*bias*/, int /*head_dim*/) {}
  virtual void SetQkvMRoPEPositions(const int32_t* /*positions*/, int /*sec0*/,
                                    float /*theta_scale*/) {}
  virtual bool HasQkvMRoPEHalfOutput() const { return false; }
  virtual const half* QkvMRoPEHalfRows() const {
    LOG(FATAL) << "QkvMRoPEHalfRows is not supported on ("
               << this->GetLayerName() << ")";
    return nullptr;
  }

  virtual bool CanShareF16InputCastOf(const LinearGPU<T>& /*producer*/) const {
    return false;
  }

  // Reuses the input cast of `producer`'s immediately preceding Forward.
  virtual void ForwardF16ReusingInputCastOf(const LinearGPU<T>& /*producer*/) {
    LOG(FATAL) << "ForwardF16ReusingInputCastOf is only supported on "
               << "LinearDense (" << this->GetLayerName() << ")";
  }

  virtual bool CanConsumePrefilledF16Input() const { return false; }

  virtual bool TakesPrefilledF16Input(size_t rows) const {
    (void)rows;
    return CanConsumePrefilledF16Input();
  }

  // A producer stores D4 block_q8_1_mmq rows, zero blocks past in_features.
  virtual bool TakesPrefilledQ8_1Input(size_t rows) const {
    (void)rows;
    return false;
  }
  virtual void* PrefilledQ8_1Input(size_t rows) {
    (void)rows;
    LOG(FATAL) << "PrefilledQ8_1Input is not supported on ("
               << this->GetLayerName() << ")";
    return nullptr;
  }
  virtual size_t PrefilledQ8_1Columns() const { return 0; }
  virtual void ForwardMMQOnPrefilledQ8_1Input() {
    LOG(FATAL) << "ForwardMMQOnPrefilledQ8_1Input is not supported on ("
               << this->GetLayerName() << ")";
  }

  // Holds at least `rows` x in_features values.
  virtual half* PrefilledF16Input(size_t rows) {
    (void)rows;
    LOG(FATAL) << "PrefilledF16Input is not supported on ("
               << this->GetLayerName() << ")";
    return nullptr;
  }

  virtual void ForwardF16OnPrefilledInput() {
    LOG(FATAL) << "ForwardF16OnPrefilledInput is not supported on ("
               << this->GetLayerName() << ")";
  }
  // Call before InitLinearFromRaw, on a layer that never runs as W8A16.
  virtual void AllowPaddedQ8_0RowChunks() {}

protected:
  const Tensor<block_q8_0>* borrowed_weight_q8_0_ = nullptr;

  const Tensor<T>* borrowed_weight_fp32_ = nullptr;

  inline const block_q8_0* ResolveQ8_0WeightGPUPtr() const {
    return borrowed_weight_q8_0_ != nullptr
        ? borrowed_weight_q8_0_->GetConstGPUDataPtr()
        : this->weight_q8_0_->GetConstGPUDataPtr();
  }
  inline const T* ResolveFP32WeightGPUPtr() const {
    return borrowed_weight_fp32_ != nullptr
        ? borrowed_weight_fp32_->GetConstGPUDataPtr()
        : this->weight_->GetConstGPUDataPtr();
  }

  bool use_cublaslt_dispatch_ = false;

  int nsm_ = 0;
};

// Scoped owner lease of the process-shared quantized Linear scratch.
template <typename T>
class LinearSharedBufferLease {
public:
  LinearSharedBufferLease(size_t max_M, size_t max_K, size_t max_N, int nsm) {
    LinearGPU<T>::PreallocateSharedBuffers(max_M, max_K, max_N, nsm, this);
  }

  ~LinearSharedBufferLease() { LinearGPU<T>::ReleaseSharedBuffers(this); }

  LinearSharedBufferLease(const LinearSharedBufferLease&) = delete;
  LinearSharedBufferLease& operator=(const LinearSharedBufferLease&) = delete;
};

} // namespace oris_ai
