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

#include <memory>
#include <vector>

#include "oris_ai/layer/layer.h"
#include "oris_ai/layer/activation.h"
#include "oris_ai/layer/linear.h"
#include "oris_ai/layer/elementwise.h"
#include "oris_ai/gguf_reader/llm_layer_info.h"
#include "oris_ai/layer/custom_language/rmsnorm.h"

namespace oris_ai {

template <typename T> class LinearQuant;

// FFN(x) = ffn_down(SiLU(ffn_gate(x)) * ffn_up(x)).
template <typename T>
class FFN : public HiddenLayerAbstract<T> {
public:
  // Only SwiGLU is compatible with `fuse_swiglu_quantize`.
  FFN(const std::string &layer_name, Device target_device, bool add,
      bool inplace_residual = false, bool fuse_swiglu_quantize = false,
      ActivationType gate_act = ActivationType::SwiGLU);

  ~FFN();

  void InitFFNFromRaw(
      const LLMLayerInfo& gate_info, const void* gate_data,
      const LLMLayerInfo& up_info, const void* up_data,
      const LLMLayerInfo& down_info, const void* down_data,
      size_t max_M_hint,
      PoolType linear_output_pool_type = PoolType::ACTIVATION,
      PoolType intermediate_pool_type = PoolType::ACTIVATION,
      PoolType elementwise_output_pool_type = PoolType::ACTIVATION);

  void ReleaseBorrowedWeights() override;

  std::vector<typename HiddenLayerAbstract<T>::SubLayerInfo>
      GetActivationTensorInfo() override;
  int32_t GetSubStepCount() const override {
    int32_t count = 5;
    if (fuse_swiglu_quantize_) count--;
    if (inplace_residual_) count--;
    return count;
  }
  std::vector<std::pair<size_t, int32_t>> GetInputLastUseHints() const override;

  inline Tensor<T>* GetOutputTensor() override {
    if (inplace_residual_) return this->GetInputTensor(1);
    if (add_) return elementwise_->GetOutputTensor();
    CHECK(ffn_down_ != nullptr) << "FFN::GetOutputTensor called before init";
    return ffn_down_->GetOutputTensor();
  }

  void EnablePrefillF16Gemm(const PrefillF16Config& config,
                            PrefillF16Demand* demand) {
    for (Linear<T>* layer : {ffn_gate_.get(), ffn_up_.get(), ffn_down_.get()})
      if (layer) layer->EnablePrefillF16Gemm(config, demand);
  }

  bool AllProjectionWeightsAre(ggml_type dtype) const {
    for (const std::unique_ptr<Linear<T>>* proj :
         {&ffn_gate_, &ffn_up_, &ffn_down_}) {
      if (*proj != nullptr && (*proj)->GetWeightDtype() != dtype) return false;
    }
    return true;
  }

  void Forward() override;

  // `norm` is not owned.
  void SetFFNNorm(RMSNorm<T>* norm);

  // `norm` is not owned.
  void SetNextAttnNormForFusedAdd(RMSNorm<T>* norm) {
    next_attn_norm_for_fused_add_ = norm;
  }
  void EnableResidualEpilogue() { residual_epilogue_ = true; }

  // Batches of 2 to 8 tokens.
  void SetMultiRowResidualSplit(bool enabled) {
    multi_row_residual_split_ = enabled;
  }

  // Call before InitFFNFromRaw; pass the scratch to SetF16Scratch after it.
  void SetF16NativePreference(bool enabled) {
    dense_f16_native_pref_ = enabled;
  }

  // Call after InitFFNFromRaw.
  void SetF16Scratch(Tensor<half>* input_scratch,
                     Tensor<half>* output_scratch);

private:
  bool dense_f16_native_pref_ = false;

  std::unique_ptr<Linear<T>> ffn_gate_;
  std::unique_ptr<Linear<T>> ffn_up_;
  std::unique_ptr<Linear<T>> ffn_down_;
  std::unique_ptr<Activation<T>> swiglu_;
  std::unique_ptr<ElementWise<T>> elementwise_;

  Device target_device_;

  Tensor<T>* gate_output_;
  Tensor<T>* up_output_;

  bool add_;
  bool inplace_residual_;
  bool fuse_swiglu_quantize_;
  ActivationType gate_act_;
  bool glu_mmq_fusion_ = false;
  bool glu_f16_epilogue_ = false;

  // Returns true if taken.
  bool ForwardFusedGateUpMMVQ();

  void* fused_q8_1_buf_ = nullptr;
  size_t fused_q8_1_buf_size_ = 0;

  const void* down_prequantized_ = nullptr;

  LinearQuant<T>* q8_gate_ = nullptr;
  LinearQuant<T>* q8_up_ = nullptr;

  RMSNorm<T>* ffn_norm_ptr_ = nullptr;

  RMSNorm<T>* next_attn_norm_for_fused_add_ = nullptr;
  bool residual_epilogue_ = false;
  bool multi_row_residual_split_ = false;
  bool decode_residual_split_ = false;
};

} // namespace oris_ai
