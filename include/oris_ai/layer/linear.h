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

#include "oris_ai/layer/layer.h"
#include "oris_ai/layer/glu_gate_kind.h"
#include "oris_ai/gguf_reader/gguf_types.h"
#include "oris_ai/tensor/q8_0/tensor_q8_0.h"
#include "oris_ai/tensor/fp16/tensor_fp16.h"

#include <glog/logging.h>

#include <string>
#include <vector>

namespace oris_ai {

// Batches up to this size use MMVQ, larger ones MMQ.
// Reference: llama.cpp MMVQ_MAX_BATCH_SIZE.
static constexpr size_t MMVQ_MAX_BATCH_SIZE = 8;

// What InitLinearFromRaw does with an F16 weight: keep it or widen it to FP32.
enum class LinearF16Mode {
  Native,
  DequantFallback,
};

// Row count range [m_lo, m_hi] on one prefill FP16 backend.
struct PrefillF16Range {
  size_t m_lo = 0;
  size_t m_hi = 0;
  bool main_loop_unpack = false;
};

// `ranges` is ascending and disjoint; nullptr when none holds `rows`.
inline const PrefillF16Range* FindPrefillF16Range(
    const std::vector<PrefillF16Range>& ranges, size_t rows) {
  for (const PrefillF16Range& range : ranges) {
    if (rows < range.m_lo) return nullptr;
    if (rows <= range.m_hi) return &range;
  }
  return nullptr;
}

// Vision precision of a prefill plan; kNoPlan when none applies.
enum class PlanVisionPrecision { kNoPlan, kMmq, kW8A16 };

// Prefill FP16 GEMM selection resolved once at model construction.
struct PrefillF16Config {
  std::vector<PrefillF16Range> ranges;
  size_t weight_budget_elems = 0;
  bool overlap = true;
  bool plan_pending = false;
  bool fallback_model_default = false;
  unsigned long long fallback_min_m = 0;
  PlanVisionPrecision plan_vision = PlanVisionPrecision::kNoPlan;
  std::string plan_id;

  // One backend from min_m rows up; min_m == 0 disables it.
  static PrefillF16Config FromMinM(size_t min_m, size_t weight_budget_elems,
                                   bool overlap, bool main_loop_unpack) {
    PrefillF16Config config;
    config.weight_budget_elems = weight_budget_elems;
    config.overlap = overlap;
    if (min_m != 0) {
      config.ranges.push_back({min_m, static_cast<size_t>(-1), main_loop_unpack});
    }
    return config;
  }
  bool Enabled() const { return !ranges.empty(); }
  // nullptr when those rows keep MMQ.
  const PrefillF16Range* Find(size_t rows) const {
    return FindPrefillF16Range(ranges, rows);
  }
  bool UsesCuBLASLt() const {
    for (const PrefillF16Range& range : ranges) {
      if (!range.main_loop_unpack) return true;
    }
    return false;
  }
};

// Weight rows unpacked at once, at most out_features.
inline size_t Q8W8A16TileRows(size_t out_features, size_t in_features) {
  constexpr size_t kBudgetHalves = size_t{8} << 20;
  size_t rows = (kBudgetHalves / in_features) / 128 * 128;
  if (rows < 128) rows = 128;
  return rows < out_features ? rows : out_features;
}

// Row limit of the fragment-layout vision GEMM.
constexpr size_t kVisionQ8FragMaxRows = 4096;
constexpr size_t kVisionQ8FragUnresolved = static_cast<size_t>(-1);

// Shared scratch the enabled projections of one model need.
struct PrefillF16Demand {
  size_t weight_elems = 0;
  size_t max_in_features = 0;
  size_t layers = 0;
  bool overlap = false;
  std::vector<const void*> projections;
  std::vector<const void*> eligible;
};

// Fully connected layer.
template <typename T>
class Linear : public HiddenLayerAbstract<T> {
public:
  Linear(const std::string& layer_name)
    : HiddenLayerAbstract<T>(layer_name),
      use_view_input_shape_(true),
      batch_size_(0), in_features_(0), out_features_(0) {}

  virtual ~Linear() = default;

  // Borrows `weight_data` and `bias_data` without a copy.
  // `max_M_hint` is the upper bound of the forward batch M.
  virtual void InitLinearFromRaw(
      size_t in_features, size_t out_features, bool use_bias,
      const void* weight_data, size_t weight_bytes,
      const void* bias_data, size_t bias_bytes,
      ggml_type dtype,
      size_t max_M_hint,
      PoolType output_pool_type = PoolType::STATIC,
      ActivationType act = ActivationType::NONE) = 0;

  virtual bool SupportsOutputRebuild() const { return false; }

  virtual void RebuildOutputForInput(size_t max_M_hint);

  // Call after GPU work drains and before a planned scope is detached.
  void InvalidatePlannedWeightStorage();

  // Restores the raw weight and bias after the scope is reattached.
  void RebindRawWeightSource(const void* weight_data, size_t weight_bytes,
                             const void* bias_data, size_t bias_bytes,
                             ggml_type source_dtype);

  virtual void OnPlannedWeightStorageInvalidated() {}

 protected:
  // Shape {rows, columns} of the Q8_0 weight tensor; padded layouts enlarge it.
  virtual std::vector<size_t> Q8_0StorageShape(size_t out_features,
                                               size_t in_features) const {
    return {out_features, in_features};
  }

 public:

  virtual void Forward() = 0;

  virtual void SetCpuFastPath(bool /*on*/) {}

  // Owned by this layer; the producer must stop using it before teardown.
  virtual void* GetMMVQProducerDestination() const { return nullptr; }

  // Forward* calls that return bool return whether that path was taken.
  virtual bool ForwardMMVQPrequantized(const void* pre_quantized_q8_1) {
    (void)pre_quantized_q8_1;
    return false;
  }

  // nullptr when the Forward with `rows` rows takes another route.
  virtual void* GetMMQProducerDestination(size_t rows, bool* use_ds4) const {
    (void)rows;
    (void)use_ds4;
    return nullptr;
  }

  // The same for a layer that clamps its input: the rows hold clamped values.
  virtual void* GetClampedMMQProducerDestination(size_t rows, float* clamp_lo,
                                                 float* clamp_hi) const {
    (void)rows;
    (void)clamp_lo;
    (void)clamp_hi;
    return nullptr;
  }

  // The producer already wrote the rows to the destination above.
  virtual bool ForwardMMQPrequantized(const void* pre_quantized_mmq) {
    (void)pre_quantized_mmq;
    return false;
  }

  // kv_stride = num_kv_heads * head_dim; kv_pos is the stored token count.
  virtual bool ForwardMMVQWithKVWriteF16(const void* pre_quantized_q8_1,
                                         void* kv_dst, size_t kv_stride,
                                         size_t kv_pos) {
    (void)pre_quantized_q8_1;
    (void)kv_dst;
    (void)kv_stride;
    (void)kv_pos;
    return false;
  }

  // kv_stride equals the output features.
  virtual bool ForwardPrefillWithKVWriteF16(void* kv_dst, size_t kv_stride,
                                            size_t kv_pos) {
    (void)kv_dst;
    (void)kv_stride;
    (void)kv_pos;
    return false;
  }

  virtual bool ForwardMMVQWithResidual(const float* residual,
                                       float* output_dst) {
    (void)residual;
    (void)output_dst;
    return false;
  }

  virtual bool ForwardMMVQWithResidualPrequantized(
      const void* pre_quantized_q8_1, const float* residual,
      float* output_dst) {
    (void)pre_quantized_q8_1;
    (void)residual;
    (void)output_dst;
    return false;
  }

  virtual bool ForwardMMQWithGlu(const float* gate_ptr, GluGateKind gate_kind) {
    (void)gate_ptr;
    (void)gate_kind;
    return false;
  }

  virtual void EnablePrefillF16Gemm(const PrefillF16Config& config,
                                    PrefillF16Demand* demand) {
    (void)config;
    (void)demand;
  }
  virtual bool PrefillF16Active(size_t rows) const {
    (void)rows;
    return false;
  }
  virtual bool ForwardPrefillF16WithGlu(const float* gate,
                                        GluGateKind kind) {
    (void)gate;
    (void)kind;
    return false;
  }
  // Stores act(gate) * up as FP16 rows at `rows` instead of the FP32 output.
  virtual bool ForwardPrefillF16GluEpilogue(const float* gate, GluGateKind kind,
                                            void* rows) {
    (void)gate;
    (void)kind;
    (void)rows;
    return false;
  }
  // `rows` holds M x in_features FP16 values.
  virtual bool ForwardPrefillF16OnRows(const void* rows) {
    (void)rows;
    return false;
  }
  // The next forward over exactly `rows` rows adds into `residual` instead of
  // storing its output.
  virtual void ArmResidualAdd(float* residual, const float* bias, size_t rows) {
    (void)residual;
    (void)bias;
    (void)rows;
  }
  virtual bool TakeResidualAdded() { return false; }

  // Sticky per-element input clamp.
  virtual void SetActivationClamp(float /*min*/, float /*max*/) {}

  // Sticky bias add and GELU transform of the input.
  virtual void SetActivationBiasGeluErf(Tensor<T>* /*bias*/) {}

  virtual void SetActivationBiasGeluTanh(Tensor<T>* /*bias*/) {}

  virtual bool ForwardMMQReusingQuantizedActivation() { return false; }

  // Runs multi-row Forwards of a Q8_0 projection on FP16 inputs and weights.
  virtual bool EnableQ8W8A16(size_t frag_max_rows = 0) {
    (void)frag_max_rows;
    return false;
  }

  // Stops unless the F16 weights are already on the GPU.
  void ReleaseBorrowedWeights() override {
    if (weight_) weight_->ClearBorrowedState();
    if (bias_) bias_->ClearBorrowedState();
    if (weight_q8_0_) weight_q8_0_->ClearBorrowedState();
    if (weight_f16_) {
      CHECK_EQ(static_cast<int>(weight_f16_->GetDevice()),
               static_cast<int>(Device::GPU))
          << "Linear (" << this->GetLayerName()
          << ") ReleaseBorrowedWeights called before F16 native weight "
          << "Stage A H2D — GPU side has no valid source after this point";
      weight_f16_->ClearBorrowedState();
    }
    if (!f16_dequant_buf_.empty()) {
      CHECK(weight_ != nullptr)
          << "Linear (" << this->GetLayerName()
          << ") F16 dequant buffer present but weight_ is null";
      CHECK_EQ(static_cast<int>(weight_->GetDevice()),
               static_cast<int>(Device::GPU))
          << "Linear (" << this->GetLayerName()
          << ") ReleaseBorrowedWeights called before F16 weight H2D — "
          << "the GPU side has no valid source after this point";
      std::vector<float>().swap(f16_dequant_buf_);
    }
  }

  bool AllWeightsReleased() const override {
    const bool dense_released =
           (weight_      == nullptr || !weight_->IsBorrowed()) &&
           (bias_        == nullptr || !bias_->IsBorrowed());
    return dense_released &&
           (weight_q8_0_ == nullptr || !weight_q8_0_->IsBorrowed());
  }

  inline void SetF16Mode(LinearF16Mode mode) { f16_mode_ = mode; }
  inline LinearF16Mode GetF16Mode() const { return f16_mode_; }

  inline Tensor<T>* GetWeight() { return weight_.get(); }

  inline Tensor<T>* GetBias() { return bias_ ? bias_.get() : nullptr; }

  // nullptr when the weight is not Q8_0.
  inline Tensor<block_q8_0>* GetWeightQ8_0() { return weight_q8_0_.get(); }

  inline Tensor<half>* GetWeightF16() { return weight_f16_.get(); }

  inline ggml_type GetWeightDtype() const { return weight_dtype_; }

  // True only after the source was copied into owned host storage.
  bool RawWeightCopiedAtInit() const { return raw_weight_copied_at_init_; }

  inline size_t GetInFeatures() const { return in_features_; }

  inline size_t GetOutFeatures() const { return out_features_; }

  inline size_t GetBatchSize() const { return batch_size_; }

  inline ActivationType GetActivationType() const { return activation_type_; }

  inline void SetUseViewInputShape(bool use_view) { use_view_input_shape_ = use_view; }

protected:
  void LinearSetupFromRaw(
      size_t in_features, size_t out_features, bool use_bias,
      const void* weight_data, size_t weight_bytes,
      const void* bias_data, size_t bias_bytes,
      ggml_type dtype,
      PoolType output_pool_type,
      ActivationType act = ActivationType::NONE);

  void SetupCNNToLinearFlatten();

  void HandleCNN4DInput(Tensor<T>* input_tensor, const std::vector<size_t>& cnn_shape);

  void Handle2DInput(const std::vector<size_t>& linear_shape);

  void Handle3DInput(const std::vector<size_t>& input_shape);

  void SetupLinearOutput(PoolType output_pool_type);

  bool use_view_input_shape_;
  size_t batch_size_;
  size_t in_features_;
  size_t out_features_;
  ActivationType activation_type_ = ActivationType::NONE;
  std::unique_ptr<Tensor<T>> weight_;
  std::unique_ptr<Tensor<T>> bias_;

  bool WeightsHostOnly() const {
    return false;
  }

  std::unique_ptr<Tensor<block_q8_0>> weight_q8_0_;

  std::vector<float> f16_dequant_buf_;

  std::unique_ptr<Tensor<half>> weight_f16_;

  // F16 only on the Native path.
  ggml_type weight_dtype_ = GGML_TYPE_F32;
  bool raw_weight_copied_at_init_ = false;
  ggml_type raw_source_dtype_ = GGML_TYPE_F32;
  bool raw_source_initialized_ = false;

  LinearF16Mode f16_mode_ = LinearF16Mode::Native;

  std::string VectorToString(const std::vector<size_t>& vec) const {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) result += ",";
      result += std::to_string(vec[i]);
    }
    result += "]";
    return result;
  }
};

} // namespace oris_ai
