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

#include "oris_ai/accelerator/active_scope_guard.h"
#include "oris_ai/accelerator/cuda/cuda_define.h"
#include "oris_ai/accelerator/cuda/functions/cuda_functions_llm.h"
#include "oris_ai/accelerator/cuda/unique_aux_handle.h"
#include "oris_ai/accelerator/cuda/unique_aux_stream.h"
#include "oris_ai/gguf_reader/gguf_open_util.h"
#include "oris_ai/gguf_reader/gguf_reader.h"
#include "oris_ai/model/language/llama/tokenizer.h"
#include "oris_ai/model/language/qwen3/qwen3.h"
#include "oris_ai/layer/gpu/linear_gpu.h"
#include "oris_ai/model/vision/vision_q8_w8a16.h"
#include "oris_ai/tensor/tensor_memory_planner.h"

#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace oris_ai {

// Per-instance planner scope ids of a VLM container; not copyable.
class VlmScopePair {
 public:
  explicit VlmScopePair(const std::string& prefix) {
    static std::atomic<uint64_t> counter{0};
    const uint64_t id = counter.fetch_add(1);
    llm_ = prefix + "_llm_" + std::to_string(id);
    vit_ = prefix + "_vit_" + std::to_string(id);
    persistent_ = prefix + "_persistent_" + std::to_string(id);
  }
  VlmScopePair(const VlmScopePair&) = delete;
  VlmScopePair& operator=(const VlmScopePair&) = delete;
  VlmScopePair(VlmScopePair&&) = delete;
  VlmScopePair& operator=(VlmScopePair&&) = delete;

  const std::string& llm() const { return llm_; }
  const std::string& vit() const { return vit_; }
  const std::string& persistent() const { return persistent_; }

 private:
  std::string llm_;
  std::string vit_;
  std::string persistent_;
};

inline TensorMemoryPlanner::ScopeOptions VisionTowerScopeOptions() {
  TensorMemoryPlanner::ScopeOptions options;
  options.skip_overwritten_static_zero = true;
  return options;
}

constexpr bool kVlmSharedActivationsByDefault = true;

inline bool VlmSharedActivationsEnabledByEnv() {
  const char* value = std::getenv("ORIS_VLM_SHARED_ACTIVATIONS");
  if (value == nullptr || value[0] == '\0') return kVlmSharedActivationsByDefault;
  return value[0] == '1' && value[1] == '\0';
}

constexpr bool kVlmSharedActivationsSplitByDefault = true;

inline bool VlmSharedActivationsSplitEnabledByEnv() {
  const char* value = std::getenv("ORIS_VLM_SHARED_ACTIVATIONS_SPLIT");
  if (value == nullptr || value[0] == '\0') {
    return kVlmSharedActivationsSplitByDefault;
  }
  return value[0] == '1' && value[1] == '\0';
}

// A byte (hexadecimal), or -1.
inline int VlmSharedActivationsPoisonByte() {
  const char* value = std::getenv("ORIS_VLM_SHARED_ACTIVATIONS_POISON");
  if (value == nullptr || value[0] == '\0') return -1;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 16);
  if (end == value || *end != '\0' || parsed > 0xFFul) {
    LOG(WARNING) << "ORIS_VLM_SHARED_ACTIVATIONS_POISON='" << value
                 << "' is not a hexadecimal byte; ignored";
    return -1;
  }
  return static_cast<int>(parsed);
}

// Handoff of the decoder's activation range to its vision tower.
class VlmSharedActivations {
 public:
  VlmSharedActivations() = default;
  VlmSharedActivations(const VlmSharedActivations&) = delete;
  VlmSharedActivations& operator=(const VlmSharedActivations&) = delete;
  ~VlmSharedActivations() {
    if (event_ != nullptr) cudaEventDestroy(event_);
  }

  bool enabled() const { return enabled_; }

  bool Enable(const VlmScopePair& scopes, const char* tag) {
    if (enabled_) return true;
    if (!VlmSharedActivationsEnabledByEnv()) return false;
    const bool split = VlmSharedActivationsSplitEnabledByEnv();
    TensorMemoryPlanner::Instance().LendActivationStorage(scopes.llm(),
                                                          scopes.vit(), split);
    enabled_ = true;
    last_user_ = User::kLlm;
    LOG(INFO) << "TMP_ACTIVATION_SHARE_ENABLED container=" << tag
              << " owner=" << scopes.llm() << " borrower=" << scopes.vit()
              << " poison=" << VlmSharedActivationsPoisonByte()
              << " split=" << split;
    return true;
  }

  // Call before the first write of an encode into the borrowed range.
  void BeginVisionEncode(const VlmScopePair& scopes, cudaStream_t main,
                         cudaStream_t aux) {
    if (!enabled_) return;
    if (last_user_ != User::kVision) {
      Order(main, aux);
      last_user_ = User::kVision;
    }
    Poison(scopes.vit(), aux, /*whole_decoder_range=*/false);
  }

  // Call before the decoder's first work after an encode.
  void BeginLlmPrefill(const VlmScopePair& scopes, cudaStream_t main,
                       cudaStream_t aux) {
    if (!enabled_) return;
    if (last_user_ == User::kVision) {
      Order(aux, main);
      last_user_ = User::kLlm;
    }
    Poison(scopes.vit(), main, /*whole_decoder_range=*/true);
  }

 private:
  enum class User : uint8_t { kLlm, kVision };

  void Order(cudaStream_t before, cudaStream_t after) {
    if (event_ == nullptr) {
      CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(event_, before));
    CUDA_CHECK(cudaStreamWaitEvent(after, event_, 0));
  }

  void Poison(const std::string& vit_scope, cudaStream_t stream,
              bool whole_decoder_range) {
    const int byte = VlmSharedActivationsPoisonByte();
    if (byte < 0) return;
    auto& planner = TensorMemoryPlanner::Instance();
    void* base = nullptr;
    size_t bytes = 0;
    if (!planner.GetActivationBorrowRange(vit_scope, &base, &bytes)) return;
    if (whole_decoder_range) {
      bytes = std::max(bytes, planner.GetActivationShare(vit_scope).shareable_bytes);
    }
    CUDA_CHECK(cudaMemsetAsync(base, byte, bytes, stream));
    LOG(INFO) << "TMP_ACTIVATION_SHARE_POISON side="
              << (whole_decoder_range ? "decoder" : "vision")
              << " byte=" << byte << " bytes=" << bytes;
    void* own_base = nullptr;
    size_t own_bytes = 0;
    if (!whole_decoder_range &&
        planner.GetActivationOwnLivenessRange(vit_scope, &own_base, &own_bytes)) {
      CUDA_CHECK(cudaMemsetAsync(own_base, byte, own_bytes, stream));
      LOG(INFO) << "TMP_ACTIVATION_SHARE_POISON side=vision_own byte=" << byte
                << " bytes=" << own_bytes;
    }
  }

  bool enabled_ = false;
  User last_user_ = User::kLlm;
  cudaEvent_t event_ = nullptr;
};

// Idempotent.
inline void EnsureVlmAuxChannels(UniqueAuxStream& aux_stream,
                                 UniqueAuxHandle& aux_handle) {
  if (!aux_stream.valid()) {
    aux_stream = UniqueAuxStream::Create();
  }
  if (!aux_handle.valid()) {
    aux_handle = UniqueAuxHandle::Create();
  }
}

// `configure` runs before the decoder opens.
template <typename LlmT, typename ConfigureFn, typename OpenFn>
std::unique_ptr<LlmT> OpenVlmDecoderInScope(const VlmScopePair& scopes,
                                            Device target_device,
                                            size_t max_prefill_seq_len,
                                            size_t max_seq_len,
                                            const std::string& llm_path,
                                            const char* tag,
                                            ConfigureFn&& configure,
                                            OpenFn&& open) {
  TensorMemoryPlanner::Instance().CreateModelScope(scopes.llm());
  ActiveScopeGuard g(scopes.llm());
  auto llm = std::make_unique<LlmT>(target_device);
  llm->SetModelScope(scopes.llm());
  llm->SetMaxPrefillSeqLen(max_prefill_seq_len);
  llm->SetMaxSeqLen(max_seq_len);
  configure(*llm);
  llm->CreateInputTensor(/*batch_size=*/1, max_prefill_seq_len);
  if (!open(*llm)) {
    LOG(ERROR) << tag << "llm Open(" << llm_path << ") failed";
    return nullptr;
  }
  return llm;
}

template <typename T>
std::unique_ptr<Qwen3<T>> OpenVlmLlmInScope(
    const VlmScopePair& scopes,
    Device target_device,
    size_t max_prefill_seq_len,
    size_t max_seq_len,
    const std::string& llm_path,
    std::unique_ptr<GGUFReader> reader,
    std::unique_ptr<Qwen3VLParser> parser,
    const char* tag,
    size_t max_logits_rows = 1,
    size_t kv_initial_tokens = 0) {
  return OpenVlmDecoderInScope<Qwen3<T>>(
      scopes, target_device, max_prefill_seq_len, max_seq_len, llm_path, tag,
      [&](Qwen3<T>& llm) {
        llm.SetMaxLogitsRows(max_logits_rows);
        llm.SetKvInitialTokens(kv_initial_tokens);
      },
      [&](Qwen3<T>& llm) {
        return llm.Open(llm_path, std::move(reader), std::move(parser));
      });
}

template <typename T>
std::unique_ptr<Qwen3<T>> OpenVlmLlmInScope(const VlmScopePair& scopes,
                                            Device target_device,
                                            size_t max_prefill_seq_len,
                                            size_t max_seq_len,
                                            const std::string& llm_path,
                                            const char* tag,
                                            size_t max_logits_rows = 1) {
  return OpenVlmLlmInScope<T>(scopes, target_device, max_prefill_seq_len,
                              max_seq_len, llm_path, /*reader=*/nullptr,
                              /*parser=*/nullptr, tag, max_logits_rows);
}

// Needs the decoder's GGUF header still alive.
template <typename LlmT>
std::unique_ptr<Tokenizer> BuildTokenizerFromLiveLlmHeader(
    const LlmT& llm,
    const char* tokenizer_name,
    const std::string& llm_path,
    const char* tag) {
  const GGUFReader* llm_metadata = llm.MetadataReaderIfAlive();
  auto tokenizer = std::make_unique<Tokenizer>(tokenizer_name);
  if (llm_metadata == nullptr ||
      !tokenizer->InitFromMetadata(*llm_metadata)) {
    LOG(ERROR) << tag << "tokenizer InitFromMetadata failed for " << llm_path;
    return nullptr;
  }
  return tokenizer;
}

template <typename ParserT>
bool OpenMmprojReaderAndParser(const std::string& mmproj_path,
                               const char* tag,
                               const char* parser_type_name,
                               std::unique_ptr<GGUFReader>* reader_out,
                               std::unique_ptr<ParserT>* parser_out) {
  return ReadGgufHeader(mmproj_path, tag, reader_out) &&
         MapGgufDataSection(reader_out->get(), mmproj_path, tag) &&
         ParseGgufHeaderWith<ParserT>(reader_out->get(), &ParserT::ParseVision,
                                      tag, parser_type_name, "ParseVision",
                                      parser_out);
}

inline bool CheckMmprojProjectionDim(size_t projection_dim,
                                     size_t llm_embedding_dim,
                                     const char* tag) {
  if (projection_dim == llm_embedding_dim) return true;
  LOG(ERROR) << tag << "mmproj projection_dim=" << projection_dim
             << " mismatches LLM embedding_dim=" << llm_embedding_dim
             << " (mmproj/LLM size mismatch)";
  return false;
}

inline bool CheckDecoderHasNoDeepstack(size_t n_deepstack_layers,
                                       const char* tag) {
  if (n_deepstack_layers == 0) return true;
  LOG(ERROR) << tag << "decoder has " << n_deepstack_layers
             << " DeepStack layers, but this container has no injection path";
  return false;
}

// A step that throws is logged and dropped.
template <typename Fn>
void BestEffortVisionCleanup(const char* step_name, Fn&& step) noexcept {
  try {
    step();
  } catch (const std::exception& e) {
    LOG(ERROR) << "VLM_CLEANUP_ERROR step=" << step_name << " what=" << e.what();
  } catch (...) {
    LOG(ERROR) << "VLM_CLEANUP_ERROR step=" << step_name << " what=unknown";
  }
}

// When a kept vision tower frees its activation pool.
struct VisionActivationReleaseMode {
  bool at_stage = false;
  bool at_idle = false;
  size_t idle_ms = 0;
};

inline VisionActivationReleaseMode VisionActivationReleaseModeFromEnv() {
  VisionActivationReleaseMode mode;
  const char* value = std::getenv("ORIS_VISION_TOWER_RELEASE_ACTIVATIONS");
  if (value == nullptr) return mode;
  if (std::strcmp(value, "1") == 0) {
    mode.at_stage = true;
    return mode;
  }
  constexpr char kIdle[] = "idle:";
  constexpr size_t kIdleLen = sizeof(kIdle) - 1;
  if (std::strncmp(value, kIdle, kIdleLen) != 0) return mode;
  const char* digits = value + kIdleLen;
  size_t ms = 0;
  size_t n = 0;
  for (; digits[n] != '\0'; ++n) {
    if (n == 5 || digits[n] < '0' || digits[n] > '9') return mode;
    ms = ms * 10 + static_cast<size_t>(digits[n] - '0');
  }
  if (n == 0 || ms > 60000) return mode;
  mode.at_idle = true;
  mode.idle_ms = ms;
  return mode;
}

// True when every tower activation sits in the decoder's storage.
inline bool VisionActivationsOwnNothing(const std::string& vit_scope) {
  return TensorMemoryPlanner::Instance().GetPoolCapacityBytes(
             PoolType::ACTIVATION, vit_scope) == 0;
}

inline std::string VisionSteadyWindowNs(
    std::chrono::steady_clock::time_point begin) {
  const auto ns = [](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               t.time_since_epoch()).count();
  };
  return " begin_ns=" + std::to_string(ns(begin)) +
         " end_ns=" + std::to_string(ns(std::chrono::steady_clock::now()));
}

// `rows` is the largest row count of the prefill-shaped Forwards.
template <typename T, typename LayerInfoT>
std::unique_ptr<LinearSharedBufferLease<T>> ReserveVisionSharedScratch(
    const std::vector<LayerInfoT>& vit_layers, size_t rows, const char* tag,
    PlanVisionPrecision vision_precision = PlanVisionPrecision::kNoPlan,
    size_t q8_frag_max_rows = kVisionQ8FragUnresolved) {
  size_t max_k = 0, max_n = 0;
  for (const LayerInfoT& info : vit_layers) {
    if (info.shape.size() != 2) continue;
    const bool quantized = info.dtype != GGML_TYPE_F32 &&
                           info.dtype != GGML_TYPE_F16 &&
                           info.dtype != GGML_TYPE_BF16;
    if (!quantized) continue;
    max_k = std::max(max_k, info.shape[0]);
    max_n = std::max(max_n, info.shape[1]);
  }
  if (rows == 0 || max_k == 0 || max_n == 0) return nullptr;
  int nsm = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device);
  CHECK_GT(nsm, 0) << tag << "cudaDevAttrMultiProcessorCount";
  auto lease =
      std::make_unique<LinearSharedBufferLease<T>>(rows, max_k, max_n, nsm);
  if (VisionQ8W8A16Requested(vision_precision)) {
    const bool within_frag_limit = rows <= VisionQ8FragMaxRows(q8_frag_max_rows);
    PrefillF16Demand demand;
    for (const LayerInfoT& info : vit_layers) {
      if (info.shape.size() == 2 && info.dtype == GGML_TYPE_Q8_0 &&
          info.shape[0] % 128 == 0) {
        demand.max_in_features = std::max(demand.max_in_features, info.shape[0]);
        if (within_frag_limit &&
            LinearGPU<T>::Q8W8A16FragEligible(info.shape[1], info.shape[0])) {
          continue;
        }
        demand.weight_elems = std::max(
            demand.weight_elems,
            Q8W8A16TileRows(info.shape[1], info.shape[0]) * info.shape[0]);
      }
    }
    if (demand.max_in_features != 0) {
      demand.layers = 1;
      LinearGPU<T>::ReservePrefillF16Buffers(lease.get(), rows, demand);
    }
  }
  return lease;
}

template <typename LlmT>
void EnsureVlmLlmMemoryPlanFinalized(LlmT* llm, const VlmScopePair& scopes) {
  if (llm == nullptr) return;
  ActiveScopeGuard g(scopes.llm());
  llm->FinalizeMemoryPlan();
}

// Call after the auxiliary stream has been drained.
template <typename LlmT, typename VitT>
void ReleaseVlmVisionScopeAfterSync(VitT* vit, LlmT* llm,
                                    const VlmScopePair& scopes) {
  if (vit != nullptr) vit->SetModelScope("");
  TensorMemoryPlanner::Instance().ReleaseModel(scopes.vit());
  EnsureVlmLlmMemoryPlanFinalized(llm, scopes);
}

template <typename LlmT, typename VitT>
void ReleaseVlmVisionScope(VitT* vit, LlmT* llm,
                           const VlmScopePair& scopes,
                           const UniqueAuxStream& aux_stream) {
  if (aux_stream.valid()) {
    CUDA_CHECK(cudaStreamSynchronize(aux_stream.get()));
  }
  ReleaseVlmVisionScopeAfterSync(vit, llm, scopes);
}

template <typename LlmT>
void ReleaseVlmLlmWeights(LlmT* llm, const VlmScopePair& scopes) {
  if (llm != nullptr) {
    ActiveScopeGuard g(scopes.llm());
    llm->ReleaseWeightsMemory();
  }
}

template <typename LlmT, typename VitT>
void ReleaseVlmLlmVitWeights(LlmT* llm, VitT* vit,
                             const VlmScopePair& scopes,
                             const UniqueAuxStream& aux_stream,
                             const UniqueAuxHandle& aux_handle) {
  if (llm != nullptr) {
    ActiveScopeGuard g(scopes.llm());
    llm->ReleaseWeightsMemory();
  }
  if (vit != nullptr) {
    ActiveScopeGuard g(scopes.vit(), aux_stream.get(), aux_handle.get());
    vit->ReleaseWeightsMemory();
  }
}

}  // namespace oris_ai
