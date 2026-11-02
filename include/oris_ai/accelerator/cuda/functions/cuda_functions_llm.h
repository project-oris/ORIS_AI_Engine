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

// LLM CUDA kernels. This release uses the d4 scales (use_ds4 = false).
// Reference: llama.cpp k_get_rows, rms_norm, rope_norm, mul_mat_vec_q.

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/layer/activation_type.h"
#include "oris_ai/gguf_reader/gguf_types.h"
#include "oris_ai/layer/glu_gate_kind.h"

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace oris_ai {

// Row-major FP16 copy of `n_elements` Q8_0 weight values.
void CUDADequantWeightToF16(ggml_type type, const void* weight, half* out,
                            int64_t n_elements, cudaStream_t stream);
// Saturates to +-65504.
void CUDACastF32ToF16Saturate(const float* in, half* out, size_t count,
                              cudaStream_t stream);
// Saturating cast of clamp(x, lo, hi).
void CUDACastF32ToF16Clamp(const float* in, half* out, size_t count, float lo,
                           float hi, cudaStream_t stream);
void CUDAGluToF16(const float* gate, const float* up, half* out,
                  size_t count, GluGateKind kind, cudaStream_t stream);
// `bytes` is a multiple of 16; `buffer` is initialized and larger than L2.
void CUDAEvictL2ByRead(const void* buffer, size_t bytes, cudaStream_t stream);
// Returns the largest finite |x|. Synchronizes `stream`.
float CUDAF16AbsMax(const half* in, size_t count, size_t* non_finite,
                    cudaStream_t stream);

// One CTA's greedy candidate; token is -1 when it saw no finite logit.
struct TokenSelectPartial {
  float score;
  int32_t token;
};

// Greedy selection over one FP32 vocabulary row.
void LaunchTokenSelectGreedy(const float* logits, size_t vocab_size,
                             int partial_count, TokenSelectPartial* partials,
                             int32_t* result, cudaStream_t stream);

// Per-slice softmax weight sums of one FP32 vocabulary row.
void LaunchTokenSelectSoftmaxSums(const float* logits, size_t vocab_size,
                                  int slice, int block_count,
                                  const float* max_score, double temperature,
                                  double window, double* block_sums,
                                  int32_t* block_counts, cudaStream_t stream);

// First id in [begin, end) past `residual`, else the last candidate.
void LaunchTokenSelectSoftmaxPick(const float* logits, int begin, int end,
                                  float max_score, double temperature,
                                  double window, double residual,
                                  int32_t* result, cudaStream_t stream);

// Embedding lookup.
template <typename T>
void CUDAGetRows(const Tensor<T>* weight_tensor,
                 const Tensor<T>* indices_tensor,
                 Tensor<T>* output_tensor,
                 const size_t batch_size,
                 const size_t seq_len,
                 const size_t embedding_dim);

// Embedding lookup from Q8_0 weights into FP32 rows.
void LaunchCUDAGetRowsQ8_0ToFP32(const block_q8_0* weight,
                                 const float* indices,
                                 float* output,
                                 int num_embeddings,
                                 int embedding_dim,
                                 int n_tok,
                                 cudaStream_t stream,
                                 float out_scale = 1.0f);

// Embedding lookup from F16 weights into FP32 rows.
void LaunchCUDAGetRowsF16ToFP32(const half* weight,
                                const float* indices,
                                float* output,
                                int num_embeddings,
                                int embedding_dim,
                                int n_tok,
                                cudaStream_t stream,
                                float out_scale = 1.0f);

void CUDARMSNormMMQ(float* input, const float* add, const float* weight,
                    float* output, block_q8_1_mmq* quantized,
                    size_t rows, size_t dim, float eps, bool use_ds4,
                    cudaStream_t stream);
// RMSNorm into `output`, also stored as D4 MMQ rows of the clamped values.
// Returns false, with nothing launched, for a width it does not cover.
bool CUDARMSNormMMQClamped(const float* input, const float* weight, float* output,
                           block_q8_1_mmq* quantized, size_t rows, size_t dim,
                           float eps, float clamp_lo, float clamp_hi,
                           cudaStream_t stream);

// RMSNorm, after the residual add when `add` is set, plus saturated FP16 rows.
void CUDARMSNormF16(float* input, const float* add, const float* weight,
                    float* output, half* f16, size_t rows, size_t dim,
                    float eps, cudaStream_t stream);

// `input_offset` is from the input start; `has_clamp` clamps every input load.
template <typename T>
void CUDARMSNorm(const Tensor<T>* input_tensor,
                 const Tensor<T>* weight_tensor,
                 Tensor<T>* output_tensor,
                 const size_t batch_size,
                 const size_t norm_size,
                 const T eps,
                 const size_t input_offset = 0,
                 const bool has_clamp = false,
                 const T clamp_lo = T(0),
                 const T clamp_hi = T(0));

// residual = (residual + RMSNorm(input)) * out_scale; input != residual.
template <typename T>
void CUDARMSNormThenAdd(Tensor<T>* residual, const Tensor<T>* input,
                        const Tensor<T>* weight, size_t rows, size_t norm_size,
                        T eps, T out_scale);
// Whether CUDARMSNormThenAdd runs its compact kernel.
bool CUDARMSNormThenAddCompact(size_t rows, size_t norm_size);
// Post-norm add, then RMSNorm with FP16 rows, in one launch.
// Needs CUDARMSNormThenAddCompact and a width multiple of 128.
void CUDARMSNormPostAddThenF16(const float* addend, const float* post_weight, float* residual,
                               const float* norm_weight, float* output, half* f16,
                               size_t rows, size_t dim, float post_eps, float norm_eps,
                               float out_scale, cudaStream_t stream);
// Post-norm add, then MMQ quantization, in one launch, on the same conditions.
void CUDARMSNormPostAddThenMMQ(const float* addend, const float* post_weight, float* residual,
                               block_q8_1_mmq* quantized, size_t rows, size_t dim,
                               float post_eps, float out_scale, bool use_ds4,
                               cudaStream_t stream);

// Whether CUDARMSNormWithQ8_1 splits rows of this width over several CTAs.
bool CUDARMSNormQ8_1RowSplit(size_t norm_size);

// RMSNorm that also writes Q8_1 blocks.
template <typename T>
void CUDARMSNormWithQ8_1(const Tensor<T>* input_tensor,
                         const Tensor<T>* weight_tensor,
                         Tensor<T>* output_tensor,
                         block_q8_1* q8_1_dst,
                         const size_t batch_size,
                         const size_t norm_size,
                         const T eps,
                         const size_t input_offset = 0);

// residual += add, then RMSNorm of the result.
template <typename T>
void CUDARMSNormFusedAdd(Tensor<T>* residual_tensor,
                         const Tensor<T>* add_tensor,
                         const Tensor<T>* weight_tensor,
                         Tensor<T>* output_tensor,
                         const size_t batch_size,
                         const size_t norm_size,
                         const T eps,
                         const size_t input_offset = 0);

// residual += add, then RMSNorm with Q8_1 blocks.
template <typename T>
void CUDARMSNormFusedAddWithQ8_1(Tensor<T>* residual_tensor,
                                  const Tensor<T>* add_tensor,
                                  const Tensor<T>* weight_tensor,
                                  Tensor<T>* output_tensor,
                                  block_q8_1* q8_1_dst,
                                  const size_t batch_size,
                                  const size_t norm_size,
                                  const T eps,
                                  const size_t input_offset = 0);

template <typename T>
void CUDARope(const Tensor<T>* input_tensor,
              const Tensor<T>* pos_tensor,
              const Tensor<float>* freq_factors_tensor,
              Tensor<T>* output_tensor,
              const size_t batch_size,
              const size_t seq_len,
              const size_t num_heads,
              const size_t head_dim,
              const size_t d_model,
              const float theta_scale,
              const float freq_scale,
              const float ext_factor,
              const float attn_factor,
              const float corr_dims_0,
              const float corr_dims_1);

template <typename T>
void CUDARopeInPlace(Tensor<T>* inout_tensor,
                     const Tensor<T>* pos_tensor,
                     const Tensor<float>* freq_factors_tensor,
                     const size_t batch_size,
                     const size_t seq_len,
                     const size_t num_heads,
                     const size_t head_dim,
                     const size_t d_model,
                     const float theta_scale,
                     const float freq_scale,
                     const float ext_factor,
                     const float attn_factor,
                     const float corr_dims_0,
                     const float corr_dims_1);

// RoPE written directly to the KV cache at `cache_position`.
template <typename T>
void CUDARopeWithKVWrite(
    const Tensor<T>* input_tensor,
    const Tensor<T>* pos_tensor,
    const Tensor<float>* freq_factors_tensor,
    T* kv_dst,
    const size_t batch_size,
    const size_t seq_len,
    const size_t num_heads,
    const size_t head_dim,
    const size_t d_model,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_dims_0,
    const float corr_dims_1,
    const size_t dst_row_stride,
    const size_t cache_position);

// RoPE with split-half pairs (x[i], x[i + n_dims / 2]).
template <typename T>
void CUDARopeNEOX(const Tensor<T>* input_tensor,
                  const Tensor<T>* pos_tensor,
                  const Tensor<float>* freq_factors_tensor,
                  Tensor<T>* output_tensor,
                  const size_t batch_size,
                  const size_t seq_len,
                  const size_t num_heads,
                  const size_t head_dim,
                  const size_t d_model,
                  const float theta_scale,
                  const float freq_scale,
                  const float ext_factor,
                  const float attn_factor,
                  const float corr_dims_0,
                  const float corr_dims_1);

template <typename T>
void CUDARopeInPlaceNEOX(Tensor<T>* inout_tensor,
                          const Tensor<T>* pos_tensor,
                          const Tensor<float>* freq_factors_tensor,
                          const size_t batch_size,
                          const size_t seq_len,
                          const size_t num_heads,
                          const size_t head_dim,
                          const size_t d_model,
                          const float theta_scale,
                          const float freq_scale,
                          const float ext_factor,
                          const float attn_factor,
                          const float corr_dims_0,
                          const float corr_dims_1);

template <typename T>
void CUDARopeWithKVWriteNEOX(
    const Tensor<T>* input_tensor,
    const Tensor<T>* pos_tensor,
    const Tensor<float>* freq_factors_tensor,
    T* kv_dst,
    const size_t batch_size,
    const size_t seq_len,
    const size_t num_heads,
    const size_t head_dim,
    const size_t d_model,
    const float theta_scale,
    const float freq_scale,
    const float ext_factor,
    const float attn_factor,
    const float corr_dims_0,
    const float corr_dims_1,
    const size_t dst_row_stride,
    const size_t cache_position);

// `data` moved back by `base_elems`, so a kernel indexes by logical row.
template <typename T>
inline const T* RowBasedKVPointer(const T* data, size_t base_elems) {
  return reinterpret_cast<const T*>(reinterpret_cast<uintptr_t>(data) -
                                    base_elems * sizeof(T));
}

// Tin is the K/V type, Tout the Q and output type.
template <typename Tin, typename Tout = Tin>
void CUDAFlashAttention(const Tensor<Tout>* q_tensor,
                        const Tensor<Tin>* k_tensor,
                        const Tensor<Tin>* v_tensor,
                        Tensor<Tout>* output_tensor,
                        const size_t batch_size,
                        const size_t seq_len,
                        const size_t kv_len,
                        const size_t num_query_heads,
                        const size_t num_kv_heads,
                        const size_t head_dim,
                        const float scale,
                        float2* fixup_ptr = nullptr,
                        int fixup_nblocks = 0,
                        float* split_kv_acc_ptr = nullptr,
                        float* split_kv_m_ptr = nullptr,
                        float* split_kv_l_ptr = nullptr,
                        int split_kv_max_splits = 0,
                        int swa_window = 0,
                        const int32_t* span_descs = nullptr,
                        int n_spans = 0,
                        bool allow_mma = true,
                        size_t q_row_offset = 0,
                        size_t kv_row_base = 0);

// Split-KV first step: writes the (m, l, acc) partials of each split.
template <typename Tin, typename Tout = Tin>
void CUDAFlashAttentionSplitKVStage1(const Tensor<Tout>* q_tensor,
                                     const Tensor<Tin>* k_tensor,
                                     const Tensor<Tin>* v_tensor,
                                     float* split_kv_acc_ptr,
                                     float* split_kv_m_ptr,
                                     float* split_kv_l_ptr,
                                     const size_t batch_size,
                                     const size_t seq_len,
                                     const size_t kv_len,
                                     const size_t num_query_heads,
                                     const size_t num_kv_heads,
                                     const size_t head_dim,
                                     const float scale,
                                     int num_splits,
                                     int swa_window,
                                     size_t q_row_offset = 0,
                                     size_t kv_row_base = 0);

// Split-KV reduction of the partials into the output.
void CUDAFlashAttentionSplitKVReduce(
    const float* split_kv_acc_ptr,
    const float* split_kv_m_ptr,
    const float* split_kv_l_ptr,
    Tensor<float>* output_tensor,
    int num_splits, int batch_size, int num_q_heads, int seq_len,
    int head_dim, int out_row_offset = 0);

// source_K is a multiple of 32; padded_K is a multiple of 256, >= source_K.
// The clamp also applies to the padding zeros.
void CUDAQuantizeFP32ToQ8_1_MMQPadded(
    const float* src, block_q8_1_mmq* dst, int M, int source_K, int padded_K,
    cudaStream_t stream, bool has_clamp = false,
    float clamp_lo = 0.f, float clamp_hi = 0.f);

// Largest absolute input behind `n_blocks` MMQ blocks; diagnostics only.
float CUDAMMQActivationAbsMax(const block_q8_1_mmq* act, size_t n_blocks,
                              bool use_ds4, cudaStream_t stream);

// head_dim is 128, 256 or 512; output holds batch * heads * head_dim / 32 blocks.
void CUDAFlashAttentionSplitKVReduceQ8_1(
    const float* acc, const float* maxima, const float* sums,
    block_q8_1* output, int splits, int batch, int heads, int head_dim,
    cudaStream_t stream);

// Decode attention storing batch_size * num_query_heads * head_dim / 32 blocks.
bool CUDAFlashAttentionDecodeQ8_1Supported(size_t head_dim, size_t seq_len,
                                           size_t num_query_heads,
                                           size_t num_kv_heads);
void CUDAFlashAttentionDecodeQ8_1(
    const float* Q, const half* K, const half* V, block_q8_1* output,
    size_t batch_size, size_t kv_len, size_t num_query_heads,
    size_t num_kv_heads, size_t head_dim, float scale, int swa_window,
    cudaStream_t stream);

// Prefill (seq_len > 1) attention with block_q8_1_mmq rows; [K / 128, rows].
// CUDAFlashAttentionPrefillMMQ returns whether those rows were stored.
bool CUDAFlashAttentionPrefillMMQSupported(size_t head_dim, size_t seq_len,
                                           size_t num_query_heads,
                                           size_t num_kv_heads);
// Launch counts of the attention schedules, for diagnostics.
uint64_t FlashAttentionDecodePipeLaunchCount();
uint64_t FlashAttentionPrefillPipeLaunchCount();
uint64_t FlashAttentionPrefillPipeWideLaunchCount();
bool CUDAFlashAttentionPrefillMMQ(
    const float* Q, const half* K, const half* V, float* output,
    block_q8_1_mmq* mmq_output, bool use_ds4,
    size_t batch_size, size_t seq_len, size_t kv_len,
    size_t num_query_heads, size_t num_kv_heads, size_t head_dim,
    float scale, float2* fixup_ptr, int fixup_nblocks, int swa_window,
    const int32_t* span_descs, int n_spans, cudaStream_t stream);
// Prefill (seq_len > 1) attention with saturated FP16 rows in `f16_output`.
// CUDAFlashAttentionPrefillF16 returns whether those rows were stored.
bool CUDAFlashAttentionPrefillF16Supported(size_t head_dim, size_t seq_len,
                                           size_t num_query_heads,
                                           size_t num_kv_heads);
bool CUDAFlashAttentionPrefillF16(
    const float* Q, const half* K, const half* V, float* output,
    half* f16_output, size_t batch_size, size_t seq_len, size_t kv_len,
    size_t num_query_heads, size_t num_kv_heads, size_t head_dim,
    float scale, float2* fixup_ptr, int fixup_nblocks, int swa_window,
    const int32_t* span_descs, int n_spans, cudaStream_t stream);

// `num_elements` is a multiple of 32; 32 values make one block.
void CUDAQuantizeFP32ToQ8_1(const float* src, block_q8_1* dst,
                            size_t num_elements, cudaStream_t stream);

void CUDAQuantizeFP32ToQ8_1_MMQ(const float* src, block_q8_1_mmq* dst,
                                int M, int K, bool use_ds4,
                                cudaStream_t stream,
                                bool has_clamp = false, float clamp_lo = 0.f,
                                float clamp_hi = 0.f,
                                const float* fold_bias_gelu = nullptr,
                                bool fold_gelu_tanh = false);

// Quantizes SiLU(gate) * up.
void CUDAQuantizeSwiGLUToQ8_1_MMQ(
    const float* gate, const float* up,
    block_q8_1_mmq* dst,
    int M, int K, bool use_ds4, cudaStream_t stream);

// Quantizes the GLU gate of `gate_kind` times up.
void CUDAQuantizeGluToQ8_1_MMQ(
    const float* gate, const float* up,
    block_q8_1_mmq* dst,
    int M, int K, bool use_ds4, GluGateKind gate_kind, cudaStream_t stream);

// Storage of a Q8_0 weight on the GPU, passed to every Q8_0 kernel.
enum class Q8_0Layout { kBlocks, kRowChunks };

// Whether a weight of this shape can take the row-chunk layout.
bool CUDAQ8_0RowChunksSupported(int out_features, int in_features);
// Rewrites a block_q8_0 weight into the row-chunk layout in place.
void CUDAQ8_0ToRowChunks(block_q8_0* weight, int out_features, int in_features,
                         cudaStream_t stream);

// Whether a weight of this shape can take the padded row-chunk layout.
bool CUDAQ8_0PaddedRowChunksSupported(int out_features, int in_features);
// `packed` and `padded` must not overlap.
void CUDAQ8_0ToPaddedRowChunks(const block_q8_0* packed, block_q8_0* padded, int out_features,
                               int in_features, cudaStream_t stream);
// Version of the row-chunk byte order.
uint32_t CUDAQ8_0RowChunkLayoutVersion();
// Host version of CUDAQ8_0ToRowChunks; in_features is a multiple of 256.
void Q8_0RowChunksHost(const block_q8_0* src, void* dst, int rows, int in_features);
// Bytes of one padded row and of the whole padded weight.
size_t Q8_0PaddedRowChunksRowBytes(int in_features);
size_t Q8_0PaddedRowChunksBytes(int out_features, int in_features);
// Host version of CUDAQ8_0ToPaddedRowChunks for rows [row0, row0 + rows).
void Q8_0PaddedRowChunksHost(const block_q8_0* src, void* dst, int row0, int rows,
                             int out_features, int in_features);
void CUDAMmqQ8_0PaddedRowChunksQ8_1StreamKMMA(const block_q8_0* weight, const int* activation,
                                              float* output, float* tmp_fixup, int out_features,
                                              int in_features, int batch, int nsm, int mmq_x,
                                              cudaStream_t stream);

// MMVQ: Q8_0 weight x Q8_1 activation, for batch <= 8.
void CUDAMmvqQ8_0Q8_1(const block_q8_0* weight, Q8_0Layout layout,
                      const block_q8_1* activation,
                      float* output,
                      int out_features, int in_features, int batch,
                      cudaStream_t stream);

// output_dst[i] = MMVQ result + residual[i].
void CUDAMmvqQ8_0Q8_1WithResidual(
    const block_q8_0* weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output_dst,
    const float* residual,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

// output = SiLU(gate_weight * x) * (up_weight * x).
void CUDAMmvqQ8_0Q8_1FusedGateUp(
    const block_q8_0* gate_weight,
    const block_q8_0* up_weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

void CUDAMmvqQ8_0Q8_1FusedGateUp_LogicPort(
    const block_q8_0* gate_weight,
    const block_q8_0* up_weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

void CUDAMmvqQ8_0Q8_1_LogicPort(
    const block_q8_0* weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

// MMVQ written as FP16 into the KV cache at row `kv_pos`.
void CUDAMmvqQ8_0Q8_1WithKVWriteF16(
    const block_q8_0* weight, Q8_0Layout layout,
    const block_q8_1* activation,
    half* kv_dst,
    int out_features, int in_features, int batch,
    int kv_stride, int kv_pos,
    cudaStream_t stream);

// q, k and v projections of one token: q and k as FP32, v as FP16 in `kv_dst`.
void CUDAMmvqQ8_0Q8_1FusedQKV(
    const block_q8_0* q_weight,
    const block_q8_0* k_weight,
    const block_q8_0* v_weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* q_out, float* k_out, half* kv_dst,
    int nq, int nk, int nv, int in_features,
    int kv_stride, int kv_pos,
    cudaStream_t stream);

// Multi-token MMVQ variants: batch in [2, 8].
void CUDAMmvqQ8_0Q8_1_Ncols(const block_q8_0* weight, Q8_0Layout layout,
                            const block_q8_1* activation,
                            float* output,
                            int out_features, int in_features, int batch,
                            cudaStream_t stream);

void CUDAMmvqQ8_0Q8_1WithResidual_Ncols(
    const block_q8_0* weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output_dst,
    const float* residual,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

void CUDAMmvqQ8_0Q8_1WithKVWriteF16_Ncols(
    const block_q8_0* weight, Q8_0Layout layout,
    const block_q8_1* activation,
    half* kv_dst,
    int out_features, int in_features, int batch,
    int kv_stride, int kv_pos,
    cudaStream_t stream);

void CUDAMmvqQ8_0Q8_1_NcolsLogicPort(
    const block_q8_0* weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

void CUDAMmvqQ8_0Q8_1FusedGateUp_NcolsLogicPort(
    const block_q8_0* gate_weight,
    const block_q8_0* up_weight, Q8_0Layout layout,
    const block_q8_1* activation,
    float* output,
    int out_features, int in_features, int batch,
    cudaStream_t stream);

void CUDAMmqQ8_0Q8_1StreamKMMA(const block_q8_0* weight, Q8_0Layout layout,
                               const int* activation,
                               float* output,
                               float* tmp_fixup,
                               int out_features, int in_features, int batch,
                               int nsm, int mmq_x,
                               bool need_check,
                               cudaStream_t stream);


}  // namespace oris_ai
