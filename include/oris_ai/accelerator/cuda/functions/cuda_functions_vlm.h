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

// CUDA kernels of the Qwen3-VL vision encoder and of the steps that join
// the vision features to the language model.

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/layer/activation_type.h"
#include "oris_ai/gguf_reader/gguf_types.h"

#include <cstdint>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace oris_ai {

// Interleaved multimodal RoPE (four sections) of the Qwen3-VL decoder.
void LaunchQwen3IMROPEInplace(
    float* qk_inout,
    const int32_t* positions,
    int B, int S, int H, int D, int n_rot,
    int sec0, int sec1, int sec2, int sec3,
    float theta_scale,
    const float* freq_factors,
    cudaStream_t stream);

void LaunchQwen3IMROPEKWriteF32(
    const float* k_in,
    const int32_t* positions,
    float* kv_dst, size_t dst_row_stride, size_t cache_position,
    int B, int S, int H, int D, int n_rot,
    int sec0, int sec1, int sec2, int sec3,
    float theta_scale,
    const float* freq_factors,
    cudaStream_t stream);

void LaunchQwen3IMROPEKWriteF16(
    const float* k_in,
    const int32_t* positions,
    half*  kv_dst, size_t dst_row_stride, size_t cache_position,
    int B, int S, int H, int D, int n_rot,
    int sec0, int sec1, int sec2, int sec3,
    float theta_scale,
    const float* freq_factors,
    cudaStream_t stream);

// dst[r, c] = gelu_erf(dst[r, c] + bias[c]), in place.
void LaunchBiasAddGeluErfInplace(float* dst, const float* bias,
                                 size_t rows, size_t cols,
                                 cudaStream_t stream);

// Non-causal ViT attention, head_dim 64 (2B) or 72 (8B). qkv is [B, M, 3*H*head_dim], out
// is [B, M, H*head_dim]; the caller passes scale = 1 / sqrt(head_dim). The
// prepacked kernel needs workspace_h2_count =
// B * M * 3 * H * ((head_dim+15)/16)*8 / 2 half2 slots; without a workspace
// the launcher uses a slower kernel.
void LaunchQwen3VisionAttentionForward(
    const float* qkv,
    float* out,
    int B, int M, int H, int head_dim,
    float scale,
    cudaStream_t stream,
    void* workspace_qkv_h2 = nullptr,
    size_t workspace_h2_count = 0);

// Q, K and V as three [B, M, H*head_dim] FP32 tensors.
bool Qwen3VisionSeparatePrepackUsesRows(const float* q, const float* k,
                                        const float* v, const void* workspace);
void LaunchQwen3VisionAttentionForwardSeparate(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int B, int M, int H, int head_dim,
    float scale,
    cudaStream_t stream,
    void* workspace_qkv_h2,
    size_t workspace_h2_count);

void LaunchQwen3VisionAttentionForwardSeparateF16Output(
    const float* q,
    const float* k,
    const float* v,
    half* out,
    int B, int M, int H, int head_dim,
    float scale,
    cudaStream_t stream,
    void* workspace_qkv_h2,
    size_t workspace_h2_count);

bool Qwen3VisionAttentionFusedQkvUsesPrepack();

// Adds the QKV bias and applies the vision MRoPE to Q and K before the
// attention.
void LaunchQwen3VisionAttentionForwardBiasMRoPE(
    const float* qkv,
    const float* qkv_bias,
    const int32_t* positions,
    int sec0,
    float theta_scale,
    float* out,
    int B, int M, int H, int head_dim,
    float scale,
    cudaStream_t stream,
    void* workspace_qkv_h2,
    size_t workspace_h2_count);

void LaunchQwen3VisionAttentionForwardBiasMRoPEF16Output(
    const float* qkv,
    const float* qkv_bias,
    const int32_t* positions,
    int sec0,
    float theta_scale,
    half* out,
    int B, int M, int H, int head_dim,
    float scale,
    cudaStream_t stream,
    void* workspace_qkv_h2,
    size_t workspace_h2_count);

// Attention on half2 rows already in the prepacked workspace layout.
void LaunchQwen3VisionAttentionForwardPrepacked(
    const void* qkv_h2, float* out, int B, int M, int H, int head_dim,
    float scale, cudaStream_t stream);
void LaunchQwen3VisionAttentionForwardPrepackedF16Output(
    const void* qkv_h2, half* out, int B, int M, int H, int head_dim,
    float scale, cudaStream_t stream);

// Reference: llama.cpp rope_vision.
void LaunchQwen3VisionMRoPEStridedInplace(
    float* qk_inout,
    const int32_t* positions,
    int B, int S, int H, int head_dim,
    int sec0, int sec1,
    float theta_scale,
    int row_stride_elems,
    const float* bias,
    cudaStream_t stream);

void LaunchQwen3PositionResizeBilinearFP32(
    const float* base,         // device, [H_base * W_base * embed_dim] HWC
    float* dst,                // device, [H_pre * W_pre * embed_dim] HWC
    int H_base, int W_base,
    int H_pre, int W_pre,
    int embed_dim,
    cudaStream_t stream);

// Adds one DeepStack feature stripe to the decoder hidden state at the
// image token rows, in place.
void LaunchDeepStackInjectionAdd(
    float* hidden,
    const float* hidden_concat,
    int32_t deepstack_token_offset,
    int32_t deepstack_n_tokens,
    int32_t hidden_stride_elems_per_token,
    int32_t concat_stride_elems_per_token,
    int32_t hidden_dim,
    int32_t deepstack_idx,
    cudaStream_t stream);

// Overwrites the decoder hidden state at the image token rows with the
// image embedding.
void LaunchQwen3MainFeatureReplace(
    float* hidden,
    const float* hidden_concat,
    int32_t image_chunk_token_offset,
    int32_t n_image_tokens,
    int32_t hidden_stride_elems_per_token,
    int32_t concat_stride_elems_per_token,
    int32_t hidden_dim,
    cudaStream_t stream);

// dst[r, c] = gelu_tanh(dst[r, c] + bias[c]), in place.
void LaunchBiasAddGeluTanhInplace(float* dst, const float* bias,
                                  size_t rows, size_t cols,
                                  cudaStream_t stream);

enum class TokenIndexMode {
  SigLIP = 0,
  Qwen3  = 1,
};

// Builds the im2col operand [n_pos, C*P*P] of the patch embedding from an
// NCHW image.
void LaunchPatchEmbedIm2Col(const float* image_nchw, float* xcol,
                             int C, int image_h, int image_w, int P,
                             int pos_h, int pos_w,
                             TokenIndexMode mode, cudaStream_t stream);

void LaunchMainFeatureReplaceMultiSpan(
    float* hidden,
    const float* carrier,
    const int32_t* span_descs_gpu,
    int32_t n_spans,
    int32_t max_tokens_per_span,
    int32_t hidden_dim,
    int32_t hidden_stride_elems,
    int32_t carrier_stride_elems,
    cudaStream_t stream);

// dst[r, c] += bias[c], in place.
void LaunchBiasAddBroadcastInplace(
    float* dst,
    const float* bias,
    size_t rows, size_t out_features,
    cudaStream_t stream);

// As above with rows row_stride_elems apart.
void LaunchBiasAddBroadcastStridedInplace(
    float* dst,
    const float* bias,
    size_t rows, size_t out_features, size_t row_stride_elems,
    cudaStream_t stream);

// residual[r, c] += linear_out[r, c] + bias[c].
void LaunchBiasAddIntoResidualInplace(
    float* residual,
    const float* linear_out,
    const float* bias,
    size_t rows, size_t out_features,
    cudaStream_t stream);

// dst = linear_out + bias, and dst = (linear_out + bias) + residual.
void LaunchBiasAddToOutput(
    float* dst,
    const float* linear_out,
    const float* bias,
    size_t rows, size_t out_features,
    cudaStream_t stream);

void LaunchBiasAddResidualToOutput(
    float* dst,
    const float* linear_out,
    const float* bias,
    const float* residual,
    size_t rows, size_t out_features,
    cudaStream_t stream);

}  // namespace oris_ai
