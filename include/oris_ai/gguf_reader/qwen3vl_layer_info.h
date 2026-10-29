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

#include "oris_ai/gguf_reader/gguf_types.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace oris_ai {

// RoPE mode from the GGUF metadata: IMROPE for qwen3vl, STANDARD_1D for qwen3.
enum class Qwen3RopeMode { IMROPE, STANDARD_1D };

// Decoder architecture from general.architecture.
enum class QwenArchitecture { QWEN3, QWEN3VL };

// One Qwen3-VL GGUF tensor and its role.
struct Qwen3VLLayerInfo {
  enum class Role {
    TOKEN_EMBD,
    OUTPUT_NORM,
    OUTPUT_WEIGHT,     // absent when the LM head is tied
    BLK_ATTN_NORM,
    BLK_ATTN_Q,
    BLK_ATTN_Q_NORM,
    BLK_ATTN_K,
    BLK_ATTN_K_NORM,
    BLK_ATTN_V,
    BLK_ATTN_OUT,
    BLK_FFN_NORM,
    BLK_FFN_GATE,
    BLK_FFN_UP,
    BLK_FFN_DOWN,
    V_PATCH_EMBD_W,
    V_PATCH_EMBD_W_1,
    V_PATCH_EMBD_B,
    V_POSITION_EMBD,
    V_POST_LN_W,
    V_POST_LN_B,
    V_BLK_LN1_W, V_BLK_LN1_B,
    V_BLK_ATTN_QKV_W,
    V_BLK_ATTN_QKV_B,
    V_BLK_ATTN_OUT_W,
    V_BLK_ATTN_OUT_B,
    V_BLK_LN2_W, V_BLK_LN2_B,
    V_BLK_FFN_UP_W,
    V_BLK_FFN_UP_B,
    V_BLK_FFN_DOWN_W,
    V_BLK_FFN_DOWN_B,
    MM_0_W, MM_0_B,
    MM_2_W, MM_2_B,
    V_DEEPSTACK_NORM_W, V_DEEPSTACK_NORM_B,
    V_DEEPSTACK_FC1_W,  V_DEEPSTACK_FC1_B,
    V_DEEPSTACK_FC2_W,  V_DEEPSTACK_FC2_B,
    UNKNOWN,
  };

  std::string name;
  Role role = Role::UNKNOWN;
  ggml_type dtype = GGML_TYPE_F32;
  std::vector<size_t> shape;
  size_t data_offset = 0;
  size_t data_size_bytes = 0;
  bool is_vision = false;
  std::optional<size_t> block_id;
  std::optional<size_t> deepstack_idx;   // the ViT block index, not 0..2
};

// Text decoder settings from the GGUF. The defaults are the 2B values.
struct Qwen3VLLLMConfig {
  uint32_t block_count = 28;
  uint32_t context_length = 262144;
  uint32_t embedding_dim = 2048;
  uint32_t ffn_int = 6144;
  uint32_t num_heads = 16;
  uint32_t num_kv_heads = 8;
  uint32_t head_dim = 128;
  uint32_t vocab_size = 151936;
  float rope_freq_base = 5e6f;
  float rms_norm_eps = 1e-6f;
  std::array<int32_t, 4> rope_sections = {24, 20, 20, 0};
  Qwen3RopeMode rope_mode = Qwen3RopeMode::IMROPE;
  QwenArchitecture architecture = QwenArchitecture::QWEN3VL;
  uint32_t n_deepstack_layers = 3;
  bool tied_embedding = true;
};

// Vision encoder settings from the mmproj GGUF.
struct Qwen3VisionConfig {
  uint32_t image_size = 768;
  uint32_t patch_size = 16;
  uint32_t embed_dim = 1024;
  uint32_t ffn_int = 4096;
  uint32_t block_count = 24;
  uint32_t num_heads = 16;
  uint32_t head_dim = 64;
  uint32_t projection_dim = 2048;
  uint32_t spatial_merge = 2;
  std::vector<float> image_mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> image_std  = {0.5f, 0.5f, 0.5f};
  float layer_norm_eps = 1e-6f;
  std::vector<bool> is_deepstack_layers;
  std::string projector_type;
  bool use_gelu = true;
};

}  // namespace oris_ai
