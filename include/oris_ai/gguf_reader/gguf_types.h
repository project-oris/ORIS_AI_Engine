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

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_fp16.h>

namespace oris_ai {

constexpr const char* GGUF_MAGIC = "GGUF";
constexpr uint32_t GGUF_VERSION = 3;
constexpr size_t GGUF_DEFAULT_ALIGNMENT = 32;
constexpr uint32_t GGUF_MAX_DIMS = 4;

// GGML tensor types, with the values of llama.cpp ggml.h. This release reads
// F32, F16 and Q8_0 only.
enum ggml_type {
  GGML_TYPE_F32   = 0,
  GGML_TYPE_F16   = 1,
  GGML_TYPE_Q4_0  = 2,
  GGML_TYPE_Q4_1  = 3,
  GGML_TYPE_Q5_0  = 6,
  GGML_TYPE_Q5_1  = 7,
  GGML_TYPE_Q8_0  = 8,
  GGML_TYPE_Q8_1  = 9,
  GGML_TYPE_Q2_K  = 10,
  GGML_TYPE_Q3_K  = 11,
  GGML_TYPE_Q4_K  = 12,
  GGML_TYPE_Q5_K  = 13,
  GGML_TYPE_Q6_K  = 14,
  GGML_TYPE_Q8_K  = 15,
  GGML_TYPE_BF16  = 30,
};

inline const char* DtypeName(ggml_type t) {
  switch (t) {
    case GGML_TYPE_F32:  return "F32";
    case GGML_TYPE_F16:  return "F16";
    case GGML_TYPE_Q4_0: return "Q4_0";
    case GGML_TYPE_Q4_1: return "Q4_1";
    case GGML_TYPE_Q5_0: return "Q5_0";
    case GGML_TYPE_Q5_1: return "Q5_1";
    case GGML_TYPE_Q8_0: return "Q8_0";
    case GGML_TYPE_Q8_1: return "Q8_1";
    case GGML_TYPE_Q2_K: return "Q2_K";
    case GGML_TYPE_Q3_K: return "Q3_K";
    case GGML_TYPE_Q4_K: return "Q4_K";
    case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K";
    case GGML_TYPE_Q8_K: return "Q8_K";
    case GGML_TYPE_BF16: return "BF16";
    default:             return "UNKNOWN";
  }
}

// GGUF metadata value types.
enum gguf_type {
  GGUF_TYPE_UINT8   = 0,
  GGUF_TYPE_INT8    = 1,
  GGUF_TYPE_UINT16  = 2,
  GGUF_TYPE_INT16   = 3,
  GGUF_TYPE_UINT32  = 4,
  GGUF_TYPE_INT32   = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL    = 7,
  GGUF_TYPE_STRING  = 8,
  GGUF_TYPE_ARRAY   = 9,
  GGUF_TYPE_UINT64  = 10,
  GGUF_TYPE_INT64   = 11,
  GGUF_TYPE_FLOAT64 = 12,
};

// Q8_0 block: an FP16 scale and 32 int8 values.
constexpr int QK8_0 = 32;
struct block_q8_0 {
  uint16_t d;
  int8_t qs[QK8_0];
};

// Reference: llama.cpp ggml-common.h
constexpr int QK8_1 = 32;
struct block_q8_1 {
  union {
    struct {
      uint16_t d;
      uint16_t s;
    };
    uint32_t ds;
  };
  int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == sizeof(uint32_t) + QK8_1,
              "wrong q8_1 block size");

// Reference: llama.cpp ggml-cuda/mmq.cuh block_q8_1_mmq
// Four Q8_1 blocks in the MMQ tile layout. This release uses the d4 scales.
struct block_q8_1_mmq {
  union {
    float d4[4];
    __half2 ds4[4];
    uint32_t ds_raw[4];
  };
  int8_t qs[4 * QK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 4 * sizeof(block_q8_1),
              "wrong q8_1_mmq block size");
static_assert(sizeof(block_q8_1_mmq) == 16 + 4 * QK8_1,
              "block_q8_1_mmq DS4 union must preserve 144B ABI");

// Tensor entry of a GGUF file.
struct TensorInfo {
  std::string name;
  uint32_t n_dims;
  std::vector<int64_t> dims;
  ggml_type type;
  uint64_t offset;
  size_t size;
};

static_assert(sizeof(float) == 4, "GGUF parsing requires 32-bit float");
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "GGUF parsing requires a little-endian host");
#endif

// FP16 and FP32 conversion. Reference: Maratyszcza/FP16 through ggml.
namespace fp16 {

inline float fp32_from_bits(uint32_t w) {
  union { uint32_t as_bits; float as_value; } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f) {
  union { float as_value; uint32_t as_bits; } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

inline float to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif

  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign | (two_w < denormalized_cutoff ?
    fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));

  return fp32_from_bits(result);
}

inline uint16_t from_fp32(float x) {
  const uint32_t bits = fp32_to_bits(x);
  const uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = bits & 0x7FFFFF;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    uint16_t h = static_cast<uint16_t>(sign | (mantissa >> shift));
    const uint32_t round_bit = 1u << (shift - 1);
    if ((mantissa & round_bit) != 0 &&
        ((mantissa & (round_bit - 1)) != 0 || (h & 1u) != 0)) {
      ++h;
    }
    return h;
  }

  if (exponent >= 31) {
    if (((bits >> 23) & 0xFF) == 0xFF && mantissa != 0) {
      return static_cast<uint16_t>(sign | 0x7E00);
    }
    return static_cast<uint16_t>(sign | 0x7C00);
  }

  uint16_t h = static_cast<uint16_t>(sign | (exponent << 10) | (mantissa >> 13));
  const uint32_t round_bit = 0x1000;
  if ((mantissa & round_bit) != 0 &&
      ((mantissa & (round_bit - 1)) != 0 || (h & 1u) != 0)) {
    ++h;
  }
  return h;
}

}  // namespace fp16

}  // namespace oris_ai
