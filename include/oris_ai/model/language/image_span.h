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

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace oris_ai {

// One image chunk of a prefill sequence; the layout is part of the kernel ABI.
struct ImageSpan {
  int32_t token_offset       = 0;
  int32_t n_tokens           = 0;
  int32_t carrier_offset_rows = 0;
};
static_assert(sizeof(ImageSpan) == 12,
              "ImageSpan must be 3*int32_t = 12 bytes");
static_assert(alignof(ImageSpan) == 4,
              "ImageSpan must be 4-byte aligned");
static_assert(std::is_standard_layout<ImageSpan>::value,
              "ImageSpan must be standard layout for GPU reinterpret_cast");
static_assert(std::is_trivially_copyable<ImageSpan>::value,
              "ImageSpan must be trivially copyable for cudaMemcpy");
static_assert(offsetof(ImageSpan, token_offset) == 0,
              "ImageSpan::token_offset must be at offset 0");
static_assert(offsetof(ImageSpan, n_tokens) == 4,
              "ImageSpan::n_tokens must be at offset 4");
static_assert(offsetof(ImageSpan, carrier_offset_rows) == 8,
              "ImageSpan::carrier_offset_rows must be at offset 8");

}  // namespace oris_ai
