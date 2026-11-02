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

#include "oris_ai/gguf_reader/qwen3vl_layer_info.h"

#include <string>
#include <vector>

namespace oris_ai {

// Accepted export variants.
enum class Qwen3VLQuantVariant {
  Q8_0,
};

// Checks the weight roles and the config: Q8_0 weights and F32 norms. `err_out`
// must be non-null and receives the first violation when this returns false.
bool ValidateSupportedQwen3VLQuantFrontier(
    const Qwen3VLLLMConfig& cfg,
    const std::vector<Qwen3VLLayerInfo>& layers,
    std::string* err_out);

bool ValidateSupportedQwen3VLQuantFrontier(
    const Qwen3VLLLMConfig& cfg,
    const std::vector<Qwen3VLLayerInfo>& layers,
    Qwen3VLQuantVariant* variant_out,
    std::string* err_out);

// `err_out` must be non-null; it stays empty on accept.
bool ValidateOutputWeightShape(const Qwen3VLLLMConfig& cfg,
                               const Qwen3VLLayerInfo& info,
                               std::string* err_out);

}  // namespace oris_ai
