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
#include "oris_ai/gguf_reader/qwen3vl_parser.h"
#include "oris_ai/model/vlm/vlm_artifact_resolver.h"

#include <memory>
#include <string>
#include <vector>

namespace oris_ai {

// A validated pair with its parsed readers and parsers.
struct PreparedQwen3VLArtifactPair {
  std::string llm_path;
  std::string mmproj_path;
  std::unique_ptr<GGUFReader> llm_reader;
  std::unique_ptr<Qwen3VLParser> llm_parser;
  std::unique_ptr<GGUFReader> mmproj_reader;
  std::unique_ptr<Qwen3VLParser> mmproj_parser;
  ArtifactObservedIdentity observed_identity;
};

// Checks the topology and weight types of the 2B and 8B model files.
bool ValidatePinnedQwen3VLArtifactProfile(
    Qwen3VLSize size,
    LlmQuant quant,
    const Qwen3VLLLMConfig& cfg,
    const std::vector<Qwen3VLLayerInfo>& layers,
    std::string* error);

// Reads and cross-checks a pair without GPU allocation.
bool InspectQwen3VLArtifactPair(
    const std::string& llm_path,
    const std::string& mmproj_path,
    const ArtifactExpectedIdentity* expected_identity,
    ArtifactObservedIdentity* observed_identity,
    std::string* error);

// A successful result can be moved into Qwen3VLM::Open.
bool PrepareQwen3VLArtifactPair(
    const std::string& llm_path,
    const std::string& mmproj_path,
    const ArtifactExpectedIdentity* expected_identity,
    PreparedQwen3VLArtifactPair* prepared,
    std::string* error);

}  // namespace oris_ai
