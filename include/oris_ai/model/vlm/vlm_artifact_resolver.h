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

#include <optional>
#include <string>
#include <vector>

#include "oris_ai/chat/reasoning_mode.h"

namespace oris_ai {

// Model family.
enum class VlmFamily {
  QWEN3_VL,
};

// Qwen3-VL size.
enum class Qwen3VLSize {
  B2,
  B8,
};

// Language model quantization.
enum class LlmQuant {
  Q8_0,
};

// How a model file pair was selected.
enum class ArtifactSelectionSource {
  CATALOG,
  EXPLICIT_PAIR,
};

// Semantic selector of a model file pair.
struct ArtifactKey {
  VlmFamily family = VlmFamily::QWEN3_VL;
  std::optional<Qwen3VLSize> qwen_size = Qwen3VLSize::B2;
  ReasoningMode mode = ReasoningMode::INSTRUCT;
  LlmQuant quant = LlmQuant::Q8_0;
};

// Metadata a pair must carry.
struct ArtifactExpectedIdentity {
  VlmFamily family = VlmFamily::QWEN3_VL;
  std::optional<Qwen3VLSize> qwen_size;
  std::optional<ReasoningMode> mode;
  std::optional<LlmQuant> quant;
  std::optional<std::string> llm_architecture;
  std::optional<std::string> llm_size_label;
  std::optional<std::string> llm_finetune;
  std::optional<std::string> projector_architecture;
  std::optional<std::string> projector_general_type;
  std::optional<std::string> projector_finetune;
};

// Metadata read from a pair.
struct ArtifactObservedIdentity {
  VlmFamily family = VlmFamily::QWEN3_VL;
  std::optional<Qwen3VLSize> qwen_size;
  std::optional<ReasoningMode> mode;
  std::optional<LlmQuant> quant;
  std::string llm_architecture;
  std::string llm_size_label;
  std::string llm_finetune;
  std::string projector_architecture;
  std::string projector_general_type;
  std::string projector_finetune;
};

// One catalog entry.
struct VlmArtifactCatalogEntry {
  ArtifactKey key;
  std::string llm_filename;
  std::string mmproj_filename;
  ArtifactExpectedIdentity expected_identity;
};

// Input of ResolveVlmArtifacts.
struct VlmArtifactResolveRequest {
  ArtifactKey key;
  std::string artifact_root;
  std::string explicit_llm_path;
  std::string explicit_mmproj_path;
  bool qwen_size_was_set = false;
  bool mode_was_set = false;
  bool quant_was_set = false;
};

// Output of ResolveVlmArtifacts.
struct ResolvedArtifactPair {
  ArtifactKey key;
  std::string artifact_root;
  std::string llm_path;
  std::string mmproj_path;
  ArtifactExpectedIdentity expected_identity;
  ArtifactSelectionSource selection_source = ArtifactSelectionSource::CATALOG;
};

const std::vector<VlmArtifactCatalogEntry>& GetVlmArtifactCatalog();
const char* GetDefaultVlmArtifactRoot();

std::string ResolveVlmArtifactRoot(const std::string& cli_root,
                                   const char* environment_root);
std::string ResolveVlmArtifactRoot(const std::string& cli_root);

bool ResolveVlmArtifacts(const VlmArtifactResolveRequest& request,
                         ResolvedArtifactPair* resolved,
                         std::string* error);

ResolvedArtifactPair ResolveCatalogVlmArtifacts(
    const ArtifactKey& key, const std::string& artifact_root = "");

bool ParseQwen3VLSize(const std::string& text, Qwen3VLSize* value);
bool ParseLlmQuant(const std::string& text, LlmQuant* value);

const char* ToString(VlmFamily value);
const char* ToString(Qwen3VLSize value);
const char* ToString(LlmQuant value);
const char* ToString(ArtifactSelectionSource value);

}  // namespace oris_ai
