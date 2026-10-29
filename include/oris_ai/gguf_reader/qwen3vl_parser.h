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

#include "oris_ai/gguf_reader/gguf_reader.h"
#include "oris_ai/gguf_reader/qwen3vl_layer_info.h"

#include <string>
#include <vector>

namespace oris_ai {

// Parser of the Qwen3-VL text GGUF (ParseLLM) and mmproj GGUF (ParseVision).
class Qwen3VLParser {
 public:
  explicit Qwen3VLParser(GGUFReader* reader);

  bool ParseLLM();

  bool ParseVision();

  static bool ValidateVisionLayout(
      const Qwen3VisionConfig& cfg,
      const std::vector<Qwen3VLLayerInfo>& layers,
      std::string* error);

  const Qwen3VLLLMConfig&             GetLLMConfig() const { return llm_cfg_; }
  const std::vector<Qwen3VLLayerInfo>& GetLLMLayers() const { return llm_layers_; }

  const Qwen3VisionConfig&            GetVisionConfig() const { return vit_cfg_; }
  const std::vector<Qwen3VLLayerInfo>& GetViTLayers() const { return vit_layers_; }

  GGUFReader* reader() const { return reader_; }

  // Points into the reader's mapping. Keep the reader alive.
  const void* GetTensorData(const Qwen3VLLayerInfo& info) const;

  // Searches the LLM layers, then the ViT layers. nullptr when absent.
  const Qwen3VLLayerInfo* GetTensorInfo(const std::string& name) const;

 private:
  bool ExtractLLMMeta();
  bool ExtractVitMeta();
  bool MapLLMTensors();
  bool MapViTTensors();

  GGUFReader* reader_;
  Qwen3VLLLMConfig    llm_cfg_;
  Qwen3VisionConfig   vit_cfg_;
  std::vector<Qwen3VLLayerInfo> llm_layers_;
  std::vector<Qwen3VLLayerInfo> vit_layers_;

  bool        is_qwen3vl_ = true;
  std::string arch_prefix_ = "qwen3vl.";
};

}  // namespace oris_ai
