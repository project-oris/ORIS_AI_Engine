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

#include "oris_ai/layer/layer.h"
#include "oris_ai/model/language/decode_token_epilogue.h"
#include "oris_ai/tensor/tensor.h"

#include <memory>
#include <vector>

namespace oris_ai {

// Base class of the language models.
template <typename T>
class LanguageModel {
public:
  LanguageModel(Device target_device);
  virtual ~LanguageModel();

  virtual void Forward() = 0;

  // Call before CreateInputTensor(). An empty id uses the default pool.
  void SetModelScope(const std::string& scope_id);

  const std::string& model_scope_id() const { return model_scope_id_; }

  // nullptr removes it. It selects a token only inside
  // ArmDecodeTokenEpilogue(true); after a removal returns, the caller may release
  // its device resources.
  void SetDecodeTokenEpilogue(DecodeTokenEpilogue* epilogue);
  DecodeTokenEpilogue* decode_token_epilogue() const {
    return decode_epilogue_;
  }
  void ArmDecodeTokenEpilogue(bool armed) { decode_epilogue_armed_ = armed; }
  bool decode_token_epilogue_armed() const { return decode_epilogue_armed_; }

protected:
  // The stream the model's Forward() enqueues on.
  virtual cudaStream_t DecodeProductionStream() const;
  virtual void DropDecodeSelectGraphs() {}

  std::vector<std::unique_ptr<HiddenLayerAbstract<T>>> layers_;

  Device target_device_;

  std::string model_scope_id_;

private:
  DecodeTokenEpilogue* decode_epilogue_ = nullptr;
  bool decode_epilogue_armed_ = false;
};

// Arms the decode token epilogue while the scope lives.
template <typename T>
class DecodeTokenEpilogueScope {
public:
  explicit DecodeTokenEpilogueScope(LanguageModel<T>* model) : model_(model) {
    if (model_ != nullptr) model_->ArmDecodeTokenEpilogue(true);
  }
  ~DecodeTokenEpilogueScope() {
    if (model_ != nullptr) model_->ArmDecodeTokenEpilogue(false);
  }
  DecodeTokenEpilogueScope(const DecodeTokenEpilogueScope&) = delete;
  DecodeTokenEpilogueScope& operator=(const DecodeTokenEpilogueScope&) = delete;

private:
  LanguageModel<T>* model_;
};

} // namespace oris_ai
