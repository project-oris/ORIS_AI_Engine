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

namespace oris_ai {

/// Element-wise activation layer.
template <typename T>
class Activation : public HiddenLayerAbstract<T> {
public:
  Activation(const std::string& layer_name)
    : HiddenLayerAbstract<T>(layer_name),
      activation_type_(ActivationType::NONE) {}

  ~Activation() = default;

  /// The output has the shape of the input.
  virtual void InitActivation(ActivationType activation_type) = 0;

  inline ActivationType GetActivationType() const { return activation_type_; }

  virtual void Forward() = 0;

protected:
  void ActivationSetup(ActivationType activation_type);

  ActivationType activation_type_;
  PoolType pool_type_ = PoolType::ACTIVATION;
};

} // namespace oris_ai
