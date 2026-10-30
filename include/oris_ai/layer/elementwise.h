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

/// Element-wise operation of two inputs: ADD is A + B, SUB is A - B.
enum class ElementWiseOpType {
  ADD,
  SUB
};

/// Element-wise operation layer over two inputs of the same shape.
template <typename T>
class ElementWise : public HiddenLayerAbstract<T> {
public:
  ElementWise(const std::string& layer_name)
    : HiddenLayerAbstract<T>(layer_name) {}

  ~ElementWise() = default;

  void InitElementWise(ElementWiseOpType op_type,
                      bool use_view_input_shape_input0 = false,
                      bool use_view_input_shape_input1 = false
                      , PoolType output_pool_type = PoolType::STATIC
                      );

  virtual void Forward() = 0;

  /// Computes in0 + act(in1). Honored only by the CUDA ADD path.
  inline void SetInput1PreActivation(ActivationType act) {
    input1_pre_activation_ = act;
  }
  inline ActivationType GetInput1PreActivation() const {
    return input1_pre_activation_;
  }

protected:
  ElementWiseOpType op_type_;
  bool use_view_input_shape_[2];
  ActivationType input1_pre_activation_ = ActivationType::NONE;
};

} // namespace oris_ai
