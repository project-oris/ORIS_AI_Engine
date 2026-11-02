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

#include "oris_ai/tensor/fp16/tensor_fp16.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace oris_ai {

// Input and output scratch shared by the F16 Linear layers of one stream.
template <typename T>
class F16Scratch {
 public:
  // Sizes are in elements.
  F16Scratch(std::size_t max_input_elems, std::size_t max_output_elems,
             PoolType pool_type = PoolType::STATIC);
  ~F16Scratch() = default;

  // Layers keep non-owning pointers to the scratch.
  F16Scratch(const F16Scratch&) = delete;
  F16Scratch& operator=(const F16Scratch&) = delete;
  F16Scratch(F16Scratch&&) = delete;
  F16Scratch& operator=(F16Scratch&&) = delete;

  Tensor<half>* input() { return input_.get(); }
  // nullptr in this release.
  Tensor<half>* output() { return output_.get(); }

  std::size_t max_input_elems() const { return max_input_elems_; }
  std::size_t max_output_elems() const { return max_output_elems_; }

 private:
  std::size_t max_input_elems_;
  std::size_t max_output_elems_;
  std::unique_ptr<Tensor<half>> input_;
  std::unique_ptr<Tensor<half>> output_;
};

}  // namespace oris_ai
