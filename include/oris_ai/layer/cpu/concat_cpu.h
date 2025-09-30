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

#include "oris_ai/layer/concat.h"

namespace oris_ai {

/**
 * @class ConcatCPU
 * @brief Represents a CPU-based concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer that operates on CPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConcatCPU : public Concat<T> {
  public:
    /**
     * @brief Constructor to initialize a ConcatCPU layer.
     * @param name The name of the layer.
     */
    ConcatCPU(const std::string& layer_name)
      : Concat<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the ConcatCPU class.
     */
    ~ConcatCPU() = default;

    /**
     * @brief Performs the forward pass of the Concat layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Concat` 
     * class, providing a CPU-specific implementation for the Concat operation.
     */
    void Forward() override;
};

} // namespace oris_ai
