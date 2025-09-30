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

#include "oris_ai/layer/split.h"

namespace oris_ai {

/**
 * @class SplitGPU
 * @brief Represents a NVIDIA GPU-based Split layer in a neural network.
 * 
 * This class defines a Split layer that operates on the NVIDIA GPU, implementing
 * the transposition of specified dimensions in the input tensor.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class SplitGPU : public Split<T> {
  public:
    /**
     * @brief Constructor to initialize a SplitGPU layer.
     * @param layer_name The name of the layer.
     */
    SplitGPU(const std::string& layer_name)
      : Split<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the SplitGPU class.
     */
    ~SplitGPU() = default;

    /**
     * @brief Performs the forward pass of the Split operation using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Split` 
     * class, providing a NVIDIA GPU-specific implementation for the Split operation.
     */
    void Forward() override;
};

} // namespace oris_ai
