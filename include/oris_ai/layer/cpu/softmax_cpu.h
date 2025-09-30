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

#include "oris_ai/layer/softmax.h"

namespace oris_ai {

/**
 * @class SoftmaxCPU
 * @brief Represents a CPU-based Softmax layer in a neural network.
 * 
 * This class inherits from Softmax and implements the forward pass
 * specifically for CPU execution using Eigen for efficient computation.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class SoftmaxCPU : public Softmax<T> {
  public:
    /**
     * @brief Constructor to initialize a SoftmaxCPU layer.
     * @param layer_name The name of the layer.
     */
    explicit SoftmaxCPU(const std::string& layer_name)
      : Softmax<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the SoftmaxCPU class.
     */
    ~SoftmaxCPU() = default;

    /**
     * @brief Overrides the virtual InitSoftmax function for CPU-specific initialization.
     * 
     * This function implements the virtual InitSoftmax method defined in the Softmax 
     * base class, configuring the Softmax layer for efficient execution on the CPU.
     */
    void InitSoftmax(const size_t softmax_dim, bool use_view_input_shape) override;

    /**
     * @brief Performs the forward pass of the Softmax layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Softmax` 
     * class, providing a CPU-specific implementation for the Softmax operation.
     */
    void Forward() override;
};

} // namespace oris_ai
