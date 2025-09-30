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

#include "oris_ai/layer/convolution_transpose.h"

namespace oris_ai {

/**
 * @class ConvolutionTransposeCPU
 * @brief A class that implements transposed convolution operations on a CPU.
 * 
 * This class provides a CPU-specific implementation for performing transposed convolution 
 * operations, inheriting from the base `ConvolutionTranspose` class. It overrides the 
 * `Forward` method to perform the forward pass of the transposed convolution layer 
 * using CPU resources.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionTransposeCPU : public ConvolutionTranspose<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionTransposeCPU layer.
     * @param name The name of the layer.
     */
    ConvolutionTransposeCPU(const std::string& layer_name)
      : ConvolutionTranspose<T>(layer_name) {}

    /**
     * @brief Destructor for the ConvolutionTransposeCPU class.
     */
    ~ConvolutionTransposeCPU() = default;

    /**
     * @brief Initializes the transposed convolution layer with the given parameters.
     * 
     * This function implements the initialization for CPU-specific transposed convolution,
     * configuring the layer with the provided parameters for efficient execution on the CPU.
     * 
     * @param conv_transpose_params The TorchConvTranspose2d object containing parameters.
     */
    void InitConvolutionTranspose(const TorchConvTranspose2d& conv_transpose_params) override;

    /**
     * @brief Performs the forward pass of the transposed convolution layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `ConvolutionTranspose` 
     * class, providing a CPU-specific implementation for the transposed convolution operation.
     */
    void Forward() override;
};

} // namespace oris_ai 