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

#include "oris_ai/layer/transpose.h"

namespace oris_ai {

/**
 * @class TransposeCPU
 * @brief Represents a CPU-based transpose layer in a neural network.
 * 
 * This class defines a transpose layer that operates on the CPU, implementing
 * the transposition of specified dimensions in the input tensor.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class TransposeCPU : public Transpose<T> {
  public:
    /**
     * @brief Default constructor to initialize a TransposeCPU layer.
     * @param layer_name The name of the layer.
     */
    TransposeCPU(const std::string& layer_name)
      : Transpose<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the TransposeCPU class.
     */
    ~TransposeCPU() = default;

    /**
     * @brief Overrides the virtual InitTranspose function for CPU-specific initialization.
     * 
     * This function implements the virtual InitTranspose method defined in the Transpose 
     * base class, configuring the transpose layer with the provided parameters for
     * efficient execution on the CPU.
     * 
     * @param dim1 The first dimension to swap.
     * @param dim2 The second dimension to swap.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input
     * tensor.
     */
    void InitTranspose(size_t dim1, size_t dim2, bool use_view_input_shape = false) override;

    /**
     * @brief Performs the forward pass of the transpose operation using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Transpose` 
     * class, providing a CPU-specific implementation for the transpose operation.
     * It rearranges data from the input tensor to the output tensor based on the precomputed
     * transposed indices, which are set in the `InitTranspose` function.
     */
    void Forward() override;
};

} // namespace oris_ai
