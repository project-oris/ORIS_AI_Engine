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

// Define the ElementWiseOpType enum class
enum class ElementWiseOpType {
  ADD,          // C = A + B
  // ACCUM_A_TO_C, // C = C + A (currently, this is not supported)
  SUB           // C = A - B
};

/**
 * @class ElementWise
 * @brief Represents an element-wise operation layer in a neural network.
 * 
 * This class defines a generic element-wise operation layer that performs operations
 * like sum, product, etc. on multiple input tensors of the same shape.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ElementWise : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize an ElementWise layer.
     * @param name The name of the layer.
     */
    ElementWise(const std::string& layer_name)
      : HiddenLayerAbstract<T>(layer_name) {}

    /**
     * @brief Destructor for the ElementWise class.
     */
    ~ElementWise() = default;

    /**
     * @brief Initializes the element-wise layer with the specified operation type.
     *
     * This function sets up the element-wise layer with the given operation type
     * and validates the input tensors. Each input tensor can optionally use its
     * view shape during initialization.
     *
     * @param op_type The type of element-wise operation to perform.
     * @param use_view_input_shape_input0 Whether to use the view shape of the first
     *        input tensor.
     * @param use_view_input_shape_input1 Whether to use the view shape of the second
     *        input tensor.
     */
    void InitElementWise(ElementWiseOpType op_type,
                        bool use_view_input_shape_input0 = false,
                        bool use_view_input_shape_input1 = false);

    /**
     * @brief Pure virtual function to perform the forward pass of the element-wise layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU).
     */
    virtual void Forward() = 0;

  protected:
    ElementWiseOpType op_type_;  // The type of element-wise operation to perform
    bool use_view_input_shape_[2]; // Flags for using view shapes of input tensors
};

} // namespace oris_ai