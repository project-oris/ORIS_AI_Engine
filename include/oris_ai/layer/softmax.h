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

/**
 * @class Softmax
 * @brief Represents a Softmax layer operating along the channel dimension.
 *
 * This layer normalizes the input tensor using the softmax function. The output
 * tensor has the same shape as the input tensor.
 *
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Softmax : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a Softmax layer.
     * @param layer_name The name of the layer.
     */
    explicit Softmax(const std::string& layer_name)
      : HiddenLayerAbstract<T>(layer_name),
        softmax_dim_(1),
        use_view_input_shape_(false) {}

    /**
     * @brief Destructor for the Softmax class.
     */
    virtual ~Softmax() = default;

    /**
     * @brief Initializes the Softmax layer on the specific device.
     *
     * This virtual function finalizes the Softmax layer setup by preparing the output
     * tensor. Device-specific initializations are handled in the derived classes
     * through virtual function overrides.
     *
     * @param softmax_dim The dimension along which softmax will be applied.
     *                    Supported values are 1 (channel) and 3 (width).
     * @param use_view_input_shape Flag indicating whether to use the input tensor's
     *                             view shape for initialization.
     */
    virtual void InitSoftmax(const size_t softmax_dim, bool use_view_input_shape = false) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the Softmax layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the softmax operation.
     */
    virtual void Forward() = 0;

  protected:
    /**
     * @brief Configures the output tensor for the Softmax layer.
     * 
     * This function performs the initial setup of the Softmax layer by preparing
     * the output tensor configuration. It does not handle device-specific
     * implementation details, allowing InitSoftmax() to manage these aspects
     * for specific devices like CPU or GPU through virtual function overrides.
     */
    void SoftmaxSetup();

    size_t softmax_dim_;           // The dimension along which softmax is applied
    bool use_view_input_shape_;    // Flag to indicate whether view shape should be used
};

} // namespace oris_ai
