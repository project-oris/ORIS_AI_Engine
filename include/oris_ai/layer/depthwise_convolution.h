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

#include "oris_ai/layer/base_convolution.h"

namespace oris_ai {

/**
 * @class DepthwiseConvolution
 * @brief Represents a depthwise convolutional layer in a neural network.
 * 
 * This class defines a generic depthwise convolutional layer, including its parameters,
 * activation functions, and required buffers for operations like im2col.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DepthwiseConvolution : public BaseConvolution<T> {
  public:
    /**
     * @brief Constructor to initialize a DepthwiseConvolution layer.
     * @param name The name of the layer.
     */
    DepthwiseConvolution(const std::string& layer_name)
      : BaseConvolution<T>(layer_name),
        use_view_input_shape_(false) {}

    /**
     * @brief Destructor for the DepthwiseConvolution class.
     */
    ~DepthwiseConvolution() = default;

    /**
     * @brief Initializes the depthwise convolution layer with both convolution and 
     * activation parameters.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     * @param act The TorchActivation object containing activation function parameters.
     * @param use_view_input_shape Whether to use the view input shape.
     */
    virtual void InitDepthwiseConvolution(const TorchConv2d& conv2d_params,
                                          const TorchActivation& act,
                                          bool use_view_input_shape = false) = 0;


    /**
     * @brief Initializes the depthwise convolution layer without activation.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     * @param use_view_input_shape Whether to use the view input shape.
     */
    virtual void InitDepthwiseConvolution(const TorchConv2d& conv2d_params,
                                          bool use_view_input_shape = false) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass.
     */
    virtual void Forward() = 0;

#ifdef USE_DEBUG_MODE
    void PrintWeight();
#endif

  protected:
    /**
     * @brief Configures the depthwise convolutional layer with the given parameters.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void DepthwiseConvolutionSetup(const TorchConv2d& conv2d_params);

    bool use_view_input_shape_;
};

} // namespace oris_ai 
