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

#include "oris_ai/layer/depthwise_convolution.h"
#include "oris_ai/accelerator/cuda/cuda_functions.h"

namespace oris_ai {

/**
 * @class DepthwiseConvolutionGPU
 * @brief A class that implements depthwise convolution operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing depthwise
 * convolution operations, inheriting from the base `DepthwiseConvolution` class. 
 * It overrides the `Forward` method to execute the forward pass of the depthwise
 * convolution layer using NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DepthwiseConvolutionGPU : public DepthwiseConvolution<T> {
  public:
    /**
     * @brief Constructor to initialize a DepthwiseConvolutionGPU layer.
     * @param layer_name The name of the layer.
     */
    DepthwiseConvolutionGPU(const std::string& layer_name)
      : DepthwiseConvolution<T>(layer_name) {}
    
    /**
     * @brief Destructor for the DepthwiseConvolutionGPU class.
     */
    ~DepthwiseConvolutionGPU() = default;

    /**
     * @brief Overrides the virtual InitDepthwiseConvolution function for NVIDIA GPU-specific
     * initialization with activation.
     * 
     * This function implements the virtual InitDepthwiseConvolution method defined in the
     * DepthwiseConvolution base class, configuring the depthwise convolution layer with
     * the provided parameters and activation function for efficient execution on the NVIDIA GPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     * @param act The TorchActivation object containing activation function parameters.
     * @param use_view_input_shape Whether to use the view input shape.
     */
    void InitDepthwiseConvolution(const TorchConv2d& conv2d_params, 
                                  const TorchActivation& act,
                                  bool use_view_input_shape = false) override;

    /**
     * @brief Overrides the virtual InitDepthwiseConvolution function for NVIDIA GPU-specific
     * initialization without activation.
     * 
     * This function implements the virtual InitDepthwiseConvolution method defined in the
     * DepthwiseConvolution base class, configuring the depthwise convolution layer with
     * the provided parameters for efficient execution on the NVIDIA GPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     * @param use_view_input_shape Whether to use the view input shape.
     */
    void InitDepthwiseConvolution(const TorchConv2d& conv2d_params,
                                  bool use_view_input_shape = false) override;

    /**
     * @brief Performs the forward pass of the DepthwiseConvolution layer using NVIDIA GPU.
     * 
     * This function overrides the pure virtual `Forward` method from the base 
     * `DepthwiseConvolution` class, providing a NVIDIA GPU-specific implementation 
     * for the depthwise convolution operation.
     */
    void Forward() override;
};

} // namespace oris_ai 