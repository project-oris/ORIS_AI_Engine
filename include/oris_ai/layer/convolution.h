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
 * @class Convolution
 * @brief Represents a convolutional layer in a neural network.
 * 
 * This class defines a generic convolutional layer, including its parameters,
 * activation functions, and required buffers for operations like im2col.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Convolution : public BaseConvolution<T> {
  public:
    /**
     * @brief Constructor to initialize a Convolution layer.
     * @param name The name of the layer.
     */
    Convolution(const std::string& layer_name)
      : BaseConvolution<T>(layer_name) {}

    /**
     * @brief Destructor for the Convolution class.
     */
    ~Convolution() = default;

    /**
     * @brief Initializes the convolution layer with both convolution and activation
     * parameters.
     * 
     * This pure virtual function sets up the convolution layer on a specific device,
     * combining the initialization of convolution parameters from a TorchConv2d object and
     * the activation function from a TorchActivation object. It finalizes the setup for the
     * layer, ensuring that both the convolution operation and its post-processing activation
     * function are configured for optimal execution on the target device (CPU, GPU, etc.).
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     * @param act The TorchActivation object containing the activation type and its parameters.
     */
    virtual void InitConvolution(const TorchConv2d& conv2d_params, const TorchActivation& act) = 0;

    /**
     * @brief Initializes the convolution layer on the specific device with parameters from a
     * TorchConv2d object.
     * 
     * This virtual function finalizes the convolution layer setup by applying device-specific 
     * configurations for the layer on CPU, GPU, or other hardware. It uses the initial
     * parameters prepared by ConvolutionSetup() and extends the setup to include any
     * device-specific optimizations or settings.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    virtual void InitConvolution(const TorchConv2d& conv2d_params) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the convolution layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the convoltion operation.
     */
    virtual void Forward() = 0;

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints the weight tensor for debugging purposes.
     * 
     * This function is available only in debug mode and is used to log the current values
     * in the weight tensor for the layer.
     */
    void PrintWeight();
#endif

  protected:
    /**
     * @brief Configures the convolutional layer with the given parameters from a TorchConv2d
     * object.
     * 
     * This function performs the initial setup of convolution layer parameters, such as
     * configuring the weight tensor, bias (if applicable), kernel size, stride, padding, and
     * output dimensions. It does not handle device-specific implementation details, allowing
     * InitConvolution() to  manage these aspects for specific devices like CPU or GPU through
     * virtual function overrides.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void ConvolutionSetup(const TorchConv2d& conv2d_params);

    size_t out_channels_;   // The number of output channels for the layer.
};

} // namespace oris_ai
