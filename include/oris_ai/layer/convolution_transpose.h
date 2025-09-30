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
 * @class ConvolutionTranspose
 * @brief Represents a transposed convolutional layer in a neural network.
 * 
 * This class defines a generic transposed convolutional layer, including its parameters,
 * and required buffers for operations like col2im.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionTranspose : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionTranspose layer.
     * @param name The name of the layer.
     */
    ConvolutionTranspose(const std::string& layer_name) 
      : HiddenLayerAbstract<T>(layer_name) {}

    /**
     * @brief Destructor for the ConvolutionTranspose class.
     */
    virtual ~ConvolutionTranspose() = default;

    /**
     * @brief Initializes the transposed convolution layer with parameters from a
     * TorchConvTranspose2d object.
     * 
     * This pure virtual function sets up the transposed convolution layer on a specific device,
     * configuring the layer with the provided parameters for optimal execution on the target
     * device (CPU, GPU, etc.).
     * 
     * @param conv_transpose_params The TorchConvTranspose2d object containing parameters.
     */
    virtual void InitConvolutionTranspose(const TorchConvTranspose2d& conv_transpose_params) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the transposed convolution layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the transposed convolution operation.
     */
    virtual void Forward() = 0;

  protected:
    /**
     * @brief Configures the transposed convolutional layer with the given parameters from a
     * TorchConvTranspose2d object.
     * 
     * This function performs the initial setup of transposed convolution layer parameters, such as
     * configuring the weight tensor, bias (if applicable), kernel size, stride, padding,
     * output padding, dilation, and output dimensions. It does not handle device-specific
     * implementation details, allowing InitConvolutionTranspose() to manage these aspects for
     * specific devices like CPU or GPU through virtual function overrides.
     * 
     * @param conv_transpose_params The TorchConvTranspose2d object containing parameters.
     */
    void ConvolutionTransposeSetup(const TorchConvTranspose2d& conv_transpose_params);

    std::unique_ptr<Tensor<T>> weight_;   // A unique pointer to the weight tensor for the layer.
    std::unique_ptr<Tensor<T>> bias_;     // A unique pointer to the bias tensor, if applicable.

    size_t in_channels_, out_channels_;    // The number of input and output channels.
    size_t kernel_h_, kernel_w_;           // The height and width of the convolution kernel.
    size_t stride_h_, stride_w_;           // The stride along the height and width dimensions.
    size_t padding_h_, padding_w_;         // The padding size for the height and width dimensions.
    size_t output_padding_h_, output_padding_w_;  // The output padding size for height and width.
    size_t dilation_h_, dilation_w_;       // The dilation rate for height and width dimensions.
    // size_t groups_;                        // The number of groups for grouped convolution.
    size_t output_height_, output_width_;  // The height and width of the output tensor.

    /**
     * @brief Gets the bias tensor for the transposed convolution layer, if available.
     * 
     * @return A pointer to the bias tensor, or nullptr if no bias is present.
     */
    inline T* GetBias() const { return bias_ ? bias_->GetCPUDataPtr() : nullptr; }

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints the weight tensor for debugging purposes.
     * 
     * This function is available only in debug mode and is used to log the current values
     * in the weight tensor for the layer.
     */
    void PrintWeight();
#endif
};

} // namespace oris_ai 