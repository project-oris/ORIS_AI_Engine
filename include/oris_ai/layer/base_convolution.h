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

#ifdef USE_DEBUG_MODE
#include <glog/logging.h>
#endif

namespace oris_ai {

/**
 * @class BaseConvolution
 * @brief Base class for convolutional operations in a neural network.
 *
 * This class contains member variables and utility functions common to both
 * standard and depthwise convolution layers.
 *
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class BaseConvolution : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a BaseConvolution layer.
     * @param layer_name The name of the layer.
     */
    explicit BaseConvolution(const std::string& layer_name)
      : HiddenLayerAbstract<T>(layer_name),
        activation_type_(ActivationType::NONE),
        in_channels_(0),
        kernel_h_(0), kernel_w_(0),
        stride_h_(0), stride_w_(0),
        padding_h_(0), padding_w_(0),
        output_height_(0), output_width_(0) {}

    /**
     * @brief Virtual destructor for the BaseConvolution class.
     */
    virtual ~BaseConvolution() = default;

    /**
     * @brief Gets the weight tensor for the convolution layer.
     * @return A pointer to the weight tensor.
     */
    inline Tensor<T>* GetWeight() { return weight_.get(); }

    /**
     * @brief Gets the bias tensor for the convolution layer, if available.
     * @return A pointer to the bias tensor, or nullptr if no bias is present.
     */
    inline Tensor<T>* GetBias() { return bias_ ? bias_.get() : nullptr; }

    /**
     * @brief Pure virtual function to perform the forward pass of the convolution layer.
     */
    virtual void Forward() = 0;

  protected:
    /**
     * @brief Sets the activation function for the convolution layer.
     * @param act The TorchActivation object containing activation parameters.
     */
    void SetActivation(const TorchActivation& act);

    ActivationType activation_type_;      // Activation type used in the layer.
    std::unique_ptr<Tensor<T>> weight_;   // Weight tensor for the convolution.
    std::unique_ptr<Tensor<T>> bias_;     // Bias tensor, if applicable.

    size_t in_channels_;                  // Number of input channels.
    size_t kernel_h_, kernel_w_;          // Convolution kernel dimensions.
    size_t stride_h_, stride_w_;          // Stride along height and width.
    size_t padding_h_, padding_w_;        // Padding along height and width.
    size_t output_height_, output_width_; // Output tensor dimensions.
};

} // namespace oris_ai

