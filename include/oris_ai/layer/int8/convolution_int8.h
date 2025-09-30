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

#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/int8/padding_int8.h"

namespace oris_ai {

/**
 * @class Convolution
 * @brief Represents a convolutional layer in a neural network.
 * 
 * This class defines a generic convolutional layer, including its parameters,
 * activation functions, and required buffers.
 */
template <>
class Convolution<int8_t> : public HiddenLayerAbstract<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a Convolution layer.
     * @param name The name of the layer.
     */
    Convolution(const std::string& layer_name) 
      : HiddenLayerAbstract<int8_t>(layer_name),
        activation_type_(ActivationType::NONE),
        need_input_requant_(false),
        conv_quant_scale_(1.0f),
        act_input_scale_(1.0f),
        act_output_scale_(1.0f),
        padding_layer_(nullptr) {}

    /**
     * @brief Destructor for the Convolution class.
     */
    ~Convolution();

    // virtual void InitConvolution(const QuantizedTorchConv2d& conv2d_params, const QuantizedTorchActivation& act, const float backbone_quant_scale);

    // virtual void InitConvolution(const QuantizedTorchConv2d& conv2d_params, const QuantizedTorchActivation& act);

    // virtual void InitConvolution(const QuantizedTorchConv2d& conv2d_params);

    virtual void InitConvolution(const QuantizedTorchConv2d& conv2d_params, const QuantizedTorchActivation& act, const float backbone_quant_scale) = 0;

    virtual void InitConvolution(const QuantizedTorchConv2d& conv2d_params, const QuantizedTorchActivation& act) = 0;

    virtual void InitConvolution(const QuantizedTorchConv2d& conv2d_params) = 0;

    /**
     * @brief Gets the weight tensor for the convolution layer.
     * 
     * @return A pointer to the weight tensor.
     */
    inline Tensor<int8_t>* GetWeight() { return weight_.get(); }

    /**
     * @brief Gets the bias tensor for the convolution layer, if available.
     * 
     * @return A pointer to the bias tensor, or nullptr if no bias is present.
     */
    inline Tensor<int32_t>* GetBias() {
      return bias_ ? bias_.get() : nullptr;
    }

    virtual void Forward() = 0;

  protected:
    void CheckPadding();
    void ConvolutionSetup(const QuantizedTorchConv2d& conv2d_params);
    void SetActivation(const QuantizedTorchActivation& act);

    ActivationType activation_type_;    // The type of activation function used in the layer.
    std::unique_ptr<Tensor<int8_t>> weight_; // A unique pointer to the weight tensor for the layer.
    std::unique_ptr<Tensor<int32_t>> bias_;   // A unique pointer to the bias tensor, if applicable.
    std::unique_ptr<Tensor<int32_t>> output_int32_;
    // std::unique_ptr<Tensor<float>> output_fp32_;

    size_t out_channels_, in_channels_;   // The number of output and input channels for the layer.
    size_t kernel_h_, kernel_w_;          // The height and width of the convolution kernel.
    size_t stride_h_, stride_w_;          // The stride along the height and width dimensions.
    size_t padding_h_, padding_w_;        // The padding size for the height and width dimensions.
    size_t output_height_, output_width_; // The height and width of the output tensor.

    // conv int8 variables
    bool need_input_requant_;
    float conv_quant_scale_; // input_scale x weight_scale / output_scale
    // float* quant_scale_cuda_; // CUDA memory for quantization scales

    // activation int8 variables
    float act_input_scale_;
    float act_output_scale_;

    float backbone_quant_scale_ = 1.0f; // Quantization scale for the backbone part of the model

    // padding layer
    std::unique_ptr<PaddingINT8> padding_layer_;
};

} // namespace oris_ai
