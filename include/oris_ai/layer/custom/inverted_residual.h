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
#include "oris_ai/layer/depthwise_convolution.h"
#include "oris_ai/layer/elementwise.h"

namespace oris_ai {

/**
 * @class InvertedResidual
 * @brief Custom InvertedResidual layer for MobileNet v2, consisting of expand, depthwise, and pointwise convolutions.
 */
template <typename T>
class InvertedResidual : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize InvertedResidual layer with the given name, target device, and parameters.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param use_expand Whether to use expand convolution.
     * @param use_residual Whether to use residual connection.
     */
    InvertedResidual(const std::string& layer_name, Device target_device, 
                    bool use_expand, bool use_residual);

    /**
     * @brief Default destructor for the InvertedResidual class.
     */
    ~InvertedResidual() = default;

    /**
     * @brief Initializes the InvertedResidual layer with the given TorchLayer data.
     * @param layers Vector of TorchLayer objects for the convolutions.
     */
    void InitInvertedResidual(const std::vector<TorchLayer>& layers);

    /**
     * @brief Retrieves the output tensor from the last convolutional layer in the InvertedResidual layer.
     * @return The output tensor from the last convolutional layer.
     */
    inline Tensor<T>* GetOutputTensor() override { 
      return use_residual_ ? add_->GetOutputTensor() : pointwise_->GetOutputTensor(); 
    }

    /**
     * @brief Perform the forward pass for the InvertedResidual layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { 
      if (use_residual_) {
        add_->PrintOutput();
      } else {
        pointwise_->PrintOutput();
      }
    }
#endif

  private:
    /**
     * @brief Determines whether to use residual connection based on input and output channels.
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @return true if residual connection should be used, false otherwise.
     */
    // static bool ShouldUseResidual(size_t in_channels, size_t out_channels);

    std::unique_ptr<Convolution<T>> expand_;  // Expand convolution layer
    std::unique_ptr<DepthwiseConvolution<T>> depthwise_;  // Depthwise convolution layer
    std::unique_ptr<Convolution<T>> pointwise_;  // Pointwise convolution layer
    std::unique_ptr<ElementWise<T>> add_;  // Element-wise addition layer

    // size_t in_channels_;  // Number of input channels
    // size_t out_channels_;  // Number of output channels
    bool use_expand_;  // Whether to use expand convolution
    bool use_residual_;  // Whether to use residual connection
};

} // namespace oris_ai 