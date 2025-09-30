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

#include "oris_ai/layer/custom/inverted_residual.h"

namespace oris_ai {

/**
 * @class MobileNetBlock
 * @brief Custom MobileNetBlock layer for MobileNet v2, consisting of multiple InvertedResidual layers.
 */
template <typename T>
class MobileNetBlock : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize MobileNetBlock layer with the given parameters.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param out_channels Number of output channels.
     * @param expansion_factor Expansion factor (t) for the expand convolution.
     * @param repeat_count Number of times to repeat the InvertedResidual layer.
     * @param stride Stride value for the first InvertedResidual layer.
     */
    MobileNetBlock(const std::string& layer_name, Device target_device,
                  size_t out_channels, size_t expansion_factor,
                  size_t repeat_count, size_t stride);

    /**
     * @brief Default destructor for the MobileNetBlock class.
     */
    ~MobileNetBlock() = default;

    /**
     * @brief Initializes the MobileNetBlock layer with the given TorchLayer data.
     * @param layers Vector of TorchLayer objects for the InvertedResidual layers.
     */
    void InitMobileNetBlock(const std::vector<TorchLayer>& layers);

    /**
     * @brief Retrieves the output tensor from the last InvertedResidual layer in the MobileNetBlock.
     * @return The output tensor from the last InvertedResidual layer.
     */
    inline Tensor<T>* GetOutputTensor() override { 
      return inverted_residuals_.back()->GetOutputTensor(); 
    }

    /**
     * @brief Perform the forward pass for the MobileNetBlock layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { inverted_residuals_.back()->PrintOutput(); }
#endif

  private:
    std::vector<std::unique_ptr<InvertedResidual<T>>> inverted_residuals_;  // Vector of InvertedResidual layers
    Device target_device_;  // Target device for the MobileNetBlock
    size_t out_channels_;  // Number of output channels
    size_t expansion_factor_;  // Expansion factor for expand convolution
    size_t repeat_count_;  // Number of times to repeat the InvertedResidual layer
    size_t stride_;  // Stride value for the first InvertedResidual layer
};

} // namespace oris_ai 