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

#include "oris_ai/layer/custom/bottleneck.h"

namespace oris_ai {

/**
 * @class Bottleneck
 * @brief Represents a bottleneck layer in YOLO model.
 * 
 * This class encapsulates two convolutional layers, applies element-wise addition
 * if the input and output channels are the same, and manages forward pass and tensor 
 * operations.
 */
template <>
class Bottleneck<int8_t> : public HiddenLayerAbstract<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a Bottleneck layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU).
     * @param add Flag to determine if element-wise addition is performed.
     */
    Bottleneck(const std::string& layer_name, Device target_device, bool add);

    /**
     * @brief Destructor for the Bottleneck class.
     */
    ~Bottleneck() = default;

    /**
     * @brief Initializes the Bottleneck layer with two convolutional layers.
     * 
     * @param layer_conv1 The TorchLayer object for the first convolution layer.
     * @param layer_act1 The TorchLayer object containing the activation type.
     * @param layer_conv2 The TorchLayer object for the second convolution layer.
     * @param layer_act2 The TorchLayer object containing the activation type.
     */
    void InitBottleneck(const TorchLayer& layer_conv1, const TorchLayer& layer_act1, 
                        const TorchLayer& layer_conv2, const TorchLayer& layer_act2);

    /**
     * @brief Retrieves the output tensor from the second convolutional layer in the bottleneck.
     * @return The output tensor from the second convolutional layer.
     */
    inline Tensor<int8_t>* GetOutputTensor() override { 
      return bottleneck_cv2_->GetOutputTensor(); 
    }

    /**
     * @brief Perform the forward pass of the Bottleneck layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints output tensor in NCHW format.
     */ 
    inline void PrintOutput() override {
      bottleneck_cv2_->PrintOutput();
    }
#endif

  private:
    std::unique_ptr<Convolution<int8_t>> bottleneck_cv1_;  // First convolutional layer in bottleneck
    std::unique_ptr<Convolution<int8_t>> bottleneck_cv2_;  // Second convolutional layer in bottleneck
    bool add_;  // Flag to determine if element-wise addition is performed
};

} // namespace oris_ai
