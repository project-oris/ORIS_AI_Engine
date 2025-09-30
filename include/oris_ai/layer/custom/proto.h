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
#include "oris_ai/layer/convolution_transpose.h"

namespace oris_ai {

/**
 * @class Proto
 * @brief Represents a custom Proto layer with sequential convolution operations.
 * 
 * This class implements a sequence of convolution and convolution transpose operations
 * in the following order: Convolution -> ConvolutionTranspose -> Convolution -> Convolution.
 * Each Convolution is followed by an activation function.
 */
template <typename T>
class Proto : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a Proto layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    Proto(const std::string& layer_name, Device target_device);

    /**
     * @brief Destructor for the Proto class.
     */
    ~Proto() = default;

    /**
     * @brief Initializes the Proto layer with the given TorchLayer data.
     * @param proto_layers Vector of TorchLayer objects containing layer parameters.
     */
    void InitProto(const std::vector<TorchLayer>& proto_layers);

    /**
     * @brief Retrieves the output tensor for the Proto layer.
     * @return The output tensor.
     */
    inline Tensor<T>* GetOutputTensor() override { return cv3_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the Proto layer.
     */
    void Forward() override;

  private:
    std::unique_ptr<Convolution<T>> cv1_;               // First convolution layer
    std::unique_ptr<ConvolutionTranspose<T>> upsample_; // Convolution transpose layer
    std::unique_ptr<Convolution<T>> cv2_;               // Second convolution layer
    std::unique_ptr<Convolution<T>> cv3_;               // Third convolution layer
};

} // namespace oris_ai 