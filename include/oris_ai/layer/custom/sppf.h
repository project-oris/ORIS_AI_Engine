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

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"
// #include "oris_ai/layer/int8/convolution_int8.h"
#include "oris_ai/layer/maxpooling.h"

namespace oris_ai {

/**
 * @class SPPF
 * @brief Custom SPPF layer for YOLOv8n, consisting of convolution and max-pooling layers.
 */
template <typename T>
class SPPF : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize SPPF layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    SPPF(const std::string& layer_name, Device target_device);

    /**
     * @brief Destructor for the SPPF class.
     */
    ~SPPF() = default;

    /**
     * @brief Initializes the SPPF layer with the given TorchLayer data.
     * @param sppf_layers Vector of TorchLayer objects for the convolution and max-pooling layers.
     */
    void InitSPPF(const std::vector<TorchLayer>& sppf_layers);

    /**
     * @brief Retrieves the output tensor from the last convolutional layer in the SPPF layer.
     * @return The output tensor.
     */
    inline Tensor<T>* GetOutputTensor() override { return sppf_cv2_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the SPPF layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { sppf_cv2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> sppf_cv1_;  // First convolution layer (cv1)
    std::unique_ptr<MaxPooling<T>> sppf_maxpool1_;  // First max-pooling layer
    std::unique_ptr<MaxPooling<T>> sppf_maxpool2_;  // Second max-pooling layer
    std::unique_ptr<MaxPooling<T>> sppf_maxpool3_;  // Third max-pooling layer
    std::unique_ptr<Concat<T>> sppf_concat_;    // Layer to concatenate before cv2
    std::unique_ptr<Convolution<T>> sppf_cv2_;  // Second convolution layer (cv2)
};

}  // namespace oris_ai
