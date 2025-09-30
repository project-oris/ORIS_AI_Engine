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
#include "oris_ai/layer/split.h"
#include "oris_ai/layer/custom/psablock.h"

namespace oris_ai {

/**
 * @class C2PSA
 * @brief Custom C2PSA layer composed of two convolution layers and multiple PSABlocks.
 */
template <typename T>
class C2PSA : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize C2PSA layer with the given name, target device, and PSABlock count.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param expand_ratio The expansion ratio for the layer (default: 0.5f).
     */
    C2PSA(const std::string& layer_name, Device target_device);

    /**
     * @brief Default destructor for the C2PSA class.
     */
    ~C2PSA() = default;

    /**
     * @brief Initializes the C2PSA layer with the given TorchLayer data.
     * @param c2psa_layers Vector of TorchLayer objects for the convolutions and PSABlocks.
     */
    void InitC2PSA(const std::vector<TorchLayer>& c2psa_layers);

    /**
     * @brief Retrieves the output tensor from the second convolutional layer in the C2PSA layer.
     * @return The output tensor from the second convolutional layer.
     */
    inline Tensor<T>* GetOutputTensor() override { return c2psa_cv2_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the C2PSA layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { c2psa_cv2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> c2psa_cv1_;      // First convolution layer
    std::unique_ptr<Split<T>> c2psa_cv1_split_;      // Layer to split before PSABlock
    std::unique_ptr<PSABlock<T>> c2psa_psablock_;    // PSABlock layer
    std::unique_ptr<Concat<T>> c2psa_concat_;        // Concat layer for skip connection
    std::unique_ptr<Convolution<T>> c2psa_cv2_;      // Second convolution layer

    Tensor<T>* c2psa_cv1_split_0_;  // First split output tensor
    Tensor<T>* c2psa_cv1_split_1_;  // Second split output tensor
};

} // namespace oris_ai

