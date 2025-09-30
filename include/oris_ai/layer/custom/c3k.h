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
#include "oris_ai/layer/custom/bottleneck.h"

namespace oris_ai {

// C3k internally uses two bottlenecks.
#define kC3kBottleneckCount 2

/**
 * @class C3k
 * @brief Custom C3k layer composed of two 1x1 convolutions, a sequence of bottlenecks,
 *        and a final 1x1 convolution with concatenation.
 */
template <typename T>
class C3k : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize the C3k layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param shortcut Indicates whether to use shortcut connections inside the bottlenecks.
     */
    C3k(const std::string& layer_name, Device target_device, bool shortcut);

    /**
     * @brief Default destructor for the C3k class.
     */
    ~C3k() = default;

    /**
     * @brief Initializes the C3k layer with the given TorchLayer data.
     * @param c3k_layers Vector of TorchLayer objects for the convolutions and bottlenecks.
     */
    void InitC3k(const std::vector<TorchLayer>& c3k_layers);

    /**
     * @brief Retrieves the output tensor from the last convolutional layer in the C3k layer.
     * @return The output tensor from the last convolutional layer.
     */
    inline Tensor<T>* GetOutputTensor() override { return c3k_cv3_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the C3k layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { c3k_cv3_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> c3k_cv1_;  // First 1x1 convolution layer
    std::vector<std::unique_ptr<Bottleneck<T>>> c3k_bottlenecks_;  // Sequential bottleneck layers
    std::unique_ptr<Convolution<T>> c3k_cv2_;  // Second 1x1 convolution layer
    std::unique_ptr<Concat<T>> c3k_concat_;  // Layer to concatenate before the final convolution
    std::unique_ptr<Convolution<T>> c3k_cv3_;  // Final 1x1 convolution layer

    bool shortcut_;           // Indicates if shortcut (add) operation is enabled
};

} // namespace oris_ai

