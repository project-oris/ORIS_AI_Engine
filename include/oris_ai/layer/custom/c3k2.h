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
#include "oris_ai/layer/custom/c3k.h"

namespace oris_ai {

/**
 * @class C3k2
 * @brief C2f-style layer stacking multiple C3k blocks for YOLOv11 models.
 */
template <typename T>
class C3k2 : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize the C3k2 layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param shortcut Indicates whether to use shortcut connections inside the C3k blocks.
     * @param c3k_count The number of C3k blocks to create.
     */
    C3k2(const std::string& layer_name, Device target_device, bool shortcut, int c3k_count);

    /**
     * @brief Default destructor for the C3k2 class.
     */
    ~C3k2() = default;

    /**
     * @brief Initializes the C3k2 layer with the given TorchLayer data.
     * @param c3k2_layers Vector of TorchLayer objects for the convolutions and C3k blocks.
     */
    void InitC3k2(const std::vector<TorchLayer>& c3k2_layers);

    /**
     * @brief Retrieves the output tensor from the final convolutional layer in the C3k2 layer.
     * @return The output tensor from the final convolutional layer.
     */
    inline Tensor<T>* GetOutputTensor() override { return c3k2_cv2_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the C3k2 layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { c3k2_cv2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> c3k2_cv1_;  // Initial 1x1 convolution layer
    std::unique_ptr<Split<T>> c3k2_split_;  // Split layer for C2f pattern
    std::vector<std::unique_ptr<C3k<T>>> c3k2_blocks_;  // Vector of C3k blocks
    std::unique_ptr<Concat<T>> c3k2_concat_;  // Layer to concatenate before the final convolution
    std::unique_ptr<Convolution<T>> c3k2_cv2_;  // Final 1x1 convolution layer

    bool shortcut_;           // Indicates if shortcut (add) operation is enabled
    int c3k_count_;           // Number of C3k blocks
    Tensor<T>* c3k2_split_0_;  // First split output tensor
    Tensor<T>* c3k2_split_1_;  // Second split output tensor
};

} // namespace oris_ai

