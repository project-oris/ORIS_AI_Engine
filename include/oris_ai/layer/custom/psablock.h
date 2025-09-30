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

#include <memory>
#include <vector>

#include "oris_ai/layer/custom/attention.h"
#include "oris_ai/layer/custom/bottleneck.h"
#include "oris_ai/layer/elementwise.h"

namespace oris_ai {

/**
 * @class PSABlock
 * @brief PSABlock that begins with an Attention layer.
 * 
 * This class encapsulates an Attention layer and manages forward pass and tensor 
 * operations for PSA (Point-wise Spatial Attention) functionality.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class PSABlock : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a PSABlock layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU).
     */
    PSABlock(const std::string& layer_name, Device target_device);

    /**
     * @brief Destructor for the PSABlock class.
     */
    ~PSABlock() = default;

    /**
     * @brief Initializes the PSABlock layer with the given TorchLayer data.
     *
     * @param psablock_layers Vector of TorchLayer objects for PSABlock
     *        initialization. Expected order is qkv conv, qkv activation,
     *        positional-encoding depthwise conv and its activation,
     *        followed by projection conv and its activation.
     * @param attn_ratio The attention ratio parameter for the Attention layer.
     * @param num_heads The number of attention heads for the Attention layer.
     */
    void InitPSABlock(const std::vector<TorchLayer> &psablock_layers,
                      float attn_ratio, size_t num_heads);

    /**
     * @brief Retrieves the output tensor from the PSABlock.
     * @return The output tensor after attention and feed-forward network.
     */
    inline Tensor<T>* GetOutputTensor() override {
      return ffn_->GetOutputTensor();
    }

    /**
     * @brief Perform the forward pass of the PSABlock layer.
     */
    void Forward() override;

  private:
    std::unique_ptr<Attention<T>> attention_;   // Attention layer for PSA functionality
    std::unique_ptr<ElementWise<T>> elementwise_;  // Element-wise addition for residual
    std::unique_ptr<Bottleneck<T>> ffn_;  // Feed-forward network implemented via Bottleneck
};

} // namespace oris_ai

