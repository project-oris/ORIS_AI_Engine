/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute
 *(ETRI)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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

#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/depthwise_convolution.h"
#include "oris_ai/layer/elementwise.h"
#include "oris_ai/layer/matmul.h"
#include "oris_ai/layer/split.h"
#include "oris_ai/layer/softmax.h"

namespace oris_ai {

/**
 * @class Attention
 * @brief Attention block used inside PSABlock. Consists of qkv, proj and pe
 *        convolutions. Currently qkv and positional-encoding convolutions are
 *        implemented.
 * 
 * This class implements a multi-head attention mechanism with query, key, and value
 * processing through convolutional operations and tensor splitting.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T> class Attention : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize an Attention layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU).
     */
    Attention(const std::string &layer_name, Device target_device);

    /**
     * @brief Destructor for the Attention class.
     */
    ~Attention() = default;

    /**
     * @brief Initializes the Attention layer with the given TorchLayer data.
     *
     * @param qkv_conv The TorchLayer object for the qkv convolution.
     * @param qkv_act The TorchLayer object containing the activation type.
     * @param pe_conv The TorchLayer object for the positional-encoding
     * convolution.
     * @param pe_act The TorchLayer object containing the activation type.
     * @param proj_conv The TorchLayer object for the projection convolution.
     * @param proj_act The TorchLayer object containing the activation type.
     * @param attn_ratio The attention ratio parameter.
     * @param num_heads The number of attention heads.
     */
    void InitAttention(const TorchLayer &qkv_conv, const TorchLayer &qkv_act,
                      const TorchLayer &pe_conv, const TorchLayer &pe_act,
                      const TorchLayer &proj_conv, const TorchLayer &proj_act,
                      float attn_ratio, size_t num_heads);

    /*
     * @brief Retrieves the output tensor from the Attention layer.
     * @return The output tensor from the Attention layer.
     */
    inline Tensor<T>* GetOutputTensor() override { return proj_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass of the Attention layer.
     */
    void Forward() override;

  private:
    std::unique_ptr<Convolution<T>> qkv_;  // Query, key, value convolution layer
    std::unique_ptr<Split<T>> qkv_split_;  // Split layer used to divide qkv tensor into q, k and v parts
    std::unique_ptr<MatMul<T>> qk_matmul_; // MatMul layer used to perform q^T @ k
    std::unique_ptr<Softmax<T>> softmax_;  // Softmax layer for attention scores
    std::unique_ptr<MatMul<T>> v_attn_matmul_; // MatMul layer used to perform v @ attn^T
    std::unique_ptr<DepthwiseConvolution<T>> pe_;  // Positional encoding depthwise convolution layer
    std::unique_ptr<ElementWise<T>> elementwise_; // Element-wise addition layer
    std::unique_ptr<Convolution<T>> proj_;        // Projection convolution layer

    // Pointers to the tensors for q, k and v.
    Tensor<T> *q_;  // Query tensor
    Tensor<T> *k_;  // Key tensor
    Tensor<T> *v_;  // Value tensor
};

} // namespace oris_ai
