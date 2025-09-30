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

#include "oris_ai/layer/layer.h"
#include "oris_ai/layer/matmul.h"
#include "oris_ai/layer/softmax.h"
#include "oris_ai/layer/transpose.h"

namespace oris_ai {

/**
 * @class DFL
 * @brief Represents a specialized Deep Feature Loss (DFL) layer.
 * 
 * This class defines a DFL layer, which is a simple 1x1 convolution without bias
 * or activation. The DFL layer is intended for lightweight transformations.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DFL : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Default constructor for the DFL layer.
     * @param layer_name The name of the layer.
     * @param target_device The device (CPU or GPU) on which the layer will operate.
     */
    DFL(const std::string& layer_name, Device target_device);

    /**
     * @brief Default destructor for the DFL class.
     */
    ~DFL() = default;

    /**
     * @brief Initializes the DFL layer with parameters from a TorchConv2d object.
     * 
     * This function sets up the DFL layer by configuring the weight tensor, input/output
     * channels, and output dimensions based on the provided convolution parameters.
     * The DFL layer is composed of sub-layers (Transpose, Softmax, MatMul) that handle
     * the actual computation.
     * 
     * @param conv2d_params The TorchConv2d object containing the convolution parameters.
     */
    void InitDFL(const TorchConv2d& conv2d_params);

    /**
     * @brief Performs the forward pass of the DFL layer.
     * 
     * This function executes the DFL operation by coordinating the sub-layers:
     * first transposing the input boxes, then applying softmax, and finally
     * performing matrix multiplication with the learned weights.
     */
    void Forward();

    /**
     * @brief Returns the output tensor of the DFL layer.
     */
    inline Tensor<T>* GetOutputTensor() override { return matmul_->GetOutputTensor(); }

  protected:
    std::unique_ptr<Tensor<T>> weight_; // Unique pointer to the weight tensor for the layer
    size_t in_channels_;                // Number of input channels
    size_t out_channels_;               // Number of output channels (1x1 convolution)
    size_t output_height_;              // Height of the output tensor
    size_t output_width_;               // Width of the output tensor

    std::unique_ptr<Transpose<T>> trans_boxes_;
    std::unique_ptr<Softmax<T>> softmax_;
    std::unique_ptr<MatMul<T>> matmul_;
};

} // namespace oris_ai
