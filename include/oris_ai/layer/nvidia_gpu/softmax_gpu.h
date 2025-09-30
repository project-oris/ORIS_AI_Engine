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

#include "oris_ai/layer/softmax.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class SoftmaxGPU
 * @brief NVIDIA GPU-specific implementation of the Softmax layer.
 * 
 * This class inherits from Softmax and implements the forward pass
 * specifically for NVIDIA GPU execution using cuDNN.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class SoftmaxGPU : public Softmax<T> {
  public:
    /**
     * @brief Constructor to initialize a SoftmaxGPU layer.
     * @param layer_name The name of the layer.
     */
    explicit SoftmaxGPU(const std::string& layer_name)
      : Softmax<T>(layer_name),
        cudnn_handle_(nullptr),
        tensor_desc_(nullptr) {}

    /**
     * @brief Destructor for the SoftmaxGPU class.
     */
    ~SoftmaxGPU();

    /**
     * @brief Initializes the Softmax layer on the NVIDIA GPU.
     * 
     * This function implements the virtual InitSoftmax method defined in the Softmax 
     * base class, configuring the Softmax layer for efficient execution on the NVIDIA GPU.
     */
    void InitSoftmax(const size_t softmax_dim, bool use_view_input_shape) override;

    /**
     * @brief Performs the forward pass of the Softmax layer using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Softmax` 
     * class, providing a NVIDIA GPU-specific implementation for the Softmax operation.
     */
    void Forward() override;

  private:
    cudnnHandle_t cudnn_handle_;           // cuDNN handle for GPU operations
    cudnnTensorDescriptor_t tensor_desc_;  // Tensor descriptor for cuDNN operations
};

} // namespace oris_ai
