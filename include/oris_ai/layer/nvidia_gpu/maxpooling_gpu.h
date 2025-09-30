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

#include "oris_ai/layer/maxpooling.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class MaxPoolingCPU
 * @brief A class that implements max pooling operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing max pooling
 * operations, inheriting from the base `MaxPooling` class. It overrides the 
 * `Forward` method to execute the forward pass of the max pooling layer 
 * using NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class MaxPoolingGPU : public MaxPooling<T> {
  public:
    /**
     * @brief Constructor to initialize a MaxPoolingGPU layer.
     */
    MaxPoolingGPU(const std::string& layer_name)
      : MaxPooling<T>(layer_name) {}

    /**
     * @brief Destructor for the MaxPoolingCPU class.
     */
    ~MaxPoolingGPU();

    /**
     * @brief Overrides the virtual InitMaxPooling function for NVIDIA GPU-specific
     * initialization.
     * 
     * This function implements the virtual InitMaxPooling method defined in the pooling
     * base class, configuring the max pooling layer with the provided parameters for
     * efficient execution on the NVIDIA GPU.
     * 
     * @param maxpool2d_params The TorchMaxPool2d object containing max pooling parameters.
     */
    void InitMaxPooling(const TorchMaxPool2d& maxpool2d_params) override;

    /**
     * @brief Performs the forward pass of the MaxPoolong layer using NVIDIA GPU.
     * 
     * This function overrides the pure virtual `Forward` method from the base `MaxPoolong` 
     * class, providing a NVIDIA GPU-specific implementation for the MaxPoolong operation.
     */
    void Forward() override;

  private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnPoolingDescriptor_t pool_desc_;
};

} // namespace oris_ai
