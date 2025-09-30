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

#include "oris_ai/layer/convolution_transpose.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class ConvolutionTransposeGPU
 * @brief A class that implements transposed convolution operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing transposed convolution 
 * operations, inheriting from the base `ConvolutionTranspose` class. It overrides the 
 * `Forward` method to perform the forward pass of the transposed convolution layer 
 * using NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionTransposeGPU : public ConvolutionTranspose<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionTransposeGPU layer.
     * @param name The name of the layer.
     */
    ConvolutionTransposeGPU(const std::string& layer_name) 
      : ConvolutionTranspose<T>(layer_name),
        work_space_(nullptr),
        size_work_space_(0) {}

    /**
     * @brief Destructor for the ConvolutionTransposeGPU class.
     */
    ~ConvolutionTransposeGPU();

    /**
     * @brief Initializes the transposed convolution layer with the given parameters.
     * 
     * This function implements the initialization for NVIDIA GPU-specific transposed convolution,
     * configuring the layer with the provided parameters for efficient execution on the GPU.
     * 
     * @param conv_transpose_params The TorchConvTranspose2d object containing parameters.
     */
    void InitConvolutionTranspose(const TorchConvTranspose2d& conv_transpose_params) override;

    /**
     * @brief Performs the forward pass of the transposed convolution layer using NVIDIA GPU.
     * 
     * This function overrides the pure virtual `Forward` method from the base `ConvolutionTranspose` 
     * class, providing a NVIDIA GPU-specific implementation for the transposed convolution operation.
     */
    void Forward() override;

  private:
    /**
     * @brief Sets up cuDNN descriptors and finds optimal algorithm for transposed convolution.
     * 
     * This function initializes all necessary cuDNN descriptors and finds the best algorithm
     * for performing transposed convolution on the GPU.
     * 
     * @param conv_transpose_params The TorchConvTranspose2d object containing parameters.
     */
    void CudnnSetup(const TorchConvTranspose2d& conv_transpose_params);

    // Common variables
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    void* work_space_;
    size_t size_work_space_;

    cudnnConvolutionBwdDataAlgo_t conv_algo_;
};

} // namespace oris_ai 