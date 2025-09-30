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

#include "oris_ai/layer/convolution.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class ConvolutionGPU
 * @brief A class that implements convolution operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing convolution 
 * operations, inheriting from the base `Convolution` class. It overrides the 
 * `Forward` method to perform the forward pass of the convolution layer 
 * using NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionGPU : public Convolution<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionGPU layer.
     * @param name The name of the layer.
     */
    ConvolutionGPU(const std::string& layer_name)
      : Convolution<T>(layer_name),
        work_space_(nullptr),
        size_work_space_(0) {}

    /**
     * @brief Destructor for the ConvolutionGPU class.
     */
    ~ConvolutionGPU();

    /**
     * @brief Overrides the virtual InitConvolution function for NVIDIA GPU-specific
     * initialization with activation.
     * 
     * This function implements the virtual InitConvolution method defined in the Convolution 
     * base class, configuring the convolution layer with the provided parameters and activation function for
     * efficient execution on the NVIDIA GPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     * @param act The TorchActivation object containing activation function parameters.
     */
    void InitConvolution(const TorchConv2d& conv2d_params, const TorchActivation& act) override;

    /**
     * @brief Overrides the virtual InitConvolution function for NVIDIA GPU-specific
     * initialization.
     * 
     * This function implements the virtual InitConvolution method defined in the Convolution 
     * base class, configuring the convolution layer with the provided parameters for
     * efficient execution on the NVIDIA GPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void InitConvolution(const TorchConv2d& conv2d_params) override;

    /**
     * @brief Performs the forward pass of the Convolution layer using NVIDIA GPU.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Convolution` 
     * class, providing a NVIDIA GPU-specific implementation for the convolution operation.
     */
    void Forward() override;

  private:
    void CudnnSetup(const TorchConv2d& conv2d_params);

    // Common variables
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t bias_desc_;
    // cudnnActivationDescriptor_t act_desc_;
    void* work_space_;
    size_t size_work_space_;

    cudnnConvolutionFwdAlgo_t conv_algo_;
};

} // namespace oris_ai
