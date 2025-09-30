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

namespace oris_ai {

/**
 * @class ConvolutionCPU
 * @brief A class that implements convolution operations on a CPU.
 * 
 * This class provides a CPU-specific implementation for performing convolution 
 * operations, inheriting from the base `Convolution` class. It overrides the 
 * `Forward` method to perform the forward pass of the convolution layer 
 * using CPU resources.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionCPU : public Convolution<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionCPU layer.
     * @param name The name of the layer.
     */
    ConvolutionCPU(const std::string& layer_name)
      : Convolution<T>(layer_name),
        is_1x1_conv_(false) {}

    /**
     * @brief Destructor for the ConvolutionCPU class.
     */
    ~ConvolutionCPU() = default;

    void InitConvolution(const TorchConv2d& conv2d_params, const TorchActivation& act) override;

    /**
     * @brief Overrides the virtual InitConvolution function for CPU-specific initialization.
     * 
     * This function implements the virtual InitConvolution method defined in the Convolution 
     * base class, configuring the convolution layer with the provided parameters for
     * efficient execution on the CPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void InitConvolution(const TorchConv2d& conv2d_params) override;

    /**
     * @brief Performs the forward pass of the Convolution layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Convolution` 
     * class, providing a CPU-specific implementation for the convolution operation.
     */
    void Forward() override;

  private:
    /**
     * @brief Configures the im2col buffer for the convolution layer.
     * 
     * This function initializes the im2col buffer for the convolution layer, which is used 
     * to store the intermediate results of the convolution operation.
     * 
     * @param conv2d_params The TorchConv2d object containing the parameters.
     */
    void SetIm2Col(const TorchConv2d& conv2d_params);

    /**
     * @brief Performs the im2col operation for convolution.
     * 
     * The im2col (image to column) operation is used to transform the input data 
     * into a format suitable for matrix multiplication in the convolution process.
     * 
     * @param input_data Pointer to the input data.
     * @param col_data Pointer to the column data buffer.
     * @param input_h Height of the input data.
     * @param input_w Width of the input data.
     * @param kernel_h Height of the convolution kernel.
     * @param kernel_w Width of the convolution kernel.
     * @param pad_h Padding size along the height.
     * @param pad_w Padding size along the width.
     * @param stride_h Stride size along the height.
     * @param stride_w Stride size along the width.
     */
    void Im2Col(const T* input_data, T* col_data, int input_h, int input_w,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w);

    std::unique_ptr<Tensor<T>> im2col_buffer_; // A unique pointer to the buffer used for im2col operations.
    bool is_1x1_conv_;  // Flag indicating whether this is a 1x1 convolution (special optimization case).
};

} // namespace oris_ai
