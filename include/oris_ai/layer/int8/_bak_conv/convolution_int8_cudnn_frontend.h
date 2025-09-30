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

#include "oris_ai/layer/int8/convolution_int8.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"
#include "cudnn-frontend/include/cudnn_frontend.h"

namespace oris_ai {

/**
 * @class Convolution
 * @brief Represents a convolutional layer in a neural network.
 * 
 * This class defines a generic convolutional layer, including its parameters,
 * activation functions, and required buffers.
 */
class ConvolutionINT8Frontend : public Convolution<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionINT8Frontend layer.
     * @param name The name of the layer.
     */
    ConvolutionINT8Frontend(const std::string& layer_name)
      : Convolution<int8_t>(layer_name), work_space_(nullptr), size_work_space_(0) {}

    /**
     * @brief Destructor for the ConvolutionINT8Frontend class.
     */
    ~ConvolutionINT8Frontend() = default;

    void InitConvolution(const QuantizedTorchConv2d& conv2d_params, const QuantizedTorchActivation& act) override;

    void InitConvolution(const QuantizedTorchConv2d& conv2d_params) override;

    void Forward() override;

  private:
    void CudnnSetup();
    void GenerateStrides(const int64_t* dim, int64_t* stride);

    // cudnn variables
    cudnnHandle_t cudnn_handle_;

    // cudnn frontend variables
    std::unique_ptr<cudnn_frontend::ExecutionPlan> plan_;
    std::unique_ptr<cudnn_frontend::VariantPack> variant_pack_;
    void* work_space_;
    size_t size_work_space_;
};

} // namespace oris_ai
