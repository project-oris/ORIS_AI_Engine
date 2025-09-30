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

#include "oris_ai/layer/int8/concat_int8.h"
#include "oris_ai/layer/nvidia_gpu/concat_gpu.h"

namespace oris_ai {

/**
 * @class ConcatGPU
 * @brief Represents a NVIDIA GPU-based concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer that operates on NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <>
class ConcatGPU<int8_t> : public Concat<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a ConcatGPU layer.
     * @param name The name of the layer.
     */
    ConcatGPU(const std::string& layer_name)
      : Concat<int8_t>(layer_name),
        input_tensors_gpu_(nullptr),
        input_channels_per_input_gpu_(nullptr),
        input_channel_offset_gpu_(nullptr) {}

    ~ConcatGPU();

    /**
     * @brief Initializes the concat layer with parameters from a TorchConcat object.
     * 
     * @param concat_params The TorchConcat object containing concat parameters.
     */
    void InitConcat(const TorchConcat& concat_params) override;

    /**
     * @brief Initializes the concat layer with the specified concatenation dimension.
     * @param concat_dim The dimension along which the tensors are concatenated.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input
     * tensor.
     */
    void InitConcat(const size_t concat_dim, bool use_view_input_shape = false) override;

    /**
     * @brief Performs the forward pass of the Concat layer using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Concat` 
     * class, providing a NVIDIA GPU-specific implementation for the Concat operation.
     */
    void Forward() override;
    
  private:
    void ConcatGPUSetup();
    
    const int8_t** input_tensors_gpu_;
    int* input_channels_per_input_gpu_;
    int* input_channel_offset_gpu_;
    float* output_scales_gpu_;  // GPU memory for output scale
};

} // namespace oris_ai
