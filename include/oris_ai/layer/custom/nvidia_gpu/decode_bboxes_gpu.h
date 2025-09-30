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

#include "oris_ai/layer/custom/decode_bboxes.h"

namespace oris_ai {

/**
 * @class DecodeBboxesGPU
 * @brief Represents a CPU-specific implementation of the DecodeBboxes layer.
 * 
 * This class defines the CPU-specific version of the DecodeBboxes layer, inheriting from
 * the base DecodeBboxes class and implementing the forward pass for NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DecodeBboxesGPU : public DecodeBboxes<T> {
  public:
    /**
     * @brief Constructor to initialize a DecodeBboxesGPU layer.
     * @param layer_name The name of the layer.
     */
    DecodeBboxesGPU(const std::string& layer_name) : DecodeBboxes<T>(layer_name, Device::GPU) {}

    /**
     * @brief Default destructor for the DecodeBboxesGPU class.
     */
    ~DecodeBboxesGPU() = default;

    /**
     * @brief Performs the forward pass of the DecodeBboxes layer on the NVIDIA GPU.
     * 
     * This function implements the forward pass for the DecodeBboxes layer using NVIDIA GPU
     * operations.
     */
    void Forward() override;
};

} // namespace oris_ai
