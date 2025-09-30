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

#include "oris_ai/layer/upsample.h"

namespace oris_ai {

/**
 * @class UpsampleGPU
 * @brief Implements the forward pass of an upsampling layer on the NVIDIA GPU.
 * 
 * This class inherits from Upsample and implements the forward pass
 * specifically for NVIDIA GPU execution.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class UpsampleGPU : public Upsample<T> {
  public:
    /**
     * @brief Constructor to initialize the UpsampleGPU layer.
     * @param name The name of the layer.
     */
    UpsampleGPU(const std::string& layer_name)
      : Upsample<T>(layer_name) {}

    /**
     * @brief Destructor for the UpsampleGPU class.
     */
    ~UpsampleGPU() {}

    /**
     * @brief Performs the forward pass for the upsample operation on NVIDIA GPU.
     */
    void Forward() override;
};

} // namespace oris_ai
