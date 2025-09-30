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

namespace oris_ai {

/**
 * @class PaddingINT8
 * @brief Represents a padding layer in a neural network for int8 quantization.
 * 
 * This class defines a padding layer that pads the input channels to the nearest
 * multiple of 16 for better performance in int8 operations.
 * Input tensor format is NHWC (batch, height, width, channels).
 * 
 * @tparam T The data type for the layer operations (int8_t).
 */
class PaddingINT8 : public HiddenLayerAbstract<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a PaddingINT8 layer.
     * @param name The name of the layer.
     */
    PaddingINT8(const std::string& layer_name)
      : HiddenLayerAbstract<int8_t>(layer_name) {}

    /**
     * @brief Destructor for the PaddingINT8 class.
     */
    ~PaddingINT8() = default;

    /**
     * @brief Initializes the padding layer.
     * 
     * This function calculates the output channels as the nearest multiple of 16
     * from the input channels and prepares the output tensor.
     */
    void InitPadding();

    /**
     * @brief Pure virtual function to perform the forward pass of the padding layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the padding operation.
     */
    void Forward() override;
};

} // namespace oris_ai 