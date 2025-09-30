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
 * @class Dequant
 * @brief Represents a dequantization layer in a neural network for int8 tensors.
 * 
 * This class defines a dequantization layer that converts int8 tensors to float tensors.
 * Input tensor format is NHWC  (batch, height, width, channels).
 * Output tensor format is NCHW (batch, channels, height, width).
 */
class Dequant : public LayerAbstract<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a Dequant layer.
     * @param name The name of the layer.
     */
    Dequant(const std::string& layer_name)
      : LayerAbstract<int8_t>(layer_name) {}

    /**
     * @brief Destructor for the Dequant class.
     */
    ~Dequant() = default;

    /**
     * @brief Initializes the dequantization layer.
     * 
     * This function configures the output shapes for the dequantized tensors and
     * preparing output tensors to hold the results.
     */
    void InitDequant();

    /**
     * @brief Returns a pointer to the dequantized tensor at the specified index.
     * 
     * This function allows access to individual dequantized tensors that result from the 
     * dequantization process.
     * 
     * @param index The index of the desired dequantized tensor.
     * @return A pointer to the requested output tensor.
     */
    inline Tensor<float>* GetFloatTensor(size_t index) const { return float_tensors_.at(index).get(); }

    /**
     * @brief Pure virtual function to perform the forward pass of the dequantization layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the dequantization operation.
     */
    void Forward() override;

  protected:
    std::vector<std::vector<size_t>> output_shapes_; // Array of output shapes for each tensor

    std::vector<std::unique_ptr<Tensor<float>>> float_tensors_; // Unique pointers to the resulting dequantized tensors
};

} // namespace oris_ai
