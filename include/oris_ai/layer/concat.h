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
 * @class Concat
 * @brief Represents a concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer, which concatenates multiple input tensors 
 * along a specified dimension. The Concat class can optionally apply a Permute operation 
 * to adjust the shape of each input tensor before concatenation.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Concat : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a concatenation layer.
     * @param name The name of the layer.
     */
    Concat(const std::string& layer_name)
      : HiddenLayerAbstract<T>(layer_name),
        concat_dim_(1),
        use_view_input_shape_(false) {}

    /**
     * @brief Destructor for the Concat class.
     */
    ~Concat() = default;

    /**
     * @brief Initializes the concat layer with parameters from a TorchConcat object.
     * 
     * @param concat_params The TorchConcat object containing concat parameters.
     */
    virtual void InitConcat(const TorchConcat& concat_params);

    /**
     * @brief Initializes the concat layer with the specified concatenation dimension.
     * @param concat_dim The dimension along which the tensors are concatenated.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input
     * tensor.
     */
    virtual void InitConcat(const size_t concat_dim, bool use_view_input_shape = false);

    /**
     * @brief Pure virtual function to perform the forward pass of the concat layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the concat operation.
     */
    virtual void Forward() = 0;

  protected:
    /**
     * @brief Configures and initializes the output tensor for the concat layer.
     * @note This function is automatically called by InitConcat() after setting the 
     * concatenation dimension.
     */
    void ConcatSetup();

    size_t concat_dim_; // The dimension along which concatenation occurs
    std::vector<size_t> output_shape_; // Shape of the output tensor after concatenation

    bool use_view_input_shape_; // Flag indicating whether to use view input mode for tensor operations
};

} // namespace oris_ai
